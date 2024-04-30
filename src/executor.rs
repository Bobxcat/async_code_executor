//!
//! A stack based assembly-like language with async code callable from all functions and garbage collected values,
//! And expirement in writing an interpreter
//!

use std::{
    alloc::{GlobalAlloc, Layout},
    borrow::Borrow,
    collections::{HashMap, VecDeque},
    fmt::{Debug, Display},
    io::Write,
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZeroUsize,
    ops::Deref,
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, RwLock,
    },
    time::{Duration, Instant},
};

use crate::{
    build_status::{BuildStatus, Building, Built, FinishBuildingCtx},
    function::{CodePoint, FuncName, Function},
    gc::{DefaultProgramAlloc, GcPtr, ProgramGC},
    types::{primitives::NumTy, CustomTyName, CustomType, Type, TypeCtx, ValStack, ValStackEntry},
};

/// An allocator that can be used to allocate values used by the program
///
/// By default, this uses [System](std::alloc::System)
pub trait ProgramAlloc: GlobalAlloc + 'static {
    #[allow(unused)]
    fn alloc_type<T>(&self) -> &mut MaybeUninit<T> {
        unsafe { &mut *(GlobalAlloc::alloc(self, Layout::new::<T>()).cast::<MaybeUninit<T>>()) }
    }
    #[allow(unused)]
    fn alloc_zeroed_type<T>(&self) -> &mut MaybeUninit<T> {
        unsafe {
            &mut *(GlobalAlloc::alloc_zeroed(self, Layout::new::<T>()).cast::<MaybeUninit<T>>())
        }
    }
    unsafe fn dealloc_type<T>(&self, p: *mut T) {
        unsafe { GlobalAlloc::dealloc(self, p.cast(), Layout::new::<T>()) }
    }
}

impl<A: GlobalAlloc + 'static> ProgramAlloc for A {}

pub trait Maybe<T>: Sized + Copy {
    fn filled(v: T) -> Filled<T> {
        Filled { v }
    }
    fn get_or_with(self, f: impl FnOnce() -> T) -> T;
    fn get_or(self, t: T) -> T {
        self.get_or_with(|| t)
    }
}

impl<T: Copy> Maybe<T> for Filled<T> {
    fn get_or_with(self, _: impl FnOnce() -> T) -> T {
        self.get()
    }
}
impl<T: Copy> Maybe<T> for Empty<T> {
    fn get_or_with(self, f: impl FnOnce() -> T) -> T {
        f()
    }
}

pub trait IsFilled {}

impl<T> IsFilled for Filled<T> {}

pub trait IsEmpty {}

impl<T> IsEmpty for Empty<T> {}

#[derive(Debug, Clone, Copy)]
pub struct Filled<T> {
    v: T,
}

impl<T> Filled<T> {
    fn get(self) -> T {
        self.v
    }
}

impl<T> From<T> for Filled<T> {
    fn from(value: T) -> Self {
        Self { v: value }
    }
}

impl<T> Deref for Filled<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> Borrow<T> for Filled<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T> AsRef<T> for Filled<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

/// A field that hasn't been filled in yet
#[derive(Debug, Clone, Copy)]
pub struct Empty<T> {
    _p: PhantomData<T>,
}

impl<T> Default for Empty<T> {
    fn default() -> Self {
        Self {
            _p: Default::default(),
        }
    }
}

impl<T> Empty<T> {
    fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub struct ExecutorBuilder<
    Alloc: ProgramAlloc = DefaultProgramAlloc,
    MaybeNumThreads: Maybe<usize> = Empty<usize>,
> {
    program_alloc: Alloc,
    num_threads: MaybeNumThreads,
    functions: HashMap<FuncName, Function<Building>>,
    customs: HashMap<CustomTyName, CustomType<Building>>,
    _p: PhantomData<()>,
}

impl ExecutorBuilder {
    pub fn new() -> Self {
        Self {
            program_alloc: DefaultProgramAlloc::new(Some(1_000_000.try_into().unwrap())),
            num_threads: Empty::new(),
            functions: HashMap::new(),
            customs: HashMap::new(),
            _p: PhantomData,
        }
    }
}

impl<Alloc: ProgramAlloc, MaybeNumThreads: Maybe<usize>> ExecutorBuilder<Alloc, MaybeNumThreads> {
    pub fn set_num_threads(self, num_threads: usize) -> ExecutorBuilder<Alloc, Filled<usize>> {
        ExecutorBuilder {
            program_alloc: self.program_alloc,
            num_threads: num_threads.into(),
            functions: self.functions,
            customs: self.customs,
            _p: PhantomData,
        }
    }
    pub fn set_allocator<A: ProgramAlloc>(self, a: A) -> ExecutorBuilder<A, MaybeNumThreads> {
        ExecutorBuilder {
            program_alloc: a,
            num_threads: self.num_threads,
            functions: self.functions,
            customs: self.customs,
            _p: PhantomData,
        }
    }
    pub fn insert_function(mut self, f: Function<Building>) -> Self {
        let id = f.id.clone();
        if let Some(_) = self.functions.insert(id.clone(), f) {
            panic!("Cannot insert function {:?}, already exists", id)
        }
        self
    }
    pub fn insert_custom(mut self, ty: CustomType<Building>) -> Self {
        let id = ty.id.clone();
        if let Some(_) = self.customs.insert(id.clone(), ty) {
            panic!("Cannot insert function {:?}, already exists", id)
        }
        self
    }
    fn into_executor(self) -> Executor<Alloc> {
        let finish_ctx = FinishBuildingCtx::from_names(
            self.customs.iter().map(|(n, _)| n.clone()),
            self.functions.iter().map(|(n, _)| n.clone()),
        );

        let type_ctx = TypeCtx::build(&finish_ctx, self.customs);

        let mut functions = self
            .functions
            .iter()
            .map(|(_name, func)| func.finish(&finish_ctx))
            .collect::<Vec<_>>();
        functions.sort_unstable_by_key(|f| f.id);

        let statics = ExecutorStatics {
            gc: ProgramGC::new(self.program_alloc),
            functions,
            customs: type_ctx,
        };

        Executor {
            statics: Arc::new(statics),
            idle_routines: RwLock::default(),
        }
    }
    pub fn run(self) {
        let num_threads = self.num_threads.get_or(10);
        self.into_executor().run(ExecutorRunCfg { num_threads })
    }
}

#[derive(Debug)]
struct ExecutorRunCfg {
    num_threads: usize,
}

/// The static data in an executor which is immutable
#[derive(Debug)]
struct ExecutorStatics<A: ProgramAlloc> {
    gc: ProgramGC<A>,
    functions: Vec<Function<Built>>,
    customs: TypeCtx,
}

#[derive(Debug)]
enum TypeMismatchErr<B: BuildStatus> {
    TraitNotImplemented { ty: Type<B>, needs_trait: String },
}

impl<B: BuildStatus> Display for TypeMismatchErr<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}

impl<B: BuildStatus> std::error::Error for TypeMismatchErr<B> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CodePointIdx(pub usize);

#[derive(Debug)]
struct Routine<A: ProgramAlloc> {
    operand_stack: ValStack,
    call_stack: Vec<(*const Function<Built>, Option<CodePointIdx>)>,
    exec: Arc<ExecutorStatics<A>>,
}

unsafe impl<A: ProgramAlloc> Send for Routine<A> {}

#[derive(Debug)]
enum RoutineRunnerYieldReason {
    /// This routine is awaiting an async primitive, and will yield for now
    Awaiting,
    /// This routine ran for the given duration and timed out
    Timeout(Duration),
}

#[derive(Debug)]
enum RoutineRunnerResult<A: ProgramAlloc> {
    Finished {
        next_routine: oneshot::Sender<Routine<A>>,
    },
    Yield {
        yielded: Routine<A>,
        reason: RoutineRunnerYieldReason,
        next_routine: oneshot::Sender<Routine<A>>,
    },
}

fn run_routine<A: ProgramAlloc>(
    mut routine: Routine<A>,
) -> (
    RoutineRunnerResult<A>,
    Option<oneshot::Receiver<Routine<A>>>,
) {
    macro_rules! return_finished {
        () => {
            let (tx, rx) = oneshot::channel();
            return (RoutineRunnerResult::Finished { next_routine: tx }, Some(rx));
        };
    }

    let start = Instant::now();

    // Execute code
    loop {
        let Some((curr_func, curr_code_point_idx)) = routine.call_stack.last() else {
            return_finished!();
        };
        let curr_func = unsafe { &**curr_func };
        let Some(curr_code_point) = curr_func.program.get(curr_code_point_idx.unwrap().0) else {
            return_finished!();
        };

        let ctx = &routine.exec.customs;

        match curr_code_point {
            CodePoint::Add(ty) => {
                let b = routine
                    .operand_stack
                    .pop_ret(ctx)
                    .expect("Failed `Add`: Operand stack empty");
                let a = routine
                    .operand_stack
                    .pop_ret(ctx)
                    .expect("Failed `Add`: Operand stack empty");

                assert!(a.ty() == b.ty(), "Failed `Add`: Type mismatch");
                assert!(a.ty() == *ty, "Failed `Add`: Type mismatch");
            }
            CodePoint::Literal(lit) => routine.operand_stack.push_literal(lit.clone(), ctx),
            _ => todo!(),
        }

        let code_ptr = &mut routine.call_stack.last_mut().unwrap().1;
        match code_ptr {
            Some(n) => n.0 += 1,
            None => *code_ptr = Some(CodePointIdx(0)),
        }
    }
}

fn routine_runner<A: ProgramAlloc>(tx: Sender<RoutineRunnerResult<A>>) {
    let (first_routine_tx, first_routine_rx) = oneshot::channel();
    tx.send(RoutineRunnerResult::Finished {
        next_routine: first_routine_tx,
    })
    .unwrap();
    let mut routine: Routine<A> = first_routine_rx.recv().unwrap();

    loop {
        let (res, next_routine) = run_routine(routine);
        tx.send(res).unwrap();

        let Some(r) = next_routine else {
            println!("Closed down `routine_runner` since `run_routine` returned a `None` for the next routine");
            std::io::stdout().flush().unwrap();
            return;
        };

        routine = r.recv().unwrap();
    }
}

/// The executor, which is shared between threads and should **never** be mutably referenced (once `run` is called)
#[derive(Debug)]
struct Executor<A: ProgramAlloc> {
    statics: Arc<ExecutorStatics<A>>,
    idle_routines: RwLock<VecDeque<Routine<A>>>,
}

impl<A: ProgramAlloc> Executor<A> {
    // fn spawn_routine(&self, f: FuncName, starting_stack: Stack) {
    //     let r = Routine {
    //         data_stack: starting_stack,
    //         call_stack: vec![(&self.data.functions[&f], Some(CodePointIdx(0)))],
    //         exec: self,
    //     };
    //     let mut idle_routines = self.data.idle_routines.write().unwrap();
    //     idle_routines.push_back(r)
    // }
    pub fn run(&self, cfg: ExecutorRunCfg) {
        //!
        //! This function dispatches routines to a specified number of worker threads.
        //! There is a queue of idle routines which worker threads pull from. Newly spawned routines are put at the back of this queue.
        //!
        //! Each worker thread will run their routine for a while, until either yielding or working for a specified time.
        //! When a worker thread gives their routine back to this function, their previous routine is placed at the back of the routine queue.
        //!

        todo!();

        let mut routine_returns: Vec<Receiver<RoutineRunnerResult<A>>> = vec![];

        for _ in 0..cfg.num_threads {
            let (tx, rx) = mpsc::channel();

            std::thread::spawn(move || routine_runner::<A>(tx));
            routine_returns.push(rx);
        }

        let entrypoint = FuncName::new("main");
        // self.spawn_routine(entrypoint, Stack::empty());

        loop {
            let Some(res) = routine_returns.iter().find_map(|r| r.try_recv().ok()) else {
                continue;
            };

            let next = loop {
                let mut idle_routines = self.idle_routines.write().unwrap();
                if let Some(r) = idle_routines.pop_front() {
                    break r;
                }
            };

            match res {
                RoutineRunnerResult::Finished { next_routine } => next_routine.send(next).unwrap(),
                RoutineRunnerResult::Yield {
                    yielded,
                    reason: _,
                    next_routine,
                } => {
                    next_routine.send(next).unwrap();

                    let mut idle_routines = self.idle_routines.write().unwrap();
                    idle_routines.push_back(yielded);
                }
            }
        }
    }
}
