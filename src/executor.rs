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
    function::{CodePoint, Function, FunctionId},
    gc::{DefaultProgramAlloc, GcPtr, ProgramGC},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CustomTypeId {
    name: String,
}

impl CustomTypeId {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[derive(Debug, Clone)]
pub struct SendChannel {
    tx: Sender<SizedVal>,
}

#[derive(Debug)]
pub struct RecvChannel {
    tx: Receiver<SizedVal>,
}

#[derive(Debug, Clone, Copy)]
pub enum Literal {
    F32(f32),
    F64(f64),
}

impl From<Literal> for SizedVal {
    fn from(value: Literal) -> Self {
        match value {
            Literal::F32(f) => SizedVal::F32(f),
            Literal::F64(f) => SizedVal::F64(f),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SizedTy {
    F32,
    F64,
    SendChannel,
    RecvChannel,
    Custom(CustomTypeId),
    Ptr(Box<SizedTy>),
}

#[derive(Debug)]
pub enum SizedVal {
    F32(f32),
    F64(f64),
    SendChannel(SendChannel),
    RecvChannel(RecvChannel),
    Ptr(GcPtr, SizedTy),
}

impl SizedVal {
    pub fn type_of(&self) -> SizedTy {
        match self {
            SizedVal::F32(_) => SizedTy::F32,
            SizedVal::F64(_) => SizedTy::F64,
            SizedVal::SendChannel(_) => SizedTy::SendChannel,
            SizedVal::RecvChannel(_) => SizedTy::RecvChannel,
            SizedVal::Ptr(_, ty) => SizedTy::Ptr(Box::new(ty.clone())),
        }
    }
}

#[derive(Debug)]
struct Stack {
    values: Vec<SizedVal>,
}

impl Stack {
    pub fn empty() -> Self {
        Self { values: vec![] }
    }
    pub fn pop(&mut self) -> Option<SizedVal> {
        self.values.pop()
    }
    pub fn push(&mut self, s: SizedVal) {
        self.values.push(s)
    }
    /// Reads the `i`th element from the top of the stack
    pub fn get(&self, i: usize) -> Option<&SizedVal> {
        self.values.get(self.values.len() - 1 - i)
    }
    /// Reads the `i`th element from the top of the stack
    pub fn get_mut(&mut self, i: usize) -> Option<&mut SizedVal> {
        let last = self.values.len() - 1;
        self.values.get_mut(last - i)
    }
    pub fn reverse(mut self) -> Self {
        let v = self.values.into_iter().rev().collect();
        self.values = v;
        self
    }
}

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
    functions: HashMap<FunctionId, Function>,
    _p: PhantomData<()>,
}

impl ExecutorBuilder {
    pub fn new() -> Self {
        Self {
            program_alloc: DefaultProgramAlloc::new(Some(1_000_000.try_into().unwrap())),
            num_threads: Empty::new(),
            functions: HashMap::new(),
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
            _p: PhantomData,
        }
    }
    pub fn set_allocator<A: ProgramAlloc>(self, a: A) -> ExecutorBuilder<A, MaybeNumThreads> {
        ExecutorBuilder {
            program_alloc: a,
            num_threads: self.num_threads,
            functions: self.functions,
            _p: PhantomData,
        }
    }
    pub fn insert_function(mut self, f: Function) -> Self {
        let id = f.id.clone();
        if let Some(_) = self.functions.insert(id.clone(), f) {
            panic!("Cannot insert function {:?}, already exists", id)
        }
        self
    }
    fn into_executor(self) -> Executor<Alloc> {
        Executor {
            data: ExecutorData {
                gc: ProgramGC::new(self.program_alloc),
                functions: self.functions,
                idle_routines: RwLock::new(VecDeque::new()),
            },
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

#[derive(Debug)]
struct ExecutorData<A: ProgramAlloc> {
    gc: ProgramGC<A>,
    functions: HashMap<FunctionId, Function>,
    idle_routines: RwLock<VecDeque<Routine<A>>>,
}

#[derive(Debug)]
enum TypeMismatchErr {
    TraitNotImplemented { ty: SizedTy, needs_trait: String },
}

impl Display for TypeMismatchErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}

impl std::error::Error for TypeMismatchErr {}

impl<A: ProgramAlloc> ExecutorData<A> {
    pub fn type_is_clone(&self, ty: SizedTy) -> bool {
        match ty {
            SizedTy::F32 | SizedTy::F64 | SizedTy::SendChannel | SizedTy::Ptr(_) => true,
            SizedTy::RecvChannel => false,
            SizedTy::Custom(_) => todo!(),
        }
    }
    pub fn try_clone(&self, val: &SizedVal) -> Option<SizedVal> {
        if !self.type_is_clone(val.type_of()) {
            return None;
        }
        Some(match val {
            SizedVal::F32(f) => SizedVal::F32(*f),
            SizedVal::F64(f) => SizedVal::F64(*f),
            SizedVal::SendChannel(s) => SizedVal::SendChannel(s.clone()),
            SizedVal::Ptr(_, _) => todo!(),
            SizedVal::RecvChannel(_) => return None,
        })
    }
    pub fn try_add(&self, lhs: &SizedVal, rhs: &SizedVal) -> Result<SizedVal, TypeMismatchErr> {
        Ok(match (lhs, rhs) {
            (SizedVal::F32(a), SizedVal::F32(b)) => SizedVal::F32(*a + *b),
            (SizedVal::F64(a), SizedVal::F64(b)) => SizedVal::F64(*a + *b),
            _ => {
                return Err(TypeMismatchErr::TraitNotImplemented {
                    ty: lhs.type_of(),
                    needs_trait: format!("Add<Rhs={:?}>", rhs.type_of()),
                })
            }
        })
    }
    pub fn try_sub(&self, lhs: &SizedVal, rhs: &SizedVal) -> Result<SizedVal, TypeMismatchErr> {
        Ok(match (lhs, rhs) {
            (SizedVal::F32(a), SizedVal::F32(b)) => SizedVal::F32(*a - *b),
            (SizedVal::F64(a), SizedVal::F64(b)) => SizedVal::F64(*a - *b),
            _ => {
                return Err(TypeMismatchErr::TraitNotImplemented {
                    ty: lhs.type_of(),
                    needs_trait: format!("Sub<Rhs={:?}>", rhs.type_of()),
                })
            }
        })
    }
    pub fn try_mul(&self, lhs: &SizedVal, rhs: &SizedVal) -> Result<SizedVal, TypeMismatchErr> {
        Ok(match (lhs, rhs) {
            (SizedVal::F32(a), SizedVal::F32(b)) => SizedVal::F32(*a * *b),
            (SizedVal::F64(a), SizedVal::F64(b)) => SizedVal::F64(*a * *b),
            _ => {
                return Err(TypeMismatchErr::TraitNotImplemented {
                    ty: lhs.type_of(),
                    needs_trait: format!("Mul<Rhs={:?}>", rhs.type_of()),
                })
            }
        })
    }
    pub fn try_div(&self, lhs: &SizedVal, rhs: &SizedVal) -> Result<SizedVal, TypeMismatchErr> {
        Ok(match (lhs, rhs) {
            (SizedVal::F32(a), SizedVal::F32(b)) => SizedVal::F32(*a / *b),
            (SizedVal::F64(a), SizedVal::F64(b)) => SizedVal::F64(*a / *b),
            _ => {
                return Err(TypeMismatchErr::TraitNotImplemented {
                    ty: lhs.type_of(),
                    needs_trait: format!("Div<Rhs={:?}>", rhs.type_of()),
                })
            }
        })
    }
    pub fn try_dbg_format(&self, lhs: &SizedVal) -> Result<String, TypeMismatchErr> {
        Ok(match lhs {
            SizedVal::F32(a) => format!("{a}"),
            SizedVal::F64(a) => format!("{a}"),
            SizedVal::Ptr(a, ty) => format!("{ty:?}@{}", a.clone().as_hex_addr()),
            _ => {
                return Err(TypeMismatchErr::TraitNotImplemented {
                    ty: lhs.type_of(),
                    needs_trait: format!("DebugFormat"),
                })
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CodePointIdx(pub usize);

#[derive(Debug)]
struct Routine<A: ProgramAlloc> {
    data_stack: Stack,
    call_stack: Vec<(*const Function, Option<CodePointIdx>)>,
    exec: *const Executor<A>,
}

unsafe impl<A: ProgramAlloc> Send for Routine<A> {}

impl<A: ProgramAlloc> Routine<A> {
    fn exec(&self) -> &Executor<A> {
        unsafe { &*self.exec }
    }
}

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

        match curr_code_point {
            CodePoint::SpawnRoutine(f) => {
                let spawned_args = routine.exec().data.functions[f].params.clone();
                let mut starting_stack = Stack::empty();
                for _ty in &spawned_args {
                    starting_stack.push(routine.data_stack.pop().unwrap());
                }
                starting_stack = starting_stack.reverse();
                routine.exec().spawn_routine(f.clone(), starting_stack);
            }
            CodePoint::CallFunction(f) => {
                let func = &routine.exec().data.functions[f] as *const _;
                routine.call_stack.push((func, None));
            }
            // CodePoint::Add => {
            //     let b = routine.data_stack.pop().unwrap();
            //     let a = routine.data_stack.pop().unwrap();

            //     let res = routine.exec().data.try_add(&a, &b).unwrap();
            //     routine.data_stack.push(res);
            // }
            // CodePoint::Sub => {
            //     let b = routine.data_stack.pop().unwrap();
            //     let a = routine.data_stack.pop().unwrap();

            //     let res = routine.exec().data.try_sub(&a, &b).unwrap();
            //     routine.data_stack.push(res);
            // }
            // CodePoint::Mul => {
            //     let b = routine.data_stack.pop().unwrap();
            //     let a = routine.data_stack.pop().unwrap();

            //     let res = routine.exec().data.try_mul(&a, &b).unwrap();
            //     routine.data_stack.push(res);
            // }
            // CodePoint::Div => {
            //     let b = routine.data_stack.pop().unwrap();
            //     let a = routine.data_stack.pop().unwrap();

            //     let res = routine.exec().data.try_div(&a, &b).unwrap();
            //     routine.data_stack.push(res);
            // }
            CodePoint::Recv => todo!(),
            CodePoint::Send => todo!(),
            CodePoint::Clone => todo!(),
            // CodePoint::Literal(lhs) => routine.data_stack.push(lhs.clone().into()),
            CodePoint::Allocate => todo!(),
            CodePoint::AssertType(_) => todo!(),
            CodePoint::DebugPrint => {
                let a = routine.data_stack.get(0).unwrap();
                println!(
                    "From `{:?}`:\n{}",
                    curr_func.id,
                    routine.exec().data.try_dbg_format(a).unwrap()
                );
                std::io::stdout().flush().unwrap();
            }
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
    data: ExecutorData<A>,
}

impl<A: ProgramAlloc> Executor<A> {
    fn spawn_routine(&self, f: FunctionId, starting_stack: Stack) {
        let r = Routine {
            data_stack: starting_stack,
            call_stack: vec![(&self.data.functions[&f], Some(CodePointIdx(0)))],
            exec: self,
        };
        let mut idle_routines = self.data.idle_routines.write().unwrap();
        idle_routines.push_back(r)
    }
    pub fn run(&self, cfg: ExecutorRunCfg) {
        //!
        //! This function dispatches routines to a specified number of worker threads.
        //! There is a queue of idle routines which worker threads pull from. Newly spawned routines are put at the back of this queue.
        //!
        //! Each worker thread will run their routine for a while, until either yielding or working for a specified time.
        //! When a worker thread gives their routine back to this function, their previous routine is placed at the back of the routine queue.
        //!

        let mut routine_returns: Vec<Receiver<RoutineRunnerResult<A>>> = vec![];

        for _ in 0..cfg.num_threads {
            let (tx, rx) = mpsc::channel();

            std::thread::spawn(move || routine_runner::<A>(tx));
            routine_returns.push(rx);
        }

        let entrypoint = FunctionId::new("main");
        self.spawn_routine(entrypoint, Stack::empty());

        loop {
            let Some(res) = routine_returns.iter().find_map(|r| r.try_recv().ok()) else {
                continue;
            };

            let next = loop {
                let mut idle_routines = self.data.idle_routines.write().unwrap();
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

                    let mut idle_routines = self.data.idle_routines.write().unwrap();
                    idle_routines.push_back(yielded);
                }
            }
        }
    }
}
