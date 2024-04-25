use std::{alloc::GlobalAlloc, mem::MaybeUninit, num::NonZeroUsize, ops::DerefMut, sync::Mutex};

use num_format::{Locale, ToFormattedString};

use crate::executor::ProgramAlloc;

#[derive(Debug)]
pub struct DefaultProgramAlloc {
    bytes_allocated: Mutex<usize>,
    max: Option<NonZeroUsize>,
}

impl DefaultProgramAlloc {
    pub fn new(allocated_bytes_limit: Option<NonZeroUsize>) -> Self {
        Self {
            bytes_allocated: Mutex::new(0),
            max: allocated_bytes_limit,
        }
    }
}

impl DefaultProgramAlloc {
    fn tracking<'a>(&'a self) -> impl DerefMut<Target = usize> + 'a {
        self.bytes_allocated.lock().unwrap()
    }
}

unsafe impl GlobalAlloc for DefaultProgramAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let bytes_allocated = &mut *self.tracking();
        if let Some(max) = self.max {
            if *bytes_allocated + layout.size() > max.into() {
                let f = &Locale::en;
                panic!(
                    "`DefaultProgramAlloc` panicked when allocating:
                Tried allocating: {}B
                Already allocated: {}B
                Maximum allowed allocations: {}B
                ",
                    bytes_allocated.to_formatted_string(f),
                    layout.size().to_formatted_string(f),
                    max.to_formatted_string(f)
                )
            }
        }

        *bytes_allocated += layout.size();

        unsafe { std::alloc::alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcPtr {
    addr: NonZeroUsize,
}

impl GcPtr {
    pub fn as_hex_addr(self) -> String {
        format!("{:p}", self.as_mut_ptr::<()>())
    }
    /// Reads this as a ptr of the given type. Not guaranteed to be a valid `T`
    fn as_mut_ptr<T>(self) -> *mut T {
        usize::from(self.addr) as *mut T
    }
}

#[derive(Debug)]
pub struct ProgramGC<A: ProgramAlloc> {
    allocator: A,
    /// (`Allocation`, `num_stack_references`)
    allocations: Mutex<Vec<(GcPtr, usize)>>,
}

impl<A: ProgramAlloc> ProgramGC<A> {
    pub fn new(a: A) -> Self {
        Self {
            allocator: a,
            allocations: Mutex::new(vec![]),
        }
    }
    /// Allocates a new heap value and returns a `GcPtr` to it
    ///
    /// Sets the number of stack allocated `GcPtr`s to `1`
    pub fn allocate_uninit<T>(&self) -> GcPtr {
        let p = self.allocator.alloc_type::<T>();
        let gcp = GcPtr {
            addr: (p.as_mut_ptr() as usize).try_into().unwrap(),
        };

        let mut allocs = self.allocations.lock().unwrap();
        allocs.push((gcp.clone(), 1));

        gcp
    }
    /// Allocates a new heap value using `allocate_uninit` and initializes it using the initializer
    ///
    /// Panics if `init` returns a reference that is not the same address as what was given to `init`
    pub fn allocate_init<T>(&self, init: impl FnOnce(&mut MaybeUninit<T>) -> &mut T) -> GcPtr {
        let gcp = self.allocate_uninit::<T>();
        let p = gcp.clone().as_mut_ptr::<MaybeUninit<T>>();

        let p_init = init(unsafe { &mut *p }) as *mut T;

        if p as usize != p_init as usize {
            panic!(
                "`allocate_init` failed:
            Initializer function was given address {:p} and returned address {:p}.
            The returned address must match the supplied address",
                p, p_init
            );
        }

        gcp
    }
    pub fn register_stack_gcptr<T>(&self, pointer: GcPtr) {
        let mut allocs = self.allocations.lock().unwrap();

        let i = allocs
            .iter()
            .position(|(p, _)| p == &pointer)
            .expect("`register_stack_allocation` called on non-existent allocation");

        allocs[i].1 += 1;
    }
    pub fn drop_stack_gcptr<T>(&self, pointer: GcPtr) {
        let mut allocs = self.allocations.lock().unwrap();

        let i = allocs
            .iter()
            .position(|(p, _)| p == &pointer)
            .expect("`drop_stack_allocation` called on non-existent allocation");

        allocs[i].1 = allocs[i]
            .1
            .checked_sub(1)
            .expect("`drop_stack_allocation` called when stack allocation count was already `0`");
    }
}
