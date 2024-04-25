//!
//! A simple bump allocator implementation backing stacks in this VM. This type is not `Sync`, but is `Send` since it owns its underlying memory
//!
//! Testing done with `Miri` to avoid UB
//!

use std::{alloc::Layout, mem::ManuallyDrop, num::NonZeroUsize, ptr::NonNull};

use thiserror::Error;

use crate::ptr_ops;

///
pub struct Bump {
    base: NonNull<()>,
    len: usize,
    capacity: usize,
}

#[derive(Error, Debug)]
pub enum BumpErr {
    #[error("STACK OVERFLOW: Capacity was {capacity:?}B, Reached {reached:?}B while allocating {while_allocating:?}")]
    StackOverflow {
        capacity: usize,
        reached: usize,
        while_allocating: Layout,
    },
}

impl Bump {
    pub fn new_default() -> Self {
        Self::new(ptr_ops::MEGABYTE * 2)
    }

    pub fn new(max_capacity: usize) -> Self {
        let allocation = unsafe { std::alloc::alloc(Layout::array::<u8>(max_capacity).unwrap()) };
        let base = NonNull::new(allocation).unwrap();
        Self {
            base: base.cast(),
            len: 0,
            capacity: max_capacity,
        }
    }

    /// Gets a pointer `offset` bytes above the base of the allocation
    #[inline(always)]
    fn ptr_from_base(&self, offset: usize) -> Option<NonNull<()>> {
        if offset >= self.capacity {
            return None;
        }
        let addr = (self.base.as_ptr() as usize).checked_add(offset)?;
        NonNull::new(addr as *mut ())
    }
    /// Returns a pointer to the first byte that hasn't been allocated
    ///
    /// This returns `None` if capacity has been reached
    #[inline(always)]
    fn alloc_head(&self) -> Option<NonNull<()>> {
        self.ptr_from_base(self.len)
    }
    /// Returns the maximum number of bytes that this `Bump` can allocate
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Returns the number of bytes allocated
    #[inline(always)]
    pub fn allocate_bytes(&self) -> usize {
        self.len
    }
    /// Returns whether or not the given address is within the memory span of this `Bump`
    #[inline(always)]
    fn within_mem(&self, addr: usize) -> bool {
        let bot = self.base.as_ptr() as usize;
        let top = self.base.as_ptr() as usize + self.capacity;
        (bot..top).contains(&addr)
    }
    /// Returns whether or not the given pointer is within the memory span of this `Bump`
    #[inline(always)]
    fn within_mem_ptr(&self, addr: NonNull<()>) -> bool {
        self.within_mem(addr.as_ptr() as usize)
    }
    /// Returns whether or not the given pointer is within the allocated memory span of this `Bump`
    #[inline(always)]
    fn within_allocated(&self, addr: usize) -> bool {
        let bot = self.base.as_ptr() as usize;
        let top = self.base.as_ptr() as usize + self.len;
        (bot..top).contains(&addr)
    }
    /// Returns whether or not the given pointer is within the allocated memory span of this `Bump`
    #[inline(always)]
    fn within_allocated_ptr(&self, addr: NonNull<()>) -> bool {
        self.within_allocated(addr.as_ptr() as usize)
    }
    /// The returned pointer points to a valid T
    #[inline(always)]
    pub fn alloc_from<T: 'static>(&mut self, owned: Box<T>) -> Result<NonNull<T>, Box<BumpErr>> {
        let src = Box::leak(owned);
        let dest = self.alloc_layout(Layout::for_value(src))?.cast::<T>();
        unsafe { std::ptr::copy(src, dest.as_ptr(), std::mem::size_of::<T>()) };
        let needs_drop = unsafe { &mut *(src as *mut T as *mut ManuallyDrop<T>) };
        std::mem::drop(unsafe { Box::from_raw(needs_drop) });

        Ok(dest)
    }
    /// Allocates a byte array of the given length and returns a valid pointer to it
    ///
    /// The returned bytes are not zeroed
    #[inline(always)]
    pub fn alloc_bytes(&mut self, bytes: usize) -> Result<NonNull<[u8]>, Box<BumpErr>> {
        self.alloc_layout(Layout::array::<u8>(bytes).unwrap())
            .map(|ptr| std::ptr::slice_from_raw_parts_mut(ptr.cast::<u8>().as_ptr(), bytes))
            .map(|sl| NonNull::new(sl).unwrap())
    }
    /// Allocates a block of memory with the given layout, panicking on failure
    #[inline(always)]
    pub fn alloc_layout(&mut self, layout: Layout) -> Result<NonNull<()>, Box<BumpErr>> {
        let alloc_head = self.alloc_head().ok_or_else(|| BumpErr::StackOverflow {
            capacity: self.capacity(),
            reached: self.len,
            while_allocating: layout,
        })?;
        let (ptr, _) = ptr_ops::align_ptr_up(alloc_head.as_ptr() as usize, layout.align());

        let ptr = NonNull::new(ptr.cast_mut()).unwrap();
        let ptr_end = ptr.as_ptr() as usize + layout.size();

        if self.within_mem(ptr_end) {
            self.len = ptr_end - self.base.as_ptr() as usize;
            Ok(ptr)
        } else {
            Err(BumpErr::StackOverflow {
                capacity: self.capacity(),
                reached: ptr_end,
                while_allocating: layout,
            })?;
            unreachable!()
        }
    }
    /// Attemps to deallocate (not Drop) all the bytes between the current allocation head and `new_head`,
    ///
    /// Panics if `new_head` is not one of this stack's allocated bytes
    #[inline(always)]
    pub fn deallocate_to(&mut self, new_head: NonNull<()>) {
        if !self.within_allocated_ptr(new_head) {
            panic!("Called `deallocate_to` on {new_head:p}")
        }
        let new_len = new_head.as_ptr() as usize - self.base.as_ptr() as usize;
        self.len = new_len;
    }
}

impl Drop for Bump {
    fn drop(&mut self) {
        let sl: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr() as *mut u8, self.capacity) };
        let owned = unsafe { Box::from_raw(sl) };
        std::mem::drop(owned)
    }
}

#[test]
fn test_bump_alloc() {
    let mut stack = Bump::new_default();
    let mut allocs: Vec<&mut [u8]> = vec![];

    const LENS: &[usize] = &[10, 20, 30];

    for &len in LENS {
        let mut ptr = stack.alloc_bytes(len).unwrap();
        ptr_ops::set_bytes(ptr, 0);
        // SAFETY: alloc_bytes guarantees validity, and allocs won't be held beyond the stack's lifetime
        let sl = unsafe { ptr.as_mut() };
        allocs.push(sl);
    }

    println!("{allocs:#?}");
}
