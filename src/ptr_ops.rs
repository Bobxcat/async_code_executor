use std::{
    alloc::Layout,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub const KIBIBYTE: usize = 1024;
pub const MEBIBYTE: usize = KIBIBYTE * KIBIBYTE;
pub const GIBIBYTE: usize = MEBIBYTE * KIBIBYTE;
pub const TEBIBYTE: usize = GIBIBYTE * KIBIBYTE;

/// Returns (`aligned_ptr`, `offset_applied`)
#[inline(always)]
pub fn align_ptr_up(ptr: usize, align: usize) -> (*const (), usize) {
    if !align.is_power_of_two() {
        panic!("Alignment must be power of two");
    }
    if is_aligned(ptr, align) {
        return (ptr as *const (), 0);
    }

    let offset = align - ptr % align;
    let ptr = (ptr + offset) as *const ();
    debug_assert!(
        is_aligned(ptr as usize, align),
        "`align_ptr_up` failed: {ptr:p} (offset applied={offset}) not aligned to {align}"
    );
    (ptr, offset)
}

#[inline(always)]
pub fn is_aligned(ptr: usize, align: usize) -> bool {
    if !align.is_power_of_two() {
        panic!("Alignment must be power of two");
    }
    ptr % align == 0
}

/// An owned allocation of bytes that has a runtime specified over-alignment
#[derive(Debug)]
pub struct AlignedBytes {
    layout: Layout,
    sl: NonNull<[u8]>,
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        let mut new = Self::alloc(self.layout.size(), self.layout.align());
        new.as_mut_slice().clone_from_slice(self);
        new
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: `self.sl` was allocated using `self.layout`, this `AlignedBytes` owns its allocation
        unsafe { std::alloc::dealloc(self.sl.as_ptr().cast(), self.layout) }
    }
}

impl AlignedBytes {
    /// Allocates a slice of `len` bytes on the heap, and guarantees that the allocation is aligned to at least `align`
    ///
    /// The returned allocation is zeroed, so the slice is initialized
    ///
    /// * Panics if `align` is not a power of two
    pub fn alloc(len: usize, align: usize) -> Self {
        if !align.is_power_of_two() {
            panic!("Alignment must be power of two");
        }

        let layout = Layout::array::<u8>(len).unwrap().align_to(align).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let sl = std::ptr::slice_from_raw_parts_mut(ptr, len);

        Self {
            layout,
            sl: NonNull::new(sl).unwrap(),
        }
    }
    pub fn len(&self) -> usize {
        self.sl.len()
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `sl` is guaranteed init
        unsafe { self.sl.as_ref() }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `sl` is guaranteed init
        unsafe { self.sl.as_mut() }
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Sets every byte in the slice to `val`, without assuming initialization
pub fn set_bytes(sl: NonNull<[u8]>, val: u8) {
    let len = sl.len();
    unsafe { std::ptr::write_bytes(sl.as_ptr() as *mut u8, val, len) }
}
