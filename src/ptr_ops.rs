use std::{alloc::Layout, ptr::NonNull};

pub const KILOBYTE: usize = 1024;
pub const MEGABYTE: usize = KILOBYTE * KILOBYTE;
pub const GIGABYTE: usize = MEGABYTE * MEGABYTE;

/// Returns (`aligned_ptr`, `offset_applied`)
#[inline(always)]
pub fn align_ptr_up(ptr: usize, align: usize) -> (*const (), usize) {
    if !align.is_power_of_two() {
        panic!("Alignment must be power of two");
    }
    let offset = ptr % align;
    let ptr = (ptr + offset) as *const ();
    (ptr, offset)
}

#[inline(always)]
pub fn is_aligned(ptr: usize, align: usize) -> bool {
    if !align.is_power_of_two() {
        panic!("Alignment must be power of two");
    }
    ptr % align == 0
}

/// Allocates a slice of `len` bytes on the heap, and guarantees that the allocation is aligned to at least `align`
///
/// The returned allocation is not initialized, but any bit sequence will be a valid `u8`
///
/// * Panics if `align` is not a power of two
#[inline(always)]
pub fn alloc_bytes_aligned(len: usize, align: usize) -> Box<[u8]> {
    if !align.is_power_of_two() {
        panic!("Alignment must be power of two");
    }

    let layout = Layout::array::<u8>(len).unwrap().align_to(align).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    let sl = std::ptr::slice_from_raw_parts_mut(ptr, len);

    unsafe { Box::from_raw(sl) }
}

/// Sets every byte in the slice to `val`, without assuming initialization
pub fn set_bytes(sl: NonNull<[u8]>, val: u8) {
    let len = sl.len();
    unsafe { std::ptr::write_bytes(sl.as_ptr() as *mut u8, val, len) }
}
