use std::{
    alloc::Layout,
    collections::HashMap,
    fmt::Debug,
    num::NonZeroUsize,
    pin::Pin,
    ptr::{addr_of_mut, NonNull},
    sync::Arc,
};

use bumpalo::Bump;

use crate::{executor::ProgramAlloc, gc::ProgramGC, global_debug, println_ctx, ptr_ops};

use self::primitives::{NumTy, PrimTy};

/// A trait a primitive type might have, which describes abilities in their behavior. There are no custom traits
///
/// For example, if you have some type `A` and `B` where `A` implements `Add { rhs: B }`, then it's valid to add those two values
/// (the resulting type would be given by `output_type`)
#[derive(Debug, Clone)]
pub enum Trait {
    Add { rhs: PrimTy },
    Sub { rhs: PrimTy },
    Mul { rhs: PrimTy },
    Div { rhs: PrimTy },
    Into { out: PrimTy },
}

pub struct TraitImpl {
    pub implementor: PrimTy,
    pub output: PrimTy,
}

impl Trait {
    /// Gets the implementation data for the given trait and implementor,
    /// hopefully any excess data gets optimized away
    ///
    /// This program is far from being profiled yet...
    #[inline(always)]
    pub fn get_impl(self, implementor: PrimTy) -> Option<TraitImpl> {
        Some(match (implementor.clone(), self.clone()) {
            (
                PrimTy::Num(lhs),
                Trait::Add {
                    rhs: PrimTy::Num(rhs),
                },
            ) if lhs == rhs => TraitImpl {
                implementor,
                output: PrimTy::Num(lhs),
            },
            _ => {
                if global_debug::is_enabled() {
                    println_ctx!("Tried to `get_impl` on a bad trait combination: {:?} not implemented for {:?}", self, implementor);
                }
                return None;
            }
        })
    }
}

/// A cheap-to-clone reference to a custom type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CustomTy(Arc<str>);

impl CustomTy {
    pub fn new(s: impl AsRef<str>) -> Self {
        Self(Arc::from(s.as_ref()))
    }
}

#[derive(Debug, Clone)]
pub struct CustomTyData {
    pub fields: Vec<PrimTy>,
    alloc_layout: Layout,
}

/// Information about the custom types that currently exist
pub struct TypeCtx {
    customs: HashMap<CustomTy, CustomTyData>,
}

#[derive(Debug, Clone)]
pub enum Type {
    Custom(CustomTy),
    Primitive(PrimTy),
}

impl Type {
    #[inline(always)]
    pub fn unwrap_prim(self) -> PrimTy {
        match self {
            Type::Custom(_) => panic!("Called `unwrap_prim` on a `Type::Custom`"),
            Type::Primitive(prim) => prim,
        }
    }
    #[inline(always)]
    pub fn layout(&self, ctx: &TypeCtx) -> Layout {
        match &self {
            Type::Custom(t) => ctx.customs[t].alloc_layout,
            Type::Primitive(t) => t.layout(),
        }
    }
}

impl From<PrimTy> for Type {
    fn from(value: PrimTy) -> Self {
        Self::Primitive(value)
    }
}

/// A typed value which can live on the stack (this may be a heap value as well)
///
/// SAFETY:
/// * This will become invalidated if a deallocation is performed by the stack backing it
/// *
#[derive(Debug)]
pub struct TypeVal {
    ty: Type,
    data_addr: *mut u8,
}

impl TypeVal {
    #[inline(always)]
    pub fn as_bytes_ptr(&self, ctx: &TypeCtx) -> *mut [u8] {
        match &self.ty {
            Type::Custom(_) => todo!(),
            Type::Primitive(_) => self.as_bytes_ptr_primitive(),
        }
    }
    /// Panics if `self` is not a primitive
    #[inline(always)]
    pub fn as_bytes_ptr_primitive(&self) -> *mut [u8] {
        match &self.ty {
            Type::Custom(t) => panic!("Called `as_bytes_ptr_primitive` on non-primitive `{t:?}`"),
            Type::Primitive(ty) => {
                let len = ty.clone().layout().size();
                unsafe { std::slice::from_raw_parts_mut(self.data_addr, len) }
            }
        }
    }
    // /// Writes
    // pub fn write(&mut self, new_val: Type)
}

/// A free-standing owned allocation of a given type
///
/// An instance of `OwnedValue` is assumed to be a valid instance of its `Type`
pub struct OwnedValue {
    ty: Type,
    raw: Box<[u8]>,
}

impl OwnedValue {
    /// Does not check to make sure this type is valid
    pub unsafe fn new_raw(ty: Type, bytes: Box<[u8]>) -> Self {
        Self { ty, raw: bytes }
    }
    pub unsafe fn as_ptr<T>(&self) -> *const T {
        self.raw.as_ptr() as *const T
    }
    /// `self` must refer to a valid value of type `T`
    pub unsafe fn as_ref<T>(&self) -> &T {
        unsafe { &*self.as_ptr() }
    }
    /// `self` must refer to a valid value of type `T`
    pub unsafe fn as_mut<T>(&mut self) -> &mut T {
        unsafe { &mut *(self.as_ptr::<T>() as *mut T) }
    }
}

/// Ownership over an allocated stack value. This cannot be `Clone`d since the handle must be returned to the stack in order to deallocate
///
/// This struct **owns** the data in a stack allocation. The underlying bytes (accessable by `as_bytes(..)`) are always valid, but may not be initialized.
/// The first byte is always aligned according to the type of this `StackValHandle`
///
/// This will be invalidated if the stack gets dropped or the underlying allocation is otherwise invalidated
#[derive(Debug)]
pub struct StackValHandle {
    ty: Type,
    data_addr: NonNull<u8>,
}

impl StackValHandle {
    /// Interprets the underlying data as a reference of the given type.
    /// This is useful since it enforces lifetime and mutability rules
    ///
    /// SAFETY: this handle must represent a valid value of the given type
    pub unsafe fn as_mut<T>(&mut self) -> &mut T {
        unsafe { &mut *(self.data_addr.as_ptr() as *mut T) }
    }
    /// Interprets the underlying data as a reference of the given type
    /// This is useful since it enforces lifetime and mutability rules
    ///
    /// SAFETY: this handle must represent a valid value of the given type
    pub unsafe fn as_ref<T>(&self) -> &T {
        unsafe { &*(self.data_addr.as_ptr() as *const T) }
    }
    pub fn as_ptr<T>(&self) -> *const T {
        self.data_addr.as_ptr() as *const T
    }
    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.data_addr.as_ptr() as *mut T
    }
    #[inline(always)]
    pub fn as_bytes(&self, ctx: &TypeCtx) -> &[u8] {
        unsafe { &*self.as_bytes_ptr(ctx) }
    }
    #[inline(always)]
    pub fn as_bytes_primitive(&self) -> &[u8] {
        unsafe { &*self.as_bytes_ptr_primitive() }
    }
    #[inline(always)]
    pub fn as_mut_bytes(&mut self, ctx: &TypeCtx) -> &mut [u8] {
        unsafe { &mut *self.as_bytes_ptr(ctx) }
    }
    #[inline(always)]
    pub fn as_mut_bytes_primitive(&mut self) -> &mut [u8] {
        unsafe { &mut *self.as_bytes_ptr_primitive() }
    }
    #[inline(always)]
    pub fn as_bytes_ptr(&self, ctx: &TypeCtx) -> *mut [u8] {
        match &self.ty {
            Type::Custom(id) => {
                let len = ctx.customs[id].alloc_layout.size();
                unsafe { std::slice::from_raw_parts_mut(self.data_addr.as_ptr(), len) }
            }
            Type::Primitive(_) => self.as_bytes_ptr_primitive(),
        }
    }
    /// Panics if `self` is not a primitive
    #[inline(always)]
    pub fn as_bytes_ptr_primitive(&self) -> *mut [u8] {
        match &self.ty {
            Type::Custom(t) => panic!("Called `as_bytes_ptr_primitive` on non-primitive `{t:?}`"),
            Type::Primitive(ty) => {
                let len = ty.clone().layout().size();
                unsafe { std::slice::from_raw_parts_mut(self.data_addr.as_ptr(), len) }
            }
        }
    }
}

#[repr(C)]
pub(crate) struct ValStack {
    /// To avoid making mistakes with miri, use a pre-made bump allocator!
    alloc: Bump,
}

impl ValStack {
    /// Creates a new `ValStack` with 10MB of capacity
    pub fn new_default() -> Self {
        Self::new(Some(10_000_000.try_into().unwrap()))
    }
    pub fn new(max_capacity: Option<NonZeroUsize>) -> Self {
        let mut alloc = Bump::new();
        alloc.set_allocation_limit(max_capacity.map(|x| x.into()));
        Self { alloc }
    }
    pub fn capacity(&self) -> usize {
        // self.alloc
        todo!()
    }
    pub fn allocated_bytes(&self) -> usize {
        // self.header.len
        todo!()
    }
    /// Points to the first address after all allocated bytes
    ///
    /// Returns `*const ()` since the byte might not necessarily be valid (if this stack is at capacity)
    pub fn allocation_end(&self) -> *const () {
        // (self.blob.as_ptr() as usize + self.header.len) as *const ()
        todo!()
    }
    #[must_use]
    pub fn alloc_primitive<'data>(&'data mut self, prim: PrimTy) -> StackValHandle {
        unsafe { self.alloc_layout(prim.clone().layout(), Type::Primitive(prim)) }
    }
    #[inline(always)]
    pub fn alloc_custom(&mut self, ty: Type, ctx: &TypeCtx) -> StackValHandle {
        unsafe { self.alloc_layout(ty.layout(ctx), ty) }
    }
    /// SAFETY: `layout` must  be correct for the given `ty` parameter, there are no checks for what is passed as `ty`
    ///
    /// The returned `TypeVal` is a handle given
    #[must_use]
    pub unsafe fn alloc_layout<'data>(&'data mut self, layout: Layout, ty: Type) -> StackValHandle {
        let ptr = self.alloc.alloc_layout(layout);
        StackValHandle { ty, data_addr: ptr }
    }

    #[inline(always)]
    pub fn pop(&mut self, val: StackValHandle, ctx: &TypeCtx) {
        match &val.ty {
            Type::Custom(ty) => {
                let layout = ctx.customs[ty].alloc_layout;
                // SAFETY: The given layout is correct for this type
                unsafe { self.pop_custom(val, layout) }
            }
            Type::Primitive(_) => self.pop_primitive(val),
        }
    }

    #[inline(always)]
    pub fn pop_ret(&mut self, val: StackValHandle, ctx: &TypeCtx) -> OwnedValue {
        match &val.ty {
            Type::Custom(ty) => {
                let layout = ctx.customs[ty].alloc_layout;
                // SAFETY: The given layout is correct for this type
                unsafe { self.pop_custom_ret(val, layout) }
            }
            Type::Primitive(_) => self.pop_primitive_ret(val),
        }
    }

    #[inline(always)]
    pub fn pop_primitive(&mut self, val: StackValHandle) {
        let ty = val.ty.clone().unwrap_prim();

        // SAFETY: This layout matches the given type
        unsafe { self.pop_custom(val, ty.layout()) }
    }

    #[inline(always)]
    pub fn pop_primitive_ret(&mut self, val: StackValHandle) -> OwnedValue {
        let ty = val.ty.clone().unwrap_prim();

        // SAFETY: This layout matches the given type
        unsafe { self.pop_custom_ret(val, ty.layout()) }
    }

    #[inline(always)]
    unsafe fn pop_custom(
        &mut self,
        StackValHandle { ty: _, data_addr }: StackValHandle,
        layout: Layout,
    ) {
        // // Check to make sure this is the correct `StackValHandle`
        // {
        //     // Do not add `pre_padding`, since this goes before the `data_adddr`
        //     let end_of_handle_allocation = data_addr as usize + layout.size();

        //     if end_of_handle_allocation != self.allocation_end() as usize {
        //         panic!(
        //             "`pop_custom` called on `StackValHandle` which was not at the end of the stack:
        //         Tried to deallocate {layout:?} & padding={pre_padding} from {}
        //             (This allocation ends at {end_of_handle_allocation})
        //         Current end of allocation was at {}
        //             The end of allocation was ahead of attempted deallocation by {}",
        //             data_addr as usize - pre_padding,
        //             self.allocation_end() as usize,
        //             self.allocation_end() as usize as isize - end_of_handle_allocation as isize
        //         )
        //     }
        // }

        // let total_handle_allocation_size = pre_padding + layout.size();

        // self.header.len -= total_handle_allocation_size;
        todo!()
    }

    #[inline(always)]
    unsafe fn pop_custom_ret(&mut self, val: StackValHandle, layout: Layout) -> OwnedValue {
        let stack_bytes = val.as_bytes_primitive();

        let mut bytes = ptr_ops::alloc_bytes_aligned(layout.size(), layout.align());
        assert_eq!(bytes.len(), stack_bytes.len());
        for i in 0..bytes.len() {
            bytes[i] = stack_bytes[i];
        }
        // SAFETY:
        let ret = unsafe { OwnedValue::new_raw(val.ty.clone(), bytes) };
        // SAFETY: The `OwnedValue` has been created and no longer cares about the original stack allocation,
        // and the arguments to `pop_custom` match the arguments to this function
        unsafe { self.pop_custom(val, layout) };
        ret
    }
}

#[test]
fn test_val_stack_new() {
    use std::hint::black_box;
    fn foo(cap: usize) {
        let a = ValStack::new(Some(cap.try_into().unwrap()));
        assert_eq!(a.capacity(), cap);
        black_box(a);
    }

    foo(1000);
    foo(1_000_000);
    foo(100_000_000);
}

#[test]
fn test_val_stack_allocation() {
    let v = &mut ValStack::new_default();
    fn foo<T>(v: &mut ValStack) {
        let start_bytes = v.allocated_bytes();

        let layout = Layout::new::<T>();
        let val = unsafe { v.alloc_layout(layout, PrimTy::Void.into()) };

        let delta_bytes = v.allocated_bytes() - start_bytes;
        assert!(
            delta_bytes >= layout.size(),
            "delta_bytes={delta_bytes} < layout.size()={}",
            layout.size()
        );

        let p_addr = val.data_addr.as_ptr() as usize;

        assert!(ptr_ops::is_aligned(p_addr, layout.align()));
    }

    foo::<i32>(v);
    foo::<i64>(v);
    foo::<i128>(v);
    foo::<f32>(v);
}

#[test]
fn test_val_stack_alloc_dealloc() {
    let mut v = ValStack::new_default();
    let mut bottom = v.alloc_primitive(PrimTy::Num(NumTy::U32));
    *unsafe { bottom.as_mut::<u32>() } = 10;
    let mut handles = vec![];

    for i in 0..100 {
        for h in 0..=i {
            let mut handle = v.alloc_primitive(PrimTy::Num(NumTy::F64));
            // SAFETY: This handle is a primitive `f64`
            let f = unsafe { handle.as_mut::<f64>() };
            *f = h as f64;

            handles.push(handle);
        }
        for h in 0..=i {
            let seek = (i - h) as f64;
            let h = handles.pop().unwrap();
            println_ctx!("{i}");
            let top = v.pop_primitive_ret(h);
            let copied: f64 = unsafe { *top.as_ref() };
            assert_eq!(copied, seek);
        }
        assert!(handles.is_empty())
    }

    println_ctx!();

    let bot = v.pop_primitive_ret(bottom);
    let copied: u32 = unsafe { *bot.as_ref() };
    assert_eq!(copied, 10);
}

pub mod primitives {
    use std::{alloc::Layout, rc::Rc};

    use super::Type;

    /// Includes the number types
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum NumTy {
        F32,
        F64,
        I32,
        U32,
    }

    impl NumTy {
        #[inline(always)]
        pub fn layout(self) -> Layout {
            match self {
                NumTy::F32 => Layout::new::<f32>(),
                NumTy::F64 => Layout::new::<f64>(),
                NumTy::I32 => Layout::new::<i32>(),
                NumTy::U32 => Layout::new::<u32>(),
            }
        }
    }

    /// Includes all builtin types
    #[derive(Debug, Clone)]
    pub enum PrimTy {
        Void,
        GcPtr(Rc<Type>),
        Num(NumTy),
    }

    impl PrimTy {
        #[inline(always)]
        pub fn is_num(self) -> bool {
            match self {
                PrimTy::Num(_) => true,
                PrimTy::Void | PrimTy::GcPtr(_) => false,
            }
        }
        #[inline(always)]
        pub fn layout(&self) -> Layout {
            match self {
                PrimTy::Void => Layout::new::<()>(),
                PrimTy::GcPtr(_) => todo!(),
                PrimTy::Num(ty) => ty.layout(),
            }
        }
    }
}
