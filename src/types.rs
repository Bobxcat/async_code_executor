use crate::build_status::{CustomTyStore, FinishBuildingCtx};
use std::{alloc::Layout, marker::PhantomData, num::NonZeroUsize, ptr::NonNull};

use crate::{
    build_status::{BuildStatus, Building, Built},
    bump_alloc::Bump,
    executor::ProgramAlloc,
    gc::ProgramGC,
    global_debug, println_ctx,
    ptr_ops::{self, DEFAULT_STACK_CAP, GIBIBYTE, TEBIBYTE},
};

use self::primitives::{NumTy, PrimTy};

/// A trait a primitive type might have, which describes abilities in their behavior. There are no custom traits
///
/// For example, if you have some type `A` and `B` where `A` implements `Add { rhs: B }`, then it's valid to add those two values
/// (the resulting type would be given by `output_type`)
#[derive(Debug, Clone)]
pub enum Trait<B: BuildStatus> {
    Add { rhs: PrimTy<B> },
    Sub { rhs: PrimTy<B> },
    Mul { rhs: PrimTy<B> },
    Div { rhs: PrimTy<B> },
    Into { out: PrimTy<B> },
}

pub struct TraitImpl<B: BuildStatus> {
    pub implementor: PrimTy<B>,
    pub output: PrimTy<B>,
}

impl<B: BuildStatus> Trait<B> {
    /// Gets the implementation data for the given trait and implementor,
    /// hopefully any excess data gets optimized away
    ///
    /// This program is far from being profiled yet...
    #[inline(always)]
    pub fn get_impl(self, implementor: PrimTy<B>) -> Option<TraitImpl<B>> {
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CustomTyName(String);

impl CustomTyName {
    pub fn new(s: impl AsRef<str>) -> Self {
        Self(s.as_ref().into())
    }
}

impl<S: AsRef<str>> From<S> for CustomTyName {
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

impl From<CustomTyName> for String {
    fn from(val: CustomTyName) -> Self {
        val.0
    }
}

/// A cheap-to-clone reference to a custom type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CustomTyIdx(usize);

impl CustomTyIdx {
    pub(crate) fn new(idx: usize) -> Self {
        Self(idx)
    }
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct CustomType<B: BuildStatus> {
    pub(crate) id: B::CustomTyId,
    pub(crate) fields: FieldsLayout<B>,
}

/// Information about the custom types that currently exist
#[derive(Debug)]
pub struct TypeCtx<B: BuildStatus> {
    customs: B::CustomTyStore,
}

impl TypeCtx<Building> {
    pub fn empty() -> Self {
        Self {
            customs: Default::default(),
        }
    }
    pub fn insert_custom(&mut self, ty: CustomType<Building>) {
        if let Some(prev) = self.customs.get(ty.id.clone()) {
            panic!(
                "Called `insert_custom` on already existing type:\n{:#?}",
                prev
            );
        }
        self.customs.insert(ty)
    }
}

impl<B: BuildStatus> TypeCtx<B> {
    pub fn get(&self, ty: B::CustomTyId) -> &CustomType<B> {
        self.customs.get(ty).unwrap()
    }
}

#[derive(Debug, Clone)]
pub enum Type<B: BuildStatus> {
    Custom(B::CustomTyId),
    Primitive(PrimTy<B>),
}

impl<B: BuildStatus> Type<B> {
    #[inline(always)]
    pub fn unwrap_prim(self) -> PrimTy<B> {
        match self {
            Type::Custom(_) => panic!("Called `unwrap_prim` on a `Type::Custom`"),
            Type::Primitive(prim) => prim,
        }
    }
    #[inline(always)]
    pub fn layout(&self, ctx: &TypeCtx<B>) -> Layout {
        match &self {
            Type::Custom(t) => ctx.get(t.clone()).fields.layout(),
            Type::Primitive(t) => t.layout(),
        }
    }
}

impl Type<Building> {
    pub(crate) fn finish(self, ctx: &FinishBuildingCtx) -> Type<Built> {
        match self {
            Type::Custom(id) => Type::Custom(ctx.custom(&id)),
            Type::Primitive(t) => Type::Primitive(t.finish(ctx)),
        }
    }
}

impl<B: BuildStatus> From<PrimTy<B>> for Type<B> {
    fn from(value: PrimTy<B>) -> Self {
        Self::Primitive(value)
    }
}

impl From<CustomTyName> for Type<Building> {
    fn from(value: CustomTyName) -> Self {
        Self::Custom(value)
    }
}

impl From<CustomTyIdx> for Type<Built> {
    fn from(value: CustomTyIdx) -> Self {
        Self::Custom(value)
    }
}

/// A typed value which can live on the stack (this may be a heap value as well)
///
/// SAFETY:
/// * This will become invalidated if a deallocation is performed by the stack backing it
/// *
#[derive(Debug)]
pub struct TypeVal<B: BuildStatus> {
    ty: Type<B>,
    data_addr: *mut u8,
}

impl<B: BuildStatus> TypeVal<B> {
    #[inline(always)]
    pub fn as_bytes_ptr(&self, ctx: &TypeCtx<B>) -> *mut [u8] {
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
#[derive(Debug, Clone)]
pub struct OwnedValue<B: BuildStatus> {
    ty: Type<B>,
    raw: Box<[u8]>,
}

impl<B: BuildStatus> OwnedValue<B> {
    /// Does not check to make sure this type is valid
    pub unsafe fn new_raw(ty: Type<B>, bytes: Box<[u8]>) -> Self {
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

impl OwnedValue<Building> {
    pub(crate) fn finish(self, ctx: &FinishBuildingCtx) -> OwnedValue<Built> {
        OwnedValue {
            ty: self.ty.finish(ctx),
            raw: self.raw,
        }
    }
}

/// Ownership over an allocated stack value. This cannot be `Clone`d since the handle must be returned to the stack in order to deallocate
///
/// This struct **owns** the data in a stack allocation. The underlying bytes (accessable by `as_bytes(..)`) are always valid, but may not be initialized.
/// The first byte is always aligned according to the type of this `StackValHandle`
///
/// SAFETY:
/// * This will be invalidated if the stack gets dropped or the underlying allocation is otherwise invalidated
/// (this will not cause UB until the underlying data is accessed)
/// * ZSTs will be represented by a dangling pointer
#[derive(Debug)]
pub struct ValStackHandle<B: BuildStatus> {
    ty: Type<B>,
    data_addr: NonNull<u8>,
}

impl<B: BuildStatus> ValStackHandle<B> {
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
    pub fn as_bytes(&self, ctx: &TypeCtx<B>) -> &[u8] {
        unsafe { &*self.as_bytes_ptr(ctx) }
    }
    #[inline(always)]
    pub fn as_bytes_primitive(&self) -> &[u8] {
        unsafe { &*self.as_bytes_ptr_primitive() }
    }
    #[inline(always)]
    pub fn as_mut_bytes(&mut self, ctx: &TypeCtx<B>) -> &mut [u8] {
        unsafe { &mut *self.as_bytes_ptr(ctx) }
    }
    #[inline(always)]
    pub fn as_mut_bytes_primitive(&mut self) -> &mut [u8] {
        unsafe { &mut *self.as_bytes_ptr_primitive() }
    }
    #[inline(always)]
    pub fn as_bytes_ptr(&self, ctx: &TypeCtx<B>) -> *mut [u8] {
        match &self.ty {
            Type::Custom(id) => {
                let len = ctx.get(id.clone()).fields.layout().size();
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

/// A struct for allocating single stack values
pub(crate) struct ValStack<B: BuildStatus> {
    alloc: Bump,
    _p: PhantomData<B>,
}

impl<B: BuildStatus> ValStack<B> {
    /// Creates a new `ValStack` with 10MB of capacity
    pub fn new_default() -> Self {
        Self::new(Some(DEFAULT_STACK_CAP.try_into().unwrap()))
    }
    pub fn new(max_capacity: Option<NonZeroUsize>) -> Self {
        let alloc = Bump::new(max_capacity);
        Self {
            alloc,
            _p: PhantomData,
        }
    }
    pub fn capacity(&self) -> usize {
        self.alloc.capacity()
    }
    pub fn allocated_bytes(&self) -> usize {
        self.alloc.allocated_bytes()
    }
    /// Allocating a ZST will result in a dangling allocation, which is expected
    #[inline(always)]
    pub fn alloc_ty(&mut self, ty: impl Into<Type<B>>, ctx: &TypeCtx<B>) -> ValStackHandle<B> {
        let ty = ty.into();
        // SAFETY: `ty.layout()` guarantees correctness
        unsafe { self.alloc_layout(ty.layout(ctx), ty) }
    }
    /// Allocates a handle for the given type and layout
    ///
    /// SAFETY:
    /// * `layout` must represent the correct layout for the type
    #[inline(always)]
    pub unsafe fn alloc_layout(&mut self, layout: Layout, ty: Type<B>) -> ValStackHandle<B> {
        let data = self.alloc.alloc_layout(layout).unwrap();
        ValStackHandle {
            ty,
            data_addr: data.cast(),
        }
    }
}

/// A struct for allocating stack frames
pub(crate) struct FrameStack {
    layout: Bump,
}

impl FrameStack {
    //
}

#[test]
fn test_val_stack_new() {
    use std::hint::black_box;
    fn foo(cap: usize) {
        let a = ValStack::<Building>::new(Some(cap.try_into().unwrap()));
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
    fn foo<T>(v: &mut ValStack<Building>) {
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
    let ctx = TypeCtx::empty();
    let mut v = ValStack::new_default();
    let mut bottom = v.alloc_ty(PrimTy::Num(NumTy::U32), &ctx);
    *unsafe { bottom.as_mut::<u32>() } = 10;
    let mut handles = vec![];

    for i in 0..100 {
        for h in 0..=i {
            let mut handle = v.alloc_ty(PrimTy::Num(NumTy::F64), &ctx);
            // SAFETY: This handle is a primitive `f64`
            let f = unsafe { handle.as_mut::<f64>() };
            *f = h as f64;

            handles.push(handle);
        }
        for h in 0..=i {
            let seek = (i - h) as f64;
            let h = handles.pop().unwrap();
            println_ctx!("{i}");
            // let top = v.pop_primitive_ret(h);
            // let copied: f64 = unsafe { *top.as_ref() };
            // assert_eq!(copied, seek);
        }
        assert!(handles.is_empty())
    }

    println_ctx!();

    // let bot = v.pop_primitive_ret(bottom);
    // let copied: u32 = unsafe { *bot.as_ref() };
    // assert_eq!(copied, 10);
}

pub mod primitives {
    use std::{alloc::Layout, rc::Rc};

    use crate::build_status::{BuildStatus, Building, Built, FinishBuildingCtx};

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
    pub enum PrimTy<B: BuildStatus> {
        Void,
        GcPtr(Box<Type<B>>),
        Num(NumTy),
    }

    impl<B: BuildStatus> PrimTy<B> {
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

    impl PrimTy<Building> {
        pub(crate) fn finish(self, ctx: &FinishBuildingCtx) -> PrimTy<Built> {
            match self {
                PrimTy::Void => PrimTy::Void,
                PrimTy::Num(t) => PrimTy::Num(t),
                PrimTy::GcPtr(t) => PrimTy::GcPtr(Box::new(t.finish(ctx))),
            }
        }
    }
}

/// A layout of multiple typed fields.
/// The position of each field relative to the start of the layout is accessable by the index of that field
#[derive(Debug, Clone)]
pub struct FieldsLayout<B: BuildStatus> {
    layout: Layout,
    value_positions: Vec<(usize, Type<B>)>,
}

impl<B: BuildStatus> FieldsLayout<B> {
    pub fn new_auto(types: Vec<Type<B>>, ctx: &TypeCtx<B>) -> Self {
        let mut curr_size = 0;
        let mut max_align = 1;
        let mut vals = vec![];

        for ty in types {
            let ty_layout = ty.layout(ctx);
            // NOTE: `pos` is a super invalid ptr
            let (pos, _offset) = ptr_ops::align_ptr_up(curr_size, ty_layout.align());
            let pos = pos as usize;

            curr_size = pos + ty_layout.size();
            max_align = max_align.max(ty_layout.align());
            vals.push((pos, ty));
        }

        let layout = Layout::from_size_align(curr_size, max_align).unwrap();

        unsafe { Self::new_raw(layout, vals.into_iter()) }
    }
    /// SAFETY:
    /// * The positions of every value must be non overlapping and fit completely within `layout.size()`
    /// * `layout.align()` must be at least as large as the highest alignment in `value_positions`
    pub unsafe fn new_raw(
        layout: Layout,
        value_positions: impl IntoIterator<Item = (usize, Type<B>)>,
    ) -> Self {
        Self {
            layout,
            value_positions: value_positions.into_iter().collect(),
        }
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// Given that `base` points to the start of an allocation with this layout, gives a pointer to the field
    pub fn ptr_to(&self, field: usize, base: NonNull<()>) -> NonNull<()> {
        unsafe {
            NonNull::new_unchecked(base.as_ptr().offset(self.value_positions[field].0 as isize))
        }
    }
}
