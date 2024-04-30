use crate::{build_status::FinishBuildingCtx, ptr_ops::AlignedBytes};
use std::{
    alloc::Layout, collections::HashMap, marker::PhantomData, num::NonZeroUsize, ptr::NonNull,
};

use crate::{
    build_status::{BuildStatus, Building, Built},
    bump_alloc::Bump,
    global_debug, println_ctx,
    ptr_ops::{self},
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
    #[inline(always)]
    pub(crate) fn new(idx: usize) -> Self {
        Self(idx)
    }
    #[inline(always)]
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct CustomType<B: BuildStatus> {
    pub(crate) id: B::CustomTyId,
    pub(crate) fields: B::CustomTyLayout,
}

impl CustomType<Building> {
    pub fn new(name: impl AsRef<str>, fields: Vec<Type<Building>>) -> Self {
        Self {
            id: name.into(),
            fields,
        }
    }
}

/// Information about the custom types that currently exist
#[derive(Debug)]
pub struct TypeCtx {
    customs: Vec<CustomType<Built>>,
}

impl TypeCtx {
    pub fn empty() -> Self {
        Self {
            customs: Default::default(),
        }
    }
    pub fn build(
        ctx: &FinishBuildingCtx,
        type_data: impl IntoIterator<Item = (CustomTyName, CustomType<Building>)>,
    ) -> Self {
        #[derive(Debug)]
        struct CustomUnfinished {
            _name: CustomTyName,
            _idx: CustomTyIdx,
            fields: Vec<Type<Built>>,
        }
        let mut customs_unfinished: HashMap<CustomTyIdx, CustomUnfinished> = HashMap::new();
        let mut customs_finished: HashMap<CustomTyIdx, CustomType<Built>> = HashMap::new();
        for (id, ty) in type_data {
            let idx = ctx.custom(&id);
            let field_types = ty
                .fields
                .into_iter()
                .map(|t| t.finish(ctx))
                .collect::<Vec<_>>();

            customs_unfinished.insert(
                idx,
                CustomUnfinished {
                    _name: id,
                    _idx: idx,
                    fields: field_types,
                },
            );
        }

        loop {
            // Whether or not a type has been finished this cycle
            let mut has_finished_type = false;

            customs_unfinished.retain(|&idx, unfinished| {
                let retain: bool;
                match FieldsLayout::try_new_auto(unfinished.fields.clone(), |idx| {
                    customs_finished.get(&idx).map(|t| t.fields.layout())
                }) {
                    Some(fields) => {
                        has_finished_type = true;
                        retain = false;
                        customs_finished.insert(idx, CustomType { id: idx, fields });
                    }
                    None => retain = true,
                }

                retain
            });

            if customs_unfinished.len() == 0 {
                break;
            }

            if !has_finished_type {
                panic!("Panic when building type context: Could not resolve layout of the following types:
                =====
                {:#?}
                =====
                Make sure that there are no circular custom type layouts! If you need circular types, use indirection
                ", customs_unfinished);
            }
        }

        assert_eq!(customs_unfinished.len(), 0);
        assert_eq!(customs_finished.len(), ctx.customs().count());

        let mut customs = customs_finished
            .into_iter()
            .map(|(_idx, ty)| ty)
            .collect::<Vec<_>>();
        customs.sort_unstable_by_key(|t| t.id);

        for i in 0..customs.len() {
            assert_eq!(i, customs[i].id.get())
        }

        TypeCtx { customs }
    }
}

impl TypeCtx {
    #[inline(always)]
    pub(crate) fn get(&self, ty: CustomTyIdx) -> &CustomType<Built> {
        self.customs.get(ty.get()).unwrap()
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
}

impl Type<Built> {
    /// Tries to get the layout of this type from the given mapping
    ///
    /// Keep in mind:
    /// * A pointer to a custom type does *not* need to go through `maybe_layout`
    #[inline(always)]
    fn try_get_layout(
        &self,
        maybe_layout: &impl Fn(CustomTyIdx) -> Option<Layout>,
    ) -> Option<Layout> {
        match self {
            Type::Custom(id) => maybe_layout(*id),
            Type::Primitive(t) => Some(t.layout()),
        }
    }
    #[inline(always)]
    pub fn layout(&self, ctx: &TypeCtx) -> Layout {
        match &self {
            Type::Custom(t) => ctx.get(*t).fields.layout(),
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

impl<B: BuildStatus> Eq for Type<B> {}

impl<B: BuildStatus, R> PartialEq<R> for Type<B>
where
    R: Clone + Into<Type<B>>,
{
    #[inline(always)]
    fn eq(&self, other: &R) -> bool {
        let other = &other.clone().into();
        match (self, other) {
            (Self::Custom(l0), Self::Custom(r0)) => l0 == r0,
            (Self::Primitive(l0), Self::Primitive(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl<B: BuildStatus> From<PrimTy<B>> for Type<B> {
    fn from(value: PrimTy<B>) -> Self {
        Self::Primitive(value)
    }
}

impl<B: BuildStatus> From<NumTy> for Type<B> {
    fn from(value: NumTy) -> Self {
        Self::Primitive(PrimTy::Num(value))
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
pub struct TypeVal {
    ty: Type<Built>,
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

pub trait IntoLiteral {
    fn into_literal(self) -> OwnedValue<Building>;
}

impl IntoLiteral for f32 {
    fn into_literal(self) -> OwnedValue<Building> {
        let b = f32::to_ne_bytes(self);
        unsafe { OwnedValue::new_raw(NumTy::F32.into(), &b, std::mem::align_of::<Self>()) }
    }
}

impl IntoLiteral for f64 {
    fn into_literal(self) -> OwnedValue<Building> {
        let b = f64::to_ne_bytes(self);
        unsafe { OwnedValue::new_raw(NumTy::F64.into(), &b, std::mem::align_of::<Self>()) }
    }
}

impl IntoLiteral for i32 {
    fn into_literal(self) -> OwnedValue<Building> {
        let b = i32::to_ne_bytes(self);
        unsafe { OwnedValue::new_raw(NumTy::I32.into(), &b, std::mem::align_of::<Self>()) }
    }
}

impl IntoLiteral for u32 {
    fn into_literal(self) -> OwnedValue<Building> {
        let b = u32::to_ne_bytes(self);
        unsafe { OwnedValue::new_raw(NumTy::U32.into(), &b, std::mem::align_of::<Self>()) }
    }
}

/// A free-standing owned allocation of a given type
///
/// An instance of `OwnedValue` is assumed to be a valid instance of its `Type`
#[derive(Debug, Clone)]
pub struct OwnedValue<B: BuildStatus> {
    ty: Type<B>,
    raw: AlignedBytes,
}

impl<B: BuildStatus> OwnedValue<B> {
    pub fn ty(&self) -> Type<B> {
        self.ty.clone()
    }
    /// Does not check to make sure this type is valid,
    /// and `align` must be a power of 2 that is at least as large as the alignment of `ty`
    ///
    /// Clones `bytes` into a new buffer
    pub unsafe fn new_raw(ty: Type<B>, bytes: &[u8], align: usize) -> Self {
        let mut raw = AlignedBytes::alloc(bytes.len(), align);
        raw.clone_from_slice(bytes);
        Self { ty, raw }
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
    pub fn new<T: IntoLiteral>(val: T) -> Self {
        val.into_literal()
    }
    pub(crate) fn finish(self, ctx: &FinishBuildingCtx) -> OwnedValue<Built> {
        OwnedValue {
            ty: self.ty.finish(ctx),
            raw: self.raw,
        }
    }
}

// impl OwnedValue<Built> {
//     pub(crate) fn try_add()
// }

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
pub struct ValStackEntry {
    ty: Type<Built>,
    data_addr: NonNull<u8>,
}

impl ValStackEntry {
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
pub(crate) struct ValStack {
    alloc: Bump,
    allocated: Vec<ValStackEntry>,
}

impl std::fmt::Debug for ValStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValStack")
            .field("allocated_bytes", &self.alloc.allocated_bytes())
            .field("capacity", &self.alloc.capacity())
            .field("allocated", &self.allocated)
            .finish()
    }
}

impl ValStack {
    /// Creates a new `ValStack` with 10MB of capacity
    pub fn new_default() -> Self {
        Self::new(Some(Bump::DEFAULT_CAPACITY.try_into().unwrap()))
    }
    pub fn new(max_capacity: Option<NonZeroUsize>) -> Self {
        let alloc = Bump::new(max_capacity);
        Self {
            alloc,
            allocated: vec![],
        }
    }
    pub fn capacity(&self) -> usize {
        self.alloc.capacity()
    }
    pub fn allocated_bytes(&self) -> usize {
        self.alloc.allocated_bytes()
    }
    /// The current number of allocations
    pub fn len(&self) -> usize {
        self.allocated.len()
    }
    pub fn push_literal(&mut self, lit: OwnedValue<Built>, ctx: &TypeCtx) {
        self.alloc_ty(lit.ty, ctx)
    }
    /// Allocating a ZST will result in a dangling allocation, which is expected
    #[inline(always)]
    pub fn alloc_ty(&mut self, ty: impl Into<Type<Built>>, ctx: &TypeCtx) {
        let ty = ty.into();
        // SAFETY: `ty.layout()` guarantees correctness
        unsafe { self.alloc_layout(ty.layout(ctx), ty) }
    }
    /// Allocates a handle for the given type and layout
    ///
    /// SAFETY:
    /// * `layout` must represent the correct layout for the type
    #[inline(always)]
    pub unsafe fn alloc_layout(&mut self, layout: Layout, ty: Type<Built>) {
        let data = self.alloc.alloc_layout(layout).unwrap();
        let entry = ValStackEntry {
            ty,
            data_addr: data.cast(),
        };
        println_ctx!("\n{entry:?}");
        self.allocated.push(entry);
    }
    /// Gets all the specified allocations, indexed from the top.
    /// Checks for uniqueness of each `idx` element using a basic `O(n^2)` algorithm, which should
    /// be fast enough for small numbers of elements and may be optimized away for a statically known `idx`
    #[inline(always)]
    pub fn peekn<'a>(&'a mut self, idx: &[usize]) -> Vec<&'a mut ValStackEntry> {
        for i in 1..idx.len() {
            if idx[i..].contains(&idx[i - 1]) {
                panic!("Called `peekn` with duplicate indices:\n{idx:?}");
            }
        }
        for i in idx {
            if *i >= self.allocated.len() {
                panic!(
                    "Called `peekn` with an index beyond the number of allocations `{}`:\n{idx:?}",
                    self.allocated.len()
                )
            }
        }
        // SAFETY: We just checked for uniqueness of `idx`
        unsafe { self.peekn_unckecked(idx) }
    }
    /// Gets all the specified allocations, indexed from the top
    ///
    /// SAFETY:
    /// * `idx` must not contain any duplicates
    /// * Every element in `idx` must be within the number of allocations
    pub unsafe fn peekn_unckecked(&mut self, idx: &[usize]) -> Vec<&mut ValStackEntry> {
        self.allocated
            .iter_mut()
            .rev()
            .enumerate()
            .filter(|(this_idx, _)| idx.contains(this_idx))
            .take(idx.len()) // Only take a certain number, which will be a great shortcut
            .map(|(_, elem)| elem)
            .collect::<Vec<_>>()
    }

    pub fn peek<'a>(&'a mut self, idx: usize) -> &'a mut ValStackEntry {
        self.peekn(&[idx]).pop().unwrap()
    }
    pub fn pop_count(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        if count >= self.allocated.len() {
            panic!(
                "Called `pop_count({count})` with `{}` allocations left",
                self.allocated.len()
            );
        }
        for _ in 0..count - 1 {
            let _ = self.allocated.pop().unwrap();
        }
        let new_head = self.allocated.pop().unwrap().data_addr;
        self.alloc.dealloc_to(new_head.cast());
    }
    pub fn pop(&mut self) {
        self.pop_count(1);
    }
    pub fn pop_ret(&mut self, ctx: &TypeCtx) -> Option<OwnedValue<Built>> {
        if self.len() == 0 {
            return None;
        }
        let top = self.peek(0);
        // SAFETY: The source allocation and type are known to be valid
        let ret = unsafe {
            OwnedValue::new_raw(
                top.ty.clone(),
                top.as_bytes(ctx),
                top.ty.layout(ctx).align(),
            )
        };

        self.pop_count(1);

        Some(ret)
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
        unsafe { v.alloc_layout(layout, PrimTy::Void.into()) };

        let delta_bytes = v.allocated_bytes() - start_bytes;
        assert!(
            delta_bytes >= layout.size(),
            "delta_bytes={delta_bytes} < layout.size()={}",
            layout.size()
        );

        let val = v.peek(0);
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
    v.alloc_ty(PrimTy::Num(NumTy::U32), &ctx);
    let bottom = v.peek(0);
    *unsafe { bottom.as_mut::<u32>() } = 10;

    for i in 0..12 {
        for h in 0..=i {
            v.alloc_ty(PrimTy::Num(NumTy::F64), &ctx);
            let entry = v.peek(0);
            // SAFETY: This entry is a primitive `f64`
            let f = unsafe { entry.as_mut::<f64>() };
            *f = h as f64;
        }
        for h in (0..=i).rev() {
            let entry = v.pop_ret(&ctx).unwrap();
            // SAFETY: This entry is a primitive `f64`
            let f = unsafe { entry.as_ref::<f64>() };
            println_ctx!("{i},{h}={f}",);
        }
        assert!(v.len() == 1);
    }

    println_ctx!();
}

pub mod primitives {
    use std::{alloc::Layout, rc::Rc, sync::Arc};

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
        GcPtr(Arc<Type<B>>),
        Num(NumTy),
    }

    impl<B: BuildStatus> Eq for PrimTy<B> {}

    impl<B: BuildStatus> PartialEq for PrimTy<B> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::GcPtr(l0), Self::GcPtr(r0)) => l0 == r0,
                (Self::Num(l0), Self::Num(r0)) => l0 == r0,
                _ => core::mem::discriminant(self) == core::mem::discriminant(other),
            }
        }
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
                PrimTy::GcPtr(t) => {
                    let t = Arc::unwrap_or_clone(t);
                    PrimTy::GcPtr(Arc::new(t.finish(ctx)))
                }
            }
        }
    }
}

/// A layout of multiple typed fields.
/// The position of each field relative to the start of the layout is accessable by the index of that field
#[derive(Debug, Clone)]
pub struct FieldsLayout {
    layout: Layout,
    value_positions: Vec<(usize, Type<Built>)>,
}

impl FieldsLayout {
    /// Returns `None` iff any calls to `maybe_layout` for a type in `fields` yielded `None`
    pub(crate) fn try_new_auto(
        fields: Vec<Type<Built>>,
        maybe_layout: impl Fn(CustomTyIdx) -> Option<Layout>,
    ) -> Option<Self> {
        let mut curr_size = 0;
        let mut max_align = 1;
        let mut vals = vec![];

        for ty in fields {
            let ty_layout = ty.try_get_layout(&maybe_layout)?;
            // NOTE: `pos` is a super invalid ptr
            let (pos, _offset) = ptr_ops::align_ptr_up(curr_size, ty_layout.align());
            let pos = pos as usize;

            curr_size = pos + ty_layout.size();
            max_align = max_align.max(ty_layout.align());
            vals.push((pos, ty));
        }

        let layout = Layout::from_size_align(curr_size, max_align).unwrap();

        Some(unsafe { Self::new_raw(layout, vals.into_iter()) })
    }
    pub(crate) fn new_auto(fields: Vec<Type<Built>>, ctx: &TypeCtx) -> Self {
        Self::try_new_auto(fields, |idx| Some(ctx.get(idx).fields.layout())).unwrap()
    }
    /// SAFETY:
    /// * The positions of every value must be non overlapping and fit completely within `layout.size()`
    /// * `layout.align()` must be at least as large as the highest alignment in `value_positions`
    pub(crate) unsafe fn new_raw(
        layout: Layout,
        value_positions: impl IntoIterator<Item = (usize, Type<Built>)>,
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
