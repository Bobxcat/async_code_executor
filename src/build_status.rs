use std::{collections::HashMap, fmt::Debug};

use crate::{
    function::{FuncIdx, FuncName, Function},
    types::{CustomTyIdx, CustomTyName, CustomType, TypeCtx},
};

pub trait CustomTyStore<B: BuildStatus>: Default {
    fn get(&self, id: B::CustomTyId) -> Option<&CustomType<B>>;
}

#[derive(Debug, Default)]
pub struct CustomTyMap {
    m: HashMap<CustomTyName, CustomType<Building>>,
}

impl CustomTyMap {
    pub fn insert(&mut self, ty: CustomType<Building>) {
        self.m.insert(ty.id.clone(), ty);
    }
}

impl CustomTyStore<Building> for CustomTyMap {
    fn get(&self, id: <Building as BuildStatus>::CustomTyId) -> Option<&CustomType<Building>> {
        self.m.get(&id)
    }
}

#[derive(Debug, Default)]
pub struct CustomTyVec {
    v: Vec<CustomType<Built>>,
}

impl CustomTyStore<Built> for CustomTyVec {
    fn get(&self, id: <Built as BuildStatus>::CustomTyId) -> Option<&CustomType<Built>> {
        self.v.get(id.get())
    }
}

pub trait BuildStatus: Sized + Copy + Debug + 'static {
    type FunctionId: Debug + Clone;
    type CustomTyId: Debug + Clone;

    type CustomTyStore: CustomTyStore<Self>;
}

#[derive(Debug, Clone, Copy)]
pub struct Building;
impl BuildStatus for Building {
    type FunctionId = FuncName;
    type CustomTyId = CustomTyName;

    type CustomTyStore = CustomTyMap;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Built;
impl BuildStatus for Built {
    type FunctionId = FuncIdx;
    type CustomTyId = CustomTyIdx;

    type CustomTyStore = CustomTyVec;
}

pub struct FinishBuildingCtx {
    types: HashMap<CustomTyName, CustomTyIdx>,
    funcs: HashMap<FuncName, FuncIdx>,
}

impl FinishBuildingCtx {
    pub fn from_names(
        types: impl IntoIterator<Item = CustomTyName>,
        funcs: impl IntoIterator<Item = FuncName>,
    ) -> Self {
        let types = types
            .into_iter()
            .enumerate()
            .map(|(idx, n)| (n, CustomTyIdx::new(idx)))
            .collect::<HashMap<_, _>>();

        let funcs = funcs
            .into_iter()
            .enumerate()
            .map(|(idx, n)| (n, FuncIdx::new(idx)))
            .collect::<HashMap<_, _>>();

        Self { types, funcs }
    }
    pub fn custom(&self, id: &CustomTyName) -> CustomTyIdx {
        self.types[&id].clone()
    }
    pub fn func(&self, id: &FuncName) -> FuncIdx {
        self.funcs[&id].clone()
    }
}
