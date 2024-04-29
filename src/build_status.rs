use std::{collections::HashMap, fmt::Debug};

use crate::{
    function::{FuncIdx, FuncName, Function},
    types::{CustomTyIdx, CustomTyName, CustomType, FieldsLayout, Type, TypeCtx},
};

pub trait BuildStatus: Sized + Copy + Debug + 'static {
    type FunctionId: Debug + Clone;
    type CustomTyId: Debug + Clone;

    type CustomTyLayout;
}

#[derive(Debug, Clone, Copy)]
pub struct Building;
impl BuildStatus for Building {
    type FunctionId = FuncName;
    type CustomTyId = CustomTyName;

    type CustomTyLayout = Vec<Type<Building>>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Built;
impl BuildStatus for Built {
    type FunctionId = FuncIdx;
    type CustomTyId = CustomTyIdx;

    type CustomTyLayout = FieldsLayout;
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
    pub fn customs(&self) -> impl Iterator<Item = (&CustomTyName, &CustomTyIdx)> {
        self.types.iter()
    }
}
