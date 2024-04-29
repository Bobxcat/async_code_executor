use std::{collections::HashMap, fmt::Debug, marker::PhantomData};

use crate::{
    build_status::{BuildStatus, Building, Built, FinishBuildingCtx},
    types::{primitives::NumTy, CustomTyIdx, CustomTyName, CustomType, OwnedValue, Type, TypeCtx},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncName {
    name: String,
}

impl FuncName {
    pub fn new(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FuncIdx(usize);

impl FuncIdx {
    pub(crate) fn new(n: usize) -> Self {
        Self(n)
    }
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub enum CodePoint<B: BuildStatus> {
    /// For function `F(A, B, C, ..., Z)`
    ///
    /// Sets a starting stack for the new routine, see `CallFunction` for the stack format
    ///
    /// Spawn routine on function
    SpawnRoutine(B::FunctionId),
    /// For function `F(A, B, C, ..., Z)`:
    ///
    /// This functions should expect arguments on the stack:
    /// * Top
    /// * `Z`
    /// * `Y`
    /// * ...
    /// * `B`
    /// * `A`
    /// * Bottom
    ///
    /// Call `F(A, B, C, ..., Z)`
    CallFunction(B::FunctionId),

    // ===Binary Ops===
    /// Pop `b`, Pop `a`
    ///
    /// Both values must be of the given type
    ///
    /// Push `a + b`
    Add(NumTy),
    /// Pop `b`, Pop `a`
    ///
    /// Both values must be of the given type
    ///
    /// Push `a - b`
    Sub(NumTy),
    /// Pop `b`, Pop `a`
    ///
    /// Both values must be of the given type
    ///
    /// Push `a * b`
    Mul(NumTy),
    /// Pop `b`, Pop `a`
    ///
    /// Both values must be of the given type
    ///
    /// Push `a / b`
    Div(NumTy),

    // ===Channels===
    /// Read `a` {`top`}
    ///
    /// `a` must be `RecvChannel`
    ///
    /// Returns the next value received by `a`, does not block executor
    Recv,
    /// Read `a`, Pop `b` {`top - 1`}
    ///
    /// `a` must be `SendChannel`
    ///
    /// Sends `b` over `a`
    Send,

    // ===Stack Manipulation===
    /// Pop `a`
    ///
    /// Panicks if `a` is not Clone
    ///
    /// Push `a`, Push `a`
    Clone,
    /// Push a literal
    Literal(OwnedValue<B>),

    // ===References===
    /// Pop `a`
    ///
    /// Push `&a`
    Allocate,

    // ===Debug===
    /// Read `a` {`top`}
    ///
    /// Panicks if `a` is not the given type
    AssertType(Type<B>),

    /// Read `a` {`top`}
    ///
    /// Formats `a` and prints it
    DebugPrint,
}

#[derive(Debug)]
pub struct Function<B: BuildStatus> {
    pub id: B::FunctionId,
    pub program: Vec<CodePoint<B>>,
    pub params: Vec<Type<B>>,
    pub locals: Vec<Type<B>>,
}

impl Function<Building> {
    pub(crate) fn finish(&self, ctx: &FinishBuildingCtx) -> Function<Built> {
        use CodePoint::*;
        let program = self
            .program
            .iter()
            .cloned()
            .map(|code_point| {
                let new: CodePoint<Built> = match code_point {
                    SpawnRoutine(id) => SpawnRoutine(ctx.func(&id)),
                    CallFunction(id) => SpawnRoutine(ctx.func(&id)),
                    Add(x) => Add(x),
                    Sub(x) => Sub(x),
                    Mul(x) => Mul(x),
                    Div(x) => Div(x),
                    Recv => Recv,
                    Send => Send,
                    Clone => Clone,
                    Literal(l) => Literal(l.finish(ctx)),
                    Allocate => Allocate,
                    AssertType(t) => AssertType(t.finish(ctx)),
                    DebugPrint => DebugPrint,
                };
                new
            })
            .collect::<Vec<CodePoint<Built>>>();
        Function {
            id: ctx.func(&self.id),
            program,
            params: self
                .params
                .clone()
                .into_iter()
                .map(|t| t.finish(ctx))
                .collect(),
            locals: self
                .params
                .clone()
                .into_iter()
                .map(|t| t.finish(ctx))
                .collect(),
        }
    }
}
