use std::fmt::Debug;

use crate::{
    executor::{Literal, SizedTy, SizedVal},
    types::primitives::NumTy,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FuncIdx(pub usize);

pub trait FunctionId: Debug + Clone {
    //
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
    Literal(Literal),

    // ===References===
    /// Pop `a`
    ///
    /// Push `&a`
    Allocate,

    // ===Debug===
    /// Read `a` {`top`}
    ///
    /// Panicks if `a` is not the given type
    AssertType(SizedTy),

    /// Read `a` {`top`}
    ///
    /// Formats `a` and prints it
    DebugPrint,
}

pub trait BuildStatus {
    type FunctionId: Debug + Clone;
}

#[derive(Debug, Clone, Copy)]
pub struct Building;
impl BuildStatus for Building {
    type FunctionId = FuncName;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Built;
impl BuildStatus for Built {
    type FunctionId = FuncIdx;
}

#[derive(Debug)]
pub struct Function<B: BuildStatus> {
    pub id: FuncName,
    pub program: Vec<CodePoint<B>>,
    pub params: Vec<SizedTy>,
    pub locals: Vec<SizedTy>,
}

pub(crate) struct FunctionFrame {
    pub id: FuncName,
    pub params: Vec<SizedVal>,
    pub locals: Vec<SizedVal>,
}
