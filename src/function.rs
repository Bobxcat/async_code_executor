use crate::{
    executor::{Literal, SizedTy, SizedVal},
    types::primitives::NumTy,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionId {
    name: String,
}

impl FunctionId {
    pub fn new(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CodePoint {
    /// For function `F(A, B, C, ..., Z)`
    ///
    /// Sets a starting stack for the new routine, see `CallFunction` for the stack format
    ///
    /// Spawn routine on function
    SpawnRoutine(FunctionId),
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
    CallFunction(FunctionId),

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

#[derive(Debug)]
pub struct Function {
    pub id: FunctionId,
    pub program: Vec<CodePoint>,
    pub params: Vec<SizedTy>,
    pub locals: Vec<SizedTy>,
}

pub(crate) struct FunctionFrame {
    pub id: FunctionId,
    pub params: Vec<SizedVal>,
    pub locals: Vec<SizedVal>,
}
