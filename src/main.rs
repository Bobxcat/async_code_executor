use std::{alloc::GlobalAlloc, sync::Arc};

use async_code_executor::{
    executor::ExecutorBuilder,
    function::{CodePoint, FuncName, Function},
    types::primitives::NumTy,
};

#[derive(Debug, Clone)]
struct MiMalloc;

unsafe impl GlobalAlloc for MiMalloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        mimalloc::MiMalloc.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        mimalloc::MiMalloc.dealloc(ptr, layout)
    }
}

fn main() {
    // async_code_executor::start();
    // return;

    let allocator = MiMalloc;

    ExecutorBuilder::new()
        .insert_function(Function {
            id: FuncName::new("main"),
            params: vec![],
            program: [
                // CodePoint::Literal(Literal::F32(10.3)),
                CodePoint::DebugPrint,
                // CodePoint::Literal(Literal::F32(-3.)),
                CodePoint::DebugPrint,
                CodePoint::Add(NumTy::F32),
                CodePoint::DebugPrint,
            ]
            .into_iter()
            .collect(),
            locals: vec![],
        })
        .set_allocator(allocator)
        .run();

    print!("Execution over...");
}
