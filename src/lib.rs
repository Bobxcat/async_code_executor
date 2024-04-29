#![warn(unsafe_op_in_unsafe_fn)]

mod bump_alloc;
pub mod executor;
pub mod function;
mod gc;
pub mod global_debug;
mod ptr_ops;
pub mod types;
#[macro_use]
mod macros;
mod build_status;

pub fn start() {
    //
}
