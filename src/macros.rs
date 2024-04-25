#[macro_export]
macro_rules! println_ctx {
    ($($x:tt)*) => {
        print!("{}", $crate::format_file_ctx!());
        println!($($x)*);
    };
}

#[macro_export]
macro_rules! format_file_ctx {
    () => {
        format!("[{}::{}-{}]", file!(), line!(), column!())
    };
}
