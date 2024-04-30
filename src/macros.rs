#[macro_export]
macro_rules! println_ctx {
    ($($x:tt)*) => {
        print!("{}", $crate::format_file_ctx!());
        println!($($x)*);
    };
}

#[macro_export]
macro_rules! println_ctx_flush {
    ($($x:tt)*) => {
        $crate::flush!();
        $crate::println_ctx!($($x)*);
        $crate::flush!();
    };
}

#[macro_export]
macro_rules! format_file_ctx {
    () => {
        format!("[{}::{}-{}]", file!(), line!(), column!())
    };
}

#[macro_export]
macro_rules! flush {
    () => {
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    };
}
