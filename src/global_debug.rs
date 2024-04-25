use std::sync::{Mutex, RwLock};

static GLOBAL_DEBUG: RwLock<bool> = RwLock::new(true);

pub fn is_enabled() -> bool {
    *GLOBAL_DEBUG.read().unwrap()
}

pub fn set_global_debug(new_value: bool) {
    // Do this check to avoid any unnecessary write locking, which is more important than the extra overhead here
    if is_enabled() == new_value {
        return;
    }

    *GLOBAL_DEBUG.write().unwrap() = new_value;
}
