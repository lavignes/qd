#![deny(unused_imports)]

pub mod gfx;
pub mod log;
pub mod math;
pub mod mem;
pub mod scene;

/// Log a FATAL error and exit the progam.
#[macro_export]
macro_rules! fatal {
    ($($arg:tt)+) => ({
        ::log::log!(target: "FATAL", ::log::Level::Error, $($arg)+);
        unreachable!()
    });
}

/// Check a `Result` and log a FATAL error if it is not `Ok`.
#[macro_export]
macro_rules! ensure {
    ($result:expr, $($arg:tt)+) => (
        match $result {
            Err(err) => $crate::fatal!($($arg)+, err),
            Ok(val) => val,
        }
    );
    ($result:expr) => ($crate::ensure!($result, "{}"));
}

/// Ensure a `Result` is `Ok`, otherwise consume the `Err`
/// (to prepare and log it) and then `panic!`.
#[inline]
pub fn ensure<T, E, F>(res: Result<T, E>, func: F) -> T
where
    F: FnOnce(E) -> (),
{
    match res {
        Err(err) => {
            func(err);
            fatal!("Failed to abort")
        }
        Ok(val) => val,
    }
}
