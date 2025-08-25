mod mat;
mod vec;

pub use mat::*;
pub use vec::*;

#[derive(Copy, Clone, Default)]
pub struct Rect {
    pub origin: V2,
    pub size: V2,
}
