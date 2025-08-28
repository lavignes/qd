use super::{Quat, V3};

#[derive(Clone, Copy)]
pub struct Xform3 {
    pub pos: V3,
    pub scale: V3,
    pub rot: Quat,
}

impl Xform3 {
    pub const IDENTITY: Self = Self {
        pos: V3::splat(0.0),
        scale: V3::splat(1.0),
        rot: Quat::IDENTITY,
    };
}

pub trait Concat<Rhs = Self> {
    type Output;
    fn concat(self, rhs: Rhs) -> Self::Output;
}

macro_rules! xform3_concat_impl {
    ($lhs:ty, $rhs:ty) => {
        impl Concat<$rhs> for $lhs {
            type Output = Xform3;
            #[inline]
            fn concat(self, rhs: $rhs) -> Self::Output {
                Xform3 {
                    pos: self.pos + (self.rot * (self.scale * rhs.pos)),
                    scale: self.scale * rhs.scale,
                    rot: self.rot * rhs.rot,
                }
            }
        }
    };
}

xform3_concat_impl!(Xform3, Xform3);
xform3_concat_impl!(&Xform3, Xform3);
xform3_concat_impl!(Xform3, &Xform3);
xform3_concat_impl!(&Xform3, &Xform3);
