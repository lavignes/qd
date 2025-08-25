use std::ops::Mul;

use bytemuck::{Pod, Zeroable};

use super::{Dot, V3, V4};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Mat4(pub [V4; 4]);

impl Mat4 {
    pub const IDENTITY: Self = Self([
        V4([1.0, 0.0, 0.0, 0.0]),
        V4([0.0, 1.0, 0.0, 0.0]),
        V4([0.0, 0.0, 1.0, 0.0]),
        V4([0.0, 0.0, 0.0, 1.0]),
    ]);
}

macro_rules! vec3_mul_impl {
    ($mat:ty, $vec:ty) => {
        impl Mul<$vec> for $mat {
            type Output = V3;
            #[inline]
            fn mul(self, rhs: $vec) -> Self::Output {
                let rhs = rhs.extended(1.0);
                let [a, b, c, _] = self.0;
                V3([a.dot(rhs), b.dot(rhs), c.dot(rhs)])
            }
        }
    };
}

vec3_mul_impl!(Mat4, V3);
vec3_mul_impl!(&Mat4, V3);
vec3_mul_impl!(&Mat4, &V3);
vec3_mul_impl!(Mat4, &V3);
