use std::ops::Mul;

use bytemuck::{Pod, Zeroable};

use super::{Dot, Quat, V3, V4, Xform3};

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

    #[inline]
    pub fn transposed(&self) -> Self {
        let [a, b, c, d] = self.0;
        Mat4([
            V4([a.0[0], b.0[0], c.0[0], d.0[0]]),
            V4([a.0[1], b.0[1], c.0[1], d.0[1]]),
            V4([a.0[2], b.0[2], c.0[2], d.0[2]]),
            V4([a.0[3], b.0[3], c.0[3], d.0[3]]),
        ])
    }
}

macro_rules! from_quat_impl {
    ($quat:ty) => {
        impl From<$quat> for Mat4 {
            #[inline]
            fn from(quat: $quat) -> Self {
                let V4([x, y, z, w]) = quat.0.normalized();
                let xx = x * x;
                let xy = x * y;
                let xz = x * z;
                let xw = x * w;
                let yy = y * y;
                let yz = y * z;
                let yw = y * w;
                let zz = z * z;
                let zw = z * w;
                Mat4([
                    V4([1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw), 0.0]),
                    V4([2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw), 0.0]),
                    V4([2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy), 0.0]),
                    V4([0.0, 0.0, 0.0, 1.0]),
                ])
            }
        }
    };
}

from_quat_impl!(Quat);
from_quat_impl!(&Quat);

macro_rules! from_xform3_impl {
    ($xform:ty) => {
        impl From<$xform> for Mat4 {
            #[inline]
            fn from(xform: $xform) -> Self {
                let V3([px, py, pz]) = xform.pos;
                let V3([sx, sy, sz]) = xform.scale;
                let V4([x, y, z, w]) = xform.rot.0.normalized();
                let xx = x * x;
                let xy = x * y;
                let xz = x * z;
                let xw = x * w;
                let yy = y * y;
                let yz = y * z;
                let yw = y * w;
                let zz = z * z;
                let zw = z * w;
                Mat4([
                    V4([
                        sx * (1.0 - 2.0 * (yy + zz)),
                        sx * (2.0 * (xy + zw)),
                        sx * (2.0 * (xz - yw)),
                        px,
                    ]),
                    V4([
                        sy * (2.0 * (xy - zw)),
                        sy * (1.0 - 2.0 * (xx + zz)),
                        sy * (2.0 * (yz + xw)),
                        py,
                    ]),
                    V4([
                        sz * (2.0 * (xz + yw)),
                        sz * (2.0 * (yz - xw)),
                        sz * (1.0 - 2.0 * (xx + yy)),
                        pz,
                    ]),
                    V4([0.0, 0.0, 0.0, 1.0]),
                ])
            }
        }
    };
}

from_xform3_impl!(Xform3);
from_xform3_impl!(&Xform3);

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
