use std::ops::Mul;

use super::{Cross, V3, V4};

#[derive(Copy, Clone)]
pub struct Quat(pub V4);

impl Quat {
    pub const IDENTITY: Self = Self(V4([0.0, 0.0, 0.0, 1.0]));

    #[inline]
    pub fn from_axis_angle(axis: V3, theta: f32) -> Self {
        let V3([nx, ny, nz]) = axis.normalized();
        let half_theta = theta * 0.5;
        let sin_half_theta = half_theta.sin();
        let cos_half_theta = half_theta.cos();
        Self(V4([
            nx * sin_half_theta,
            ny * sin_half_theta,
            nz * sin_half_theta,
            cos_half_theta,
        ]))
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        Self(self.0.normalized())
    }
}

macro_rules! quat_mul_vec3_impl {
    ($lhs:ty, $rhs:ty) => {
        impl Mul<$rhs> for $lhs {
            type Output = V3;
            #[inline]
            fn mul(self, rhs: $rhs) -> V3 {
                let (v3, w) = self.normalized().0.narrowed();
                let conj = -v3;
                let t = conj.cross(rhs) * 2.0;
                rhs + (t * w) + conj.cross(t)
            }
        }
    };
}

quat_mul_vec3_impl!(Quat, V3);
quat_mul_vec3_impl!(&Quat, V3);
quat_mul_vec3_impl!(&Quat, &V3);
quat_mul_vec3_impl!(Quat, &V3);

macro_rules! quat_mul_impl {
    ($lhs:ty, $rhs:ty) => {
        impl Mul<$rhs> for $lhs {
            type Output = Quat;
            #[inline]
            fn mul(self, rhs: $rhs) -> Quat {
                let V4([x1, y1, z1, w1]) = self.0;
                let V4([x2, y2, z2, w2]) = rhs.0;
                Quat(V4([
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ]))
            }
        }
    };
}

quat_mul_impl!(Quat, Quat);
quat_mul_impl!(&Quat, Quat);
quat_mul_impl!(Quat, &Quat);
quat_mul_impl!(&Quat, &Quat);
