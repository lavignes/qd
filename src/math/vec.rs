use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use bytemuck::{Pod, Zeroable};

pub trait Dot<Rhs = Self> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

pub trait Cross<Rhs = Self> {
    type Output;
    fn cross(self, rhs: Rhs) -> Self::Output;
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct V2(pub [f32; 2]);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct IV2(pub [i32; 2]);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct UV2(pub [u32; 2]);

impl V2 {
    #[inline]
    pub fn length(&self) -> f32 {
        self.normal_squared().sqrt()
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        self / self.length()
    }
}

impl From<V2> for IV2 {
    #[inline]
    fn from(value: V2) -> Self {
        let [x, y] = value.0;
        Self([x as i32, y as i32])
    }
}

impl From<IV2> for V2 {
    #[inline]
    fn from(value: IV2) -> Self {
        let [x, y] = value.0;
        Self([x as f32, y as f32])
    }
}

impl From<UV2> for IV2 {
    #[inline]
    fn from(value: UV2) -> Self {
        let [x, y] = value.0;
        Self([x as i32, y as i32])
    }
}

macro_rules! vec2_impl {
    ($vec:ident, $scalar:ident) => {
        impl $vec {
            #[inline]
            pub const fn splat(s: $scalar) -> $vec {
                $vec([s, s])
            }

            #[inline]
            pub fn normal_squared(&self) -> $scalar {
                self.dot(self)
            }
        }
    };
}

macro_rules! vec2_neg {
    ($vec:ident, $scalar:ident) => {
        impl Neg for $vec {
            type Output = $vec;

            #[inline]
            fn neg(self) -> Self::Output {
                let [x, y] = self.0;
                $vec([-x, -y])
            }
        }
    };
}

macro_rules! vec2_binop {
    ($vec:ident, $scalar:ident, $op_trait:ident, $op_name:ident) => {
        impl $op_trait<$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $vec) -> Self::Output {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2)])
            }
        }

        impl $op_trait<$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $vec) -> Self::Output {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2)])
            }
        }

        impl $op_trait<&$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: &$vec) -> Self::Output {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2)])
            }
        }

        impl $op_trait<&$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: &$vec) -> Self::Output {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2)])
            }
        }

        impl $op_trait<$scalar> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $scalar) -> Self::Output {
                let [x, y] = self.0;
                $vec([x.$op_name(rhs), y.$op_name(rhs)])
            }
        }

        impl $op_trait<$scalar> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $scalar) -> Self::Output {
                let [x, y] = self.0;
                $vec([x.$op_name(rhs), y.$op_name(rhs)])
            }
        }
    };
}

macro_rules! vec2_dot {
    ($vec:ident, $scalar:ident) => {
        impl Dot<$vec> for $vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: $vec) -> $scalar {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                (x1 * x2) + (y1 * y2)
            }
        }

        impl Dot<&$vec> for $vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: &$vec) -> $scalar {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                (x1 * x2) + (y1 * y2)
            }
        }

        impl Dot<$vec> for &$vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: $vec) -> $scalar {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                (x1 * x2) + (y1 * y2)
            }
        }

        impl Dot<&$vec> for &$vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: &$vec) -> $scalar {
                let [x1, y1] = self.0;
                let [x2, y2] = rhs.0;
                (x1 * x2) + (y1 * y2)
            }
        }
    };
}

vec2_impl!(V2, f32);
vec2_dot!(V2, f32);
vec2_neg!(V2, f32);
vec2_binop!(V2, f32, Add, add);
vec2_binop!(V2, f32, Sub, sub);
vec2_binop!(V2, f32, Mul, mul);
vec2_binop!(V2, f32, Div, div);
vec2_binop!(V2, f32, Rem, rem);

vec2_impl!(IV2, i32);
vec2_dot!(IV2, i32);
vec2_neg!(IV2, i32);
vec2_binop!(IV2, i32, Add, add);
vec2_binop!(IV2, i32, Sub, sub);
vec2_binop!(IV2, i32, Mul, mul);
vec2_binop!(IV2, i32, Div, div);
vec2_binop!(IV2, i32, Rem, rem);

vec2_impl!(UV2, u32);
vec2_dot!(UV2, u32);
vec2_binop!(UV2, u32, Add, add);
vec2_binop!(UV2, u32, Sub, sub);
vec2_binop!(UV2, u32, Mul, mul);
vec2_binop!(UV2, u32, Div, div);
vec2_binop!(UV2, u32, Rem, rem);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct V3(pub [f32; 3]);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct IV3(pub [i32; 3]);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct UV3(pub [u32; 3]);

impl V3 {
    pub const UP: Self = Self([0.0, 1.0, 0.0]);
    pub const DOWN: Self = Self([0.0, -1.0, 0.0]);
    pub const LEFT: Self = Self([-1.0, 0.0, 0.0]);
    pub const RIGHT: Self = Self([1.0, 0.0, 0.0]);
    pub const FORWARD: Self = Self([0.0, 0.0, 1.0]);
    pub const BACKWARD: Self = Self([0.0, 0.0, -1.0]);

    #[inline]
    pub fn length(&self) -> f32 {
        self.normal_squared().sqrt()
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        self / self.length()
    }

    #[inline]
    pub const fn extended(&self, s: f32) -> V4 {
        let [x, y, z] = self.0;
        V4([x, y, z, s])
    }
}

macro_rules! vec3_impl {
    ($vec:ident, $scalar:ident) => {
        impl $vec {
            #[inline]
            pub const fn splat(s: $scalar) -> $vec {
                $vec([s, s, s])
            }

            #[inline]
            pub fn normal_squared(&self) -> $scalar {
                self.dot(self)
            }
        }
    };
}

macro_rules! vec3_neg {
    ($vec:ident, $scalar:ident) => {
        impl Neg for $vec {
            type Output = $vec;
            #[inline]
            fn neg(self) -> Self::Output {
                let [x, y, z] = self.0;
                $vec([-x, -y, -z])
            }
        }
    };
}

macro_rules! vec3_binop {
    ($vec:ident, $scalar:ident, $op_trait:ident, $op_name:ident) => {
        impl $op_trait<$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $vec) -> Self::Output {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2), z1.$op_name(z2)])
            }
        }

        impl $op_trait<$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $vec) -> Self::Output {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2), z1.$op_name(z2)])
            }
        }

        impl $op_trait<&$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: &$vec) -> Self::Output {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2), z1.$op_name(z2)])
            }
        }

        impl $op_trait<&$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: &$vec) -> Self::Output {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([x1.$op_name(x2), y1.$op_name(y2), z1.$op_name(z2)])
            }
        }

        impl $op_trait<$scalar> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $scalar) -> Self::Output {
                let [x, y, z] = self.0;
                $vec([x.$op_name(rhs), y.$op_name(rhs), z.$op_name(rhs)])
            }
        }

        impl $op_trait<$scalar> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $scalar) -> Self::Output {
                let [x, y, z] = self.0;
                $vec([x.$op_name(rhs), y.$op_name(rhs), z.$op_name(rhs)])
            }
        }
    };
}

macro_rules! vec3_dot {
    ($vec:ident, $scalar:ident) => {
        impl Dot<$vec> for $vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: $vec) -> $scalar {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2)
            }
        }

        impl Dot<&$vec> for $vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: &$vec) -> $scalar {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2)
            }
        }

        impl Dot<$vec> for &$vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: $vec) -> $scalar {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2)
            }
        }

        impl Dot<&$vec> for &$vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: &$vec) -> $scalar {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2)
            }
        }
    };
}

macro_rules! vec3_cross {
    ($vec:ident) => {
        impl Cross<$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn cross(self, rhs: $vec) -> $vec {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([
                    (y1 * z2) - (z1 * y2),
                    (z1 * x2) - (x1 * z2),
                    (x1 * y2) - (y1 * x2),
                ])
            }
        }

        impl Cross<&$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn cross(self, rhs: &$vec) -> $vec {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([
                    (y1 * z2) - (z1 * y2),
                    (z1 * x2) - (x1 * z2),
                    (x1 * y2) - (y1 * x2),
                ])
            }
        }

        impl Cross<$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn cross(self, rhs: $vec) -> $vec {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([
                    (y1 * z2) - (z1 * y2),
                    (z1 * x2) - (x1 * z2),
                    (x1 * y2) - (y1 * x2),
                ])
            }
        }

        impl Cross<&$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn cross(self, rhs: &$vec) -> $vec {
                let [x1, y1, z1] = self.0;
                let [x2, y2, z2] = rhs.0;
                $vec([
                    (y1 * z2) - (z1 * y2),
                    (z1 * x2) - (x1 * z2),
                    (x1 * y2) - (y1 * x2),
                ])
            }
        }
    };
}

vec3_impl!(V3, f32);
vec3_dot!(V3, f32);
vec3_cross!(V3);
vec3_neg!(V3, f32);
vec3_binop!(V3, f32, Add, add);
vec3_binop!(V3, f32, Sub, sub);
vec3_binop!(V3, f32, Mul, mul);
vec3_binop!(V3, f32, Div, div);
vec3_binop!(V3, f32, Rem, rem);

vec3_impl!(IV3, i32);
vec3_dot!(IV3, i32);
vec3_cross!(IV3);
vec3_neg!(IV3, i32);
vec3_binop!(IV3, i32, Add, add);
vec3_binop!(IV3, i32, Sub, sub);
vec3_binop!(IV3, i32, Mul, mul);
vec3_binop!(IV3, i32, Div, div);
vec3_binop!(IV3, i32, Rem, rem);

vec3_impl!(UV3, u32);
vec3_dot!(UV3, u32);
vec3_cross!(UV3);
vec3_binop!(UV3, u32, Add, add);
vec3_binop!(UV3, u32, Sub, sub);
vec3_binop!(UV3, u32, Mul, mul);
vec3_binop!(UV3, u32, Div, div);
vec3_binop!(UV3, u32, Rem, rem);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct V4(pub [f32; 4]);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct IV4(pub [i32; 4]);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
pub struct UV4(pub [u32; 4]);

impl V4 {
    #[inline]
    pub fn length(&self) -> f32 {
        self.normal_squared().sqrt()
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        self / self.length()
    }

    #[inline]
    pub const fn narrowed(&self) -> (V3, f32) {
        let [x, y, z, w] = self.0;
        (V3([x, y, z]), w)
    }
}

macro_rules! vec4_impl {
    ($vec:ident, $scalar:ident) => {
        impl $vec {
            #[inline]
            pub const fn splat(s: $scalar) -> $vec {
                $vec([s, s, s, s])
            }

            #[inline]
            pub fn normal_squared(&self) -> $scalar {
                self.dot(self)
            }
        }
    };
}

macro_rules! vec4_neg {
    ($vec:ident, $scalar:ident) => {
        impl Neg for $vec {
            type Output = $vec;

            #[inline]
            fn neg(self) -> Self::Output {
                let [x, y, z, w] = self.0;
                $vec([-x, -y, -z, -w])
            }
        }
    };
}

macro_rules! vec4_binop {
    ($vec:ident, $scalar:ident, $op_trait:ident, $op_name:ident) => {
        impl $op_trait<$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $vec) -> Self::Output {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                $vec([
                    x1.$op_name(x2),
                    y1.$op_name(y2),
                    z1.$op_name(z2),
                    w1.$op_name(w2),
                ])
            }
        }

        impl $op_trait<$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $vec) -> Self::Output {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                $vec([
                    x1.$op_name(x2),
                    y1.$op_name(y2),
                    z1.$op_name(z2),
                    w1.$op_name(w2),
                ])
            }
        }

        impl $op_trait<&$vec> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: &$vec) -> Self::Output {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                $vec([
                    x1.$op_name(x2),
                    y1.$op_name(y2),
                    z1.$op_name(z2),
                    w1.$op_name(w2),
                ])
            }
        }

        impl $op_trait<&$vec> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: &$vec) -> Self::Output {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                $vec([
                    x1.$op_name(x2),
                    y1.$op_name(y2),
                    z1.$op_name(z2),
                    w1.$op_name(w2),
                ])
            }
        }

        impl $op_trait<$scalar> for $vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $scalar) -> Self::Output {
                let [x, y, z, w] = self.0;
                $vec([
                    x.$op_name(rhs),
                    y.$op_name(rhs),
                    z.$op_name(rhs),
                    w.$op_name(rhs),
                ])
            }
        }

        impl $op_trait<$scalar> for &$vec {
            type Output = $vec;
            #[inline]
            fn $op_name(self, rhs: $scalar) -> Self::Output {
                let [x, y, z, w] = self.0;
                $vec([
                    x.$op_name(rhs),
                    y.$op_name(rhs),
                    z.$op_name(rhs),
                    w.$op_name(rhs),
                ])
            }
        }
    };
}

macro_rules! vec4_dot {
    ($vec:ident, $scalar:ident) => {
        impl Dot<$vec> for $vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: $vec) -> $scalar {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2) + (w1 * w2)
            }
        }

        impl Dot<&$vec> for $vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: &$vec) -> $scalar {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2) + (w1 * w2)
            }
        }

        impl Dot<$vec> for &$vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: $vec) -> $scalar {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2) + (w1 * w2)
            }
        }

        impl Dot<&$vec> for &$vec {
            type Output = $scalar;
            #[inline]
            fn dot(self, rhs: &$vec) -> $scalar {
                let [x1, y1, z1, w1] = self.0;
                let [x2, y2, z2, w2] = rhs.0;
                (x1 * x2) + (y1 * y2) + (z1 * z2) + (w1 * w2)
            }
        }
    };
}

vec4_impl!(V4, f32);
vec4_dot!(V4, f32);
vec4_neg!(V4, f32);
vec4_binop!(V4, f32, Add, add);
vec4_binop!(V4, f32, Sub, sub);
vec4_binop!(V4, f32, Mul, mul);
vec4_binop!(V4, f32, Div, div);
vec4_binop!(V4, f32, Rem, rem);

vec4_impl!(IV4, i32);
vec4_dot!(IV4, i32);
vec4_neg!(IV4, i32);
vec4_binop!(IV4, i32, Add, add);
vec4_binop!(IV4, i32, Sub, sub);
vec4_binop!(IV4, i32, Mul, mul);
vec4_binop!(IV4, i32, Div, div);
vec4_binop!(IV4, i32, Rem, rem);

vec4_impl!(UV4, u32);
vec4_dot!(UV4, u32);
vec4_binop!(UV4, u32, Add, add);
vec4_binop!(UV4, u32, Sub, sub);
vec4_binop!(UV4, u32, Mul, mul);
vec4_binop!(UV4, u32, Div, div);
vec4_binop!(UV4, u32, Rem, rem);
