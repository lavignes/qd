use bytemuck::{Pod, Zeroable};
use gl::{BufMap, TexMap};

use crate::math::{Mat4, UV2, V3, V4, Xform3};

#[cfg(feature = "gl")]
mod gl;

pub struct Gfx {
    #[cfg(feature = "gl")]
    gl: gl::Gl,
}

impl Gfx {
    #[inline]
    pub fn new(settings: &Settings) -> Self {
        Self {
            #[cfg(feature = "gl")]
            gl: gl::Gl::new(settings),
        }
    }

    #[inline]
    pub fn pass<'a>(&'a mut self, target: Target, camera: &'a Camera) -> Pass<'a> {
        Pass {
            #[cfg(feature = "gl")]
            gl: self.gl.pass(target, camera),
        }
    }

    #[inline]
    pub fn mesh_alloc(&mut self, verts: usize, idxs: usize) -> u32 {
        #[cfg(feature = "gl")]
        self.gl.mesh_alloc(verts, idxs)
    }

    #[inline]
    pub fn mesh_map<'a>(&'a mut self, hnd: u32) -> (BufMap<'a, Vtx>, BufMap<'a, u32>) {
        #[cfg(feature = "gl")]
        self.gl.mesh_map(hnd)
    }

    #[inline]
    pub fn tex_alloc(&mut self) -> u32 {
        #[cfg(feature = "gl")]
        self.gl.tex_alloc()
    }

    #[inline]
    pub fn tex_map<'a>(&'a mut self, hnd: u32) -> TexMap<'a> {
        #[cfg(feature = "gl")]
        self.gl.tex_map(hnd)
    }
}

pub struct Pass<'a> {
    #[cfg(feature = "gl")]
    gl: gl::Pass<'a>,
}

impl<'a> Pass<'a> {
    #[inline]
    pub fn clear_all(&mut self) {
        #[cfg(feature = "gl")]
        self.gl.clear_all();
    }

    #[inline]
    pub fn draw<'b, I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (&'b Xform3, &'b Drawable)>,
    {
        #[cfg(feature = "gl")]
        self.gl.draw(iter);
    }
}

#[derive(Clone, Copy)]
pub enum Proj {
    Ortho {
        size: UV2,
        near: f32,
        far: f32,
    },
    Persp {
        fov: f32,
        ratio: f32,
        near: f32,
        far: f32,
    },
}

macro_rules! from_proj_impl {
    ($proj:ty) => {
        impl From<$proj> for Mat4 {
            #[inline]
            fn from(proj: $proj) -> Self {
                match proj {
                    Proj::Ortho { size, near, far } => {
                        let depth = far - near;
                        Mat4([
                            V4([2.0 / (size.0[0] as f32), 0.0, 0.0, 0.0]),
                            V4([0.0, -2.0 / (size.0[1] as f32), 0.0, 0.0]),
                            V4([0.0, 0.0, 2.0 / depth, 0.0]),
                            V4([-1.0, 1.0, 0.0, 1.0]),
                        ])
                    }
                    Proj::Persp {
                        fov,
                        ratio,
                        near,
                        far,
                    } => {
                        let depth = far - near;
                        let tan_half_fov = (fov * 0.5).tan();
                        Mat4([
                            V4([1.0 / (tan_half_fov * ratio), 0.0, 0.0, 0.0]),
                            V4([0.0, 1.0 / tan_half_fov, 0.0, 0.0]),
                            V4([0.0, 0.0, -(far + near) / depth, -1.0]),
                            V4([0.0, 0.0, -(2.0 * far * near) / depth, 0.0]),
                        ])
                    }
                }
            }
        }
    };
}

from_proj_impl!(Proj);
from_proj_impl!(&Proj);

pub struct Camera {
    pub pos: V3,
    pub at: V3,
    pub proj: Proj,
}

#[derive(Clone, Copy, Debug)]
pub enum Drawable {
    None,
    Mesh { hnd: u32, tex: u32, blend: V4 },
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct Vtx {
    pub pos: V3,
    pub tx: f32,
    pub norm: V3,
    pub ty: f32,
    pub color: V4,
}

pub enum Target {
    Screen,
    Tex(u32),
}

#[derive(Clone, Copy)]
pub struct Settings {
    pub size: UV2,
}
