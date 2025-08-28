use bytemuck::{Pod, Zeroable};

use crate::{
    math::{UV2, V3, V4},
    scene::Query,
};

#[cfg(feature = "gl")]
mod gl;

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

pub struct Gfx {
    settings: Settings,

    #[cfg(feature = "gl")]
    gl: gl::Gl,
}

impl Gfx {
    #[inline]
    pub fn new(settings: &Settings) -> Self {
        Self {
            settings: *settings,

            #[cfg(feature = "gl")]
            gl: gl::Gl::new(settings),
        }
    }

    #[inline]
    pub fn draw(&mut self, target: Target, query: Query) {
        #[cfg(feature = "gl")]
        self.gl.draw(target, query);
    }
}
