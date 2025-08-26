use crate::{math::UV2, scene::Query};

#[cfg(feature = "gl")]
mod gl;

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
    pub fn new(settings: &Settings) -> Self {
        Self {
            settings: *settings,

            #[cfg(feature = "gl")]
            gl: gl::Gl::new(settings),
        }
    }

    pub fn draw(&mut self, target: Target, query: Query) {
        #[cfg(feature = "gl")]
        self.gl.draw(target, query);
    }
}
