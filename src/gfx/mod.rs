use crate::{math::UV2, scene::Scene};

#[cfg(feature = "gl")]
mod gl;

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

    pub fn draw(&mut self, scene: &Scene) {
        #[cfg(feature = "gl")]
        self.gl.draw(scene);
    }
}
