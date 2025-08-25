use crate::scene::Scene;

use super::Settings;

pub struct Gl {}

impl Gl {
    pub fn new(settings: &Settings) -> Self {
        Self {}
    }

    pub fn draw(&mut self, scene: &Scene) {}
}
