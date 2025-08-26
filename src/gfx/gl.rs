use crate::scene::Query;

use super::{Settings, Target};

pub struct Gl {}

impl Gl {
    pub fn new(settings: &Settings) -> Self {
        Self {}
    }

    pub fn draw(&mut self, target: Target, query: Query) {}
}
