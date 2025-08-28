pub struct Query<'a> {
    scene: &'a Scene,
}

impl<'a> Query<'a> {
    #[inline]
    pub fn all(scene: &'a Scene) -> Self {
        Query { scene }
    }
}

pub struct Scene {}
