use std::collections::VecDeque;

use crate::{
    gfx::Drawable,
    math::{Concat, Xform3},
};

pub struct Scene {
    nodes: Vec<Node>,
    nodeq: VecDeque<(u32, Xform3)>,
}

impl Scene {
    #[inline]
    pub fn new() -> Scene {
        Scene {
            nodes: Vec::new(),
            nodeq: VecDeque::new(),
        }
    }

    pub fn update(&mut self) {
        if !self.nodes.is_empty() {
            self.nodeq.push_back((0, Xform3::IDENTITY));
        }
        while let Some((mut id, ref xform)) = self.nodeq.pop_front() {
            loop {
                let node = &mut self.nodes[id as usize];
                if !node.is_active() {
                    continue;
                }
                let world = xform.concat(node.local);
                node.world = world;
                if node.kid != Node::NONE {
                    self.nodeq.push_back((node.kid, world));
                }
                id = node.sib;
                if id == Node::NONE {
                    break;
                }
            }
        }
    }

    pub fn add_node(&mut self, node: Node) {
        for n in &mut self.nodes {
            if n.is_active() {
                continue;
            }
            *n = node;
            return;
        }
        self.nodes.push(node);
    }

    pub fn active_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.nodes.iter_mut().filter(|node| node.is_active())
    }

    pub fn drawables<'a>(&'a self) -> impl Iterator<Item = (&'a Xform3, &'a Drawable)> {
        self.nodes
            .iter()
            .filter(|node| node.is_active() && node.draw.is_some())
            .map(|node| (&node.world, &node.draw))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub sib: u32,
    pub kid: u32,
    pub local: Xform3,
    pub world: Xform3,
    pub draw: Drawable,
}

impl Node {
    const INACTIVE: u32 = u32::MAX;
    pub const NONE: u32 = 0;

    #[inline]
    pub fn is_active(&self) -> bool {
        self.kid != Self::INACTIVE
    }
}
