use std::{collections::VecDeque, iter::Filter, slice::Iter};

use crate::math::{Concat, V4, Xform3};

pub struct Scene {
    active_nodes: Vec<u32>,
    nodes: Vec<Node>,
    nodeq: VecDeque<(u32, Xform3)>,
}

impl Scene {
    #[inline]
    pub fn new() -> Scene {
        Scene {
            active_nodes: Vec::new(),
            nodes: Vec::new(),
            nodeq: VecDeque::new(),
        }
    }

    pub fn update(&mut self) {
        if !self.nodes.is_empty() {
            self.active_nodes.clear();
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
                self.active_nodes.push(id);
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
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub sib: u32,
    pub kid: u32,
    pub mesh: u32,
    pub tex: u32,
    pub blend: V4,
    pub local: Xform3,
    pub world: Xform3,
}

impl Node {
    const INACTIVE: u32 = u32::MAX;
    pub const NONE: u32 = 0;

    #[inline]
    pub fn is_active(&self) -> bool {
        self.kid != Self::INACTIVE
    }
}

pub struct Query<'a> {
    iter: Filter<Iter<'a, Node>, fn(&&Node) -> bool>,
}

impl<'a> Query<'a> {
    #[inline]
    pub fn all(scene: &'a Scene) -> Query<'a> {
        Query {
            iter: scene.nodes.iter().filter(|n| n.is_active()),
        }
    }
}

impl<'a> Iterator for Query<'a> {
    type Item = &'a Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
