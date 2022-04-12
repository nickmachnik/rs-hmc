use crate::dot::Dot;
use crate::math_helpers::*;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::fmt::Debug;
use std::ops::{AddAssign, Mul, Sub};

pub struct Tree<T>
where
    T: Clone
        + Copy
        + Mul<f64, Output = T>
        + AddAssign
        + Sub<T, Output = T>
        + Debug
        + Dot<T>
        + Default,
{
    // selected leaf for each subtree
    selected_leaves: Vec<T>,
    // number of accepted leaves in each subtree, by slice critereon
    num_accepted_leaves: Vec<usize>,
    // leftmost momentum in each subtree
    leftmost_momenti: Vec<T>,
    // leftmost position in each subtree
    leftmost_positions: Vec<T>,
    // sum of acceptance probabilities of all leaves in subtree
    acceptance_probabilities: Vec<f64>,
    // num leaves that have been merged into subtrees
    num_total_leaves: Vec<usize>,
    new_leaf_ix: usize,
    // total number of leaves added to tree
    num_added_total: usize,
    // true if sub tree u turn termination criterion is met within the tree
    contains_u_turn: bool,
    rng: ThreadRng,
}

impl<T> Tree<T>
where
    T: Clone
        + Copy
        + Mul<f64, Output = T>
        + AddAssign
        + Sub<T, Output = T>
        + Debug
        + Dot<T>
        + Default,
{
    pub fn new(height: usize) -> Self {
        Self {
            selected_leaves: vec![T::default(); height],
            num_accepted_leaves: vec![0; height],
            leftmost_momenti: vec![T::default(); height],
            leftmost_positions: vec![T::default(); height],
            acceptance_probabilities: vec![0.; height],
            num_total_leaves: vec![0; height],
            new_leaf_ix: 0,
            num_added_total: 0,
            contains_u_turn: false,
            rng: thread_rng(),
        }
    }

    fn num_necessary_merges(&self) -> usize {
        if is_pow2(self.num_added_total) {
            return log2(self.num_added_total);
        }
        let rem = mod_pow2(self.num_added_total);
        if is_pow2(rem) {
            return log2(rem);
        }
        0
    }

    fn is_u_turn(
        &self,
        forward_position: &T,
        forward_momentum: &T,
        backward_position: &T,
        backward_momentum: &T,
    ) -> bool {
        (*forward_position - *backward_position).dotp(forward_momentum) < 0.
            || (*forward_position - *backward_position).dotp(backward_momentum) < 0.
    }

    fn merge(&mut self) {
        for _ in 0..self.num_necessary_merges() {
            self.new_leaf_ix -= 1;
            let ix_inner = self.new_leaf_ix - 1;
            let ix_outer = self.new_leaf_ix;
            self.contains_u_turn = self.is_u_turn(
                &self.leftmost_positions[ix_outer],
                &self.leftmost_momenti[ix_outer],
                &self.leftmost_positions[ix_inner],
                &self.leftmost_momenti[ix_inner],
            );
            let p_accept = self.num_accepted_leaves[ix_outer] as f64
                / (self.num_accepted_leaves[ix_outer] + self.num_accepted_leaves[ix_inner]) as f64;
            if self.rng.gen_range(0.0..1.0) <= p_accept {
                self.selected_leaves[ix_inner] = self.selected_leaves[ix_outer];
            }
            self.acceptance_probabilities[ix_inner] += self.acceptance_probabilities[ix_outer];
            self.num_total_leaves[ix_inner] += self.num_total_leaves[ix_outer];
        }
    }

    pub fn add(&mut self, position: &T, momentum: &T, new_leaf_stats: (usize, f64, usize)) {
        // add new leaf
        self.selected_leaves[self.new_leaf_ix] = *position;
        self.leftmost_positions[self.new_leaf_ix] = *position;
        self.leftmost_momenti[self.new_leaf_ix] = *momentum;
        self.num_accepted_leaves[self.new_leaf_ix] = new_leaf_stats.0;
        self.acceptance_probabilities[self.new_leaf_ix] = new_leaf_stats.1;
        self.num_total_leaves[self.new_leaf_ix] = new_leaf_stats.2;
        self.new_leaf_ix += 1;
        self.num_added_total += 1;
        self.merge();
    }

    pub fn selected_leaf(&self) -> &T {
        &self.selected_leaves[0]
    }

    pub fn is_valid(&self) -> bool {
        !self.contains_u_turn
    }
}
