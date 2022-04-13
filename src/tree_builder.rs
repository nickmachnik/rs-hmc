use crate::dot::Dot;
use crate::math_helpers::*;
use crate::momentum::Momentum;
use crate::nuts::{A, DELTA_MAX};
use crate::target::Target;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};

pub struct Tree<D, M>
where
    D: Target<A>,
    M: Momentum<A>,
{
    target_density: D,
    momentum_density: M,
    // selected leaf for each subtree
    selected_leaves: Vec<A>,
    // number of leaves within slice in each subtree
    num_within_slice: Vec<usize>,
    // leftmost momentum in each subtree
    leftmost_momenti: Vec<A>,
    // leftmost position in each subtree
    leftmost_positions: Vec<A>,
    // sum of acceptance probabilities of all leaves in subtree
    acceptance_probabilities: Vec<f64>,
    // num leaves that have been merged into subtrees
    num_total_leaves: Vec<usize>,
    new_leaf_ix: usize,
    // total number of leaves added to tree
    num_added_total: usize,
    // false if contains u turn or node that does not satisfy delta max critereon
    is_valid: bool,
    direction: isize,
    slice: f64,
    reference_position: A,
    reference_momentum: A,
    rng: ThreadRng,
}

impl<D, M> Tree<D, M>
where
    D: Target<A>,
    M: Momentum<A>,
{
    pub fn new(target_density: D, momentum_density: M, height: usize) -> Self {
        let dim = target_density.dim();
        Self {
            target_density,
            momentum_density,
            selected_leaves: vec![A::default(dim); height],
            num_within_slice: vec![0; height],
            leftmost_momenti: vec![A::default(dim); height],
            leftmost_positions: vec![A::default(dim); height],
            acceptance_probabilities: vec![0.; height],
            num_total_leaves: vec![0; height],
            new_leaf_ix: 0,
            num_added_total: 0,
            is_valid: true,
            direction: 1,
            slice: 0.,
            reference_position: A::default(dim),
            reference_momentum: A::default(dim),
            rng: thread_rng(),
        }
    }

    pub fn log_target_density_gradient(&self, position: &A) -> A {
        self.target_density.log_density_gradient(position)
    }

    pub fn random_momentum(&mut self) -> A {
        self.momentum_density.sample()
    }

    fn acceptance_probability(
        &self,
        new_position: &A,
        new_momentum: &A,
        initial_position: &A,
        initial_momentum: &A,
    ) -> f64 {
        let log_acc_probability = self.neg_hamiltonian(new_position, new_momentum)
            - self.neg_hamiltonian(initial_position, initial_momentum);
        if log_acc_probability >= 0. {
            return 1.;
        }
        log_acc_probability.exp()
    }

    pub fn hamiltonian_density_ratio(
        &self,
        new_position: &A,
        new_momentum: &A,
        initial_position: &A,
        initial_momentum: &A,
    ) -> f64 {
        (self.neg_hamiltonian(new_position, new_momentum)
            - self.neg_hamiltonian(initial_position, initial_momentum))
        .exp()
    }

    // This is -H = (-U) + (-K)
    pub fn neg_hamiltonian(&self, position: &A, momentum: &A) -> f64 {
        self.target_density.log_density(position) + self.momentum_density.log_density(momentum)
    }

    pub fn sum_acceptance_probabilities(&self) -> f64 {
        self.acceptance_probabilities[0]
    }

    pub fn num_added_total(&self) -> usize {
        self.num_total_leaves[0]
    }

    pub fn num_added_within_slice(&self) -> usize {
        self.num_within_slice[0]
    }

    pub fn selected_leaf(&self) -> &A {
        &self.selected_leaves[0]
    }

    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    pub fn set_random_slice(&mut self, position: &A, momentum: &A) {
        self.slice = self
            .rng
            .gen_range(0.0..self.neg_hamiltonian(position, momentum).exp());
    }

    pub fn set_reference_state(&mut self, position: &A, momentum: &A) {
        self.reference_position = position.clone();
        self.reference_momentum = momentum.clone();
    }

    pub fn build(
        &mut self,
        position: &mut A,
        momentum: &mut A,
        direction: isize,
        doubling_round: usize,
        step_size: f64,
    ) {
        self.reset();
        self.set_direction(direction);
        let n_leaves = 2_usize.pow((doubling_round - 1) as u32);
        for _ in 0..n_leaves {
            self.step(position, momentum, step_size);
        }
    }

    fn reset(&mut self) {
        self.is_valid = true;
        self.new_leaf_ix = 0;
        self.num_added_total = 0;
    }

    fn set_direction(&mut self, direction: isize) {
        self.direction = direction;
    }

    fn step(&mut self, position: &mut A, momentum: &mut A, step_size: f64) {
        self.leapfrog(position, momentum, self.direction as f64 * step_size);
        let log_h_density = self.neg_hamiltonian(position, momentum);
        let is_within_slice = self.slice.ln() <= log_h_density;
        let delta_max_satisfied = self.slice.ln() < (DELTA_MAX + log_h_density);
        let acceptance_probability = self.acceptance_probability(
            position,
            momentum,
            &self.reference_position,
            &self.reference_momentum,
        );
        if !delta_max_satisfied {
            self.is_valid = false;
        }
        self.add(position, momentum, is_within_slice, acceptance_probability);
    }

    fn add(
        &mut self,
        position: &A,
        momentum: &A,
        is_within_slice: bool,
        acceptace_probability: f64,
    ) {
        // add new leaf
        self.selected_leaves[self.new_leaf_ix] = position.clone();
        self.leftmost_positions[self.new_leaf_ix] = position.clone();
        self.leftmost_momenti[self.new_leaf_ix] = momentum.clone();
        self.num_within_slice[self.new_leaf_ix] = is_within_slice as usize;
        self.acceptance_probabilities[self.new_leaf_ix] = acceptace_probability;
        self.num_total_leaves[self.new_leaf_ix] = 1;
        self.new_leaf_ix += 1;
        self.num_added_total += 1;
        self.merge();
    }

    fn merge(&mut self) {
        for _ in 0..self.num_necessary_merges() {
            self.new_leaf_ix -= 1;
            let ix_inner = self.new_leaf_ix - 1;
            let ix_outer = self.new_leaf_ix;
            self.is_valid &= !self.is_u_turn(
                &self.leftmost_positions[ix_outer],
                &self.leftmost_momenti[ix_outer],
                &self.leftmost_positions[ix_inner],
                &self.leftmost_momenti[ix_inner],
            );
            let p_accept = self.num_within_slice[ix_outer] as f64
                / (self.num_within_slice[ix_outer] + self.num_within_slice[ix_inner]) as f64;
            if self.rng.gen_range(0.0..1.0) <= p_accept {
                self.selected_leaves[ix_inner] = self.selected_leaves[ix_outer].clone();
            }
            self.acceptance_probabilities[ix_inner] += self.acceptance_probabilities[ix_outer];
            self.num_total_leaves[ix_inner] += self.num_total_leaves[ix_outer];
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
        forward_position: &A,
        forward_momentum: &A,
        backward_position: &A,
        backward_momentum: &A,
    ) -> bool {
        if self.direction == 1 {
            (forward_position - backward_position).dotp(forward_momentum) < 0.
                || (forward_position - backward_position).dotp(backward_momentum) < 0.
        } else {
            (backward_position - forward_position).dotp(forward_momentum) < 0.
                || (backward_position - forward_position).dotp(backward_momentum) < 0.
        }
    }

    fn leapfrog(&self, position: &mut A, momentum: &mut A, step_size: f64) {
        momentum.scaled_add(step_size / 2., &self.log_target_density_gradient(position));
        position.scaled_add(step_size, momentum);
        momentum.scaled_add(step_size / 2., &self.log_target_density_gradient(position));
    }
}
