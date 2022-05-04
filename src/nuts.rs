use crate::momentum::Momentum;
use crate::target::Target;
use crate::tree_builder::Tree;
use ndarray::Array1;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};

pub const DELTA_MAX: f64 = 1000.;
const KAPPA: f64 = 0.75;
const GAMMA: f64 = 0.05;
const T0: f64 = 10.;
const AV_ACC_PROB: f64 = 0.64;

pub type A = Array1<f64>;

pub struct NUTS<D, M>
where
    D: Target<A>,
    M: Momentum<A, f64>,
{
    tree: Tree<D, M>,
    rng: ThreadRng,
}

impl<D, M> NUTS<D, M>
where
    D: Target<A>,
    M: Momentum<A, f64>,
{
    pub fn new(target_density: D, momentum_density: M, max_subtree_height: usize) -> Self {
        Self {
            tree: Tree::new(target_density, momentum_density, max_subtree_height),
            rng: thread_rng(),
        }
    }

    pub fn sample(&mut self, position0: A, n_samples: usize, n_adapt: usize) -> Vec<A> {
        let n_total = n_samples + n_adapt;
        let mut step_size = self.find_reasonable_step_size(&position0);
        let mut log_av_step_size: f64 = 0.;
        let mut av_h = 0.;
        let shrinkage_target = (10. * step_size).ln();
        let mut samples: Vec<A> = Vec::with_capacity(n_total);
        let mut init_position = position0;
        let mut init_momentum: A;
        let mut forward_position: A;
        let mut backward_position: A;
        let mut forward_momentum: A;
        let mut backward_momentum: A;
        let mut curr_sample_ix = 0;
        while samples.len() < n_total {
            samples.push(init_position.clone());
            init_momentum = self.random_momentum();
            self.tree.set_random_slice(&init_position, &init_momentum);
            self.tree
                .set_reference_state(&init_position, &init_momentum);
            forward_position = init_position.clone();
            backward_position = init_position.clone();
            forward_momentum = init_momentum.clone();
            backward_momentum = init_momentum.clone();
            let mut j = 1;
            let mut n = 1;
            let mut s = true;
            while s {
                let v = if self.rng.gen_range(0.0..1.0) < 0.5_f64 {
                    -1
                } else {
                    1
                };
                if v == -1 {
                    self.tree.build(
                        &mut backward_position,
                        &mut backward_momentum,
                        v,
                        j,
                        step_size,
                    );
                } else {
                    self.tree.build(
                        &mut forward_position,
                        &mut forward_momentum,
                        v,
                        j,
                        step_size,
                    );
                }
                if self.tree.is_valid() {
                    let r = self.tree.num_added_within_slice() as f64 / n as f64;
                    let acc_prob = if r > 1. { 1. } else { r };
                    if self.rng.gen_range(0.0..1.0) <= acc_prob {
                        samples[curr_sample_ix] = self.tree.selected_leaf().clone();
                    }
                }
                n += self.tree.num_added_within_slice();
                s = self.tree.is_valid()
                    && ((&forward_position - &backward_position).dot(&backward_momentum) >= 0.0)
                    && ((&forward_position - &backward_position).dot(&forward_momentum) >= 0.0);
                j += 1;
            }
            init_position = samples[curr_sample_ix].clone();
            curr_sample_ix += 1;
            // dual averaging
            if curr_sample_ix < n_adapt {
                let eta = 1. / (curr_sample_ix as f64 + T0);
                av_h = (1. - eta) * av_h
                    + eta
                        * (AV_ACC_PROB
                            - self.tree.sum_acceptance_probabilities()
                                / self.tree.num_added_total() as f64);
                let log_step_size =
                    shrinkage_target - ((curr_sample_ix as f64).sqrt() / GAMMA) * av_h;
                let ix_pow_neg_kappa = (curr_sample_ix as f64).powf(-KAPPA);
                log_av_step_size =
                    ix_pow_neg_kappa * log_step_size + (1. - ix_pow_neg_kappa) * log_av_step_size;
                step_size = log_step_size.exp();
            } else {
                step_size = log_av_step_size.exp();
            }
        }
        samples[n_adapt..].to_vec()
    }

    fn random_momentum(&mut self) -> A {
        self.tree.random_momentum()
    }

    fn log_target_density_gradient(&self, position: &A) -> A {
        self.tree.log_target_density_gradient(position)
    }

    fn find_reasonable_step_size(&mut self, initial_position: &A) -> f64 {
        let mut step_size = 1.;
        let initial_momentum = self.random_momentum();
        let (mut new_position, mut new_momentum) =
            self.leapfrog(initial_position, &initial_momentum, step_size);
        let mut r = self.tree.hamiltonian_density_ratio(
            &new_position,
            &new_momentum,
            initial_position,
            &initial_momentum,
        );
        let a: f64 = if r > 0.5 { 1. } else { -1. };
        while r.powf(a) > 2_f64.powf(-a) {
            step_size *= 2_f64.powf(a);
            (new_position, new_momentum) =
                self.leapfrog(initial_position, &initial_momentum, step_size);
            r = self.tree.hamiltonian_density_ratio(
                &new_position,
                &new_momentum,
                initial_position,
                &initial_momentum,
            );
        }
        step_size
    }

    fn leapfrog(&self, position: &A, momentum: &A, step_size: f64) -> (A, A) {
        let mut new_momentum = momentum.clone();
        let mut new_position = position.clone();
        new_momentum.scaled_add(step_size / 2., &self.log_target_density_gradient(position));
        new_position.scaled_add(step_size, momentum);
        new_momentum.scaled_add(step_size / 2., &self.log_target_density_gradient(position));
        (new_position, new_momentum)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::momentum::UnivariateStandardNormalMomentum;
//     use crate::target::UnivariateStandardNormal;
//     use approx::assert_abs_diff_eq;

//     #[test]
//     fn test_NUTS_univariate_normal() {
//         let mut NUTS = NUTS::new(
//             UnivariateStandardNormal::new(),
//             UnivariateStandardNormalMomentum::new(),
//         );
//         let samples = NUTS.sample(0.1, 0.01, 100, 1000);
//         let mean = samples.iter().sum::<f64>() / samples.len() as f64;
//         let variance = samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
//         assert_abs_diff_eq!(mean, 0.0);
//         assert_abs_diff_eq!(variance, 1.0);
//     }
// }
