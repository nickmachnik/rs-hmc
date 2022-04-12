use crate::dot::Dot;
use crate::momentum::Momentum;
use crate::target::Target;
use crate::tree_builder::Tree;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Sub};

pub const DELTA_MAX: f64 = 1000.;
const KAPPA: f64 = 0.75;
const GAMMA: f64 = 0.05;
const T0: f64 = 10.;
const AV_ACC_PROB: f64 = 0.64;

pub struct NUTS<D, M, T>
where
    D: Target<T>,
    M: Momentum<T>,
    T: Clone
        + Copy
        + Mul<f64, Output = T>
        + AddAssign
        + Sub<T, Output = T>
        + Debug
        + Dot<T>
        + Default,
{
    data_type: PhantomData<T>,
    tree: Tree<D, M, T>,
    rng: ThreadRng,
}

impl<D, M, T> NUTS<D, M, T>
where
    D: Target<T>,
    M: Momentum<T>,
    T: Clone
        + Copy
        + Mul<f64, Output = T>
        + AddAssign
        + Sub<T, Output = T>
        + Debug
        + Dot<T>
        + Default,
{
    pub fn new(target_density: D, momentum_density: M, max_subtree_height: usize) -> Self {
        Self {
            data_type: PhantomData,
            tree: Tree::new(target_density, momentum_density, max_subtree_height),
            rng: thread_rng(),
        }
    }

    pub fn sample(&mut self, position0: T, n_samples: usize, n_adapt: usize) -> Vec<T> {
        let n_total = n_samples + n_adapt;
        let mut step_size = self.find_reasonable_step_size(&position0);
        let mut log_av_step_size: f64 = 0.;
        let mut av_h = 0.;
        let shrinkage_target = (10. * step_size).ln();
        let mut samples: Vec<T> = Vec::with_capacity(n_total);
        let mut init_position = position0;
        let mut init_momentum: T;
        let mut forward_position: T;
        let mut backward_position: T;
        let mut forward_momentum: T;
        let mut backward_momentum: T;
        let mut curr_sample_ix = 0;
        dbg!(step_size);
        while samples.len() < n_total {
            samples.push(init_position);
            init_momentum = self.random_momentum();
            self.tree.set_random_slice(&init_position, &init_momentum);
            self.tree
                .set_reference_state(&init_position, &init_momentum);
            forward_position = init_position;
            backward_position = init_position;
            forward_momentum = init_momentum;
            backward_momentum = init_momentum;
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
                        samples[curr_sample_ix] = *self.tree.selected_leaf();
                    }
                }
                n += self.tree.num_added_within_slice();
                dbg!(self.tree.is_valid());
                s = self.tree.is_valid()
                    && ((forward_position - backward_position).dotp(&backward_momentum) >= 0.0)
                    && ((forward_position - backward_position).dotp(&forward_momentum) >= 0.0);
                dbg!(
                    forward_position,
                    backward_position,
                    forward_momentum,
                    backward_momentum
                );
                dbg!(s);
                j += 1;
            }
            init_position = samples[curr_sample_ix];
            curr_sample_ix += 1;
            // dual averaging
            if curr_sample_ix < n_adapt {
                dbg!(
                    self.tree.sum_acceptance_probabilities(),
                    self.tree.num_added_total()
                );
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
                dbg!(av_h, step_size, log_step_size, log_av_step_size);
            } else {
                step_size = log_av_step_size.exp();
            }
        }
        samples[n_adapt..].to_vec()
    }

    fn random_momentum(&mut self) -> T {
        self.tree.random_momentum()
    }

    fn log_target_density_gradient(&self, position: &T) -> T {
        self.tree.log_target_density_gradient(position)
    }

    fn find_reasonable_step_size(&mut self, initial_position: &T) -> f64 {
        let mut step_size = 1.;
        let initial_momentum = self.random_momentum();
        let (mut new_position, mut new_momentum) =
            self.leapfrog(&initial_position, &initial_momentum, step_size);
        let mut r = self.tree.hamiltonian_density_ratio(
            &new_position,
            &new_momentum,
            &initial_position,
            &initial_momentum,
        );
        dbg!(r);
        let a: f64 = if r > 0.5 { 1. } else { -1. };
        while r.powf(a) > 2_f64.powf(-a) {
            step_size *= 2_f64.powf(a);
            (new_position, new_momentum) =
                self.leapfrog(&initial_position, &initial_momentum, step_size);
            r = self.tree.hamiltonian_density_ratio(
                &new_position,
                &new_momentum,
                &initial_position,
                &initial_momentum,
            );
            dbg!(r);
        }
        step_size
    }

    fn leapfrog(&self, position: &T, momentum: &T, step_size: f64) -> (T, T) {
        let mut new_momentum = *momentum;
        let mut new_position = *position;
        new_momentum += self.log_target_density_gradient(position) * (step_size / 2.);
        new_position += *momentum * step_size;
        new_momentum += self.log_target_density_gradient(position) * (step_size / 2.);
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
