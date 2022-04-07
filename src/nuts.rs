use crate::dot::Dot;
use crate::momentum::Momentum;
use crate::target::Target;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::fmt::Debug;
use std::ops::{AddAssign, Mul, Sub};

use std::marker::PhantomData;

pub struct NUTS<D, M, T>
where
    D: Target<T>,
    M: Momentum<T>,
    T: Clone + Copy + Mul<f64, Output = T> + AddAssign + Sub<T, Output = T> + Debug + Dot<T>,
{
    target_density: D,
    momentum_density: M,
    data_type: PhantomData<T>,
    rng: ThreadRng,
}

impl<D, M, T> NUTS<D, M, T>
where
    D: Target<T>,
    M: Momentum<T>,
    T: Clone + Copy + Mul<f64, Output = T> + AddAssign + Sub<T, Output = T> + Debug + Dot<T>,
{
    pub fn new(target_density: D, momentum_density: M) -> Self {
        Self {
            target_density,
            momentum_density,
            data_type: PhantomData,
            rng: thread_rng(),
        }
    }

    pub fn sample(&mut self, position0: T, n_samples: usize, n_adapt: usize) -> Vec<T> {
        let n_total = n_samples + n_adapt;
        let mut step_size = self.find_reasonable_step_size(position0);
        let mut log_av_step_size = 0.;
        let mut av_H = 0.;
        let av_acc_prob = 0.65;
        let shrinkage_target = 10. * step_size;
        let gamma = 0.05;
        let t0 = 10;
        let kappa = 0.75;
        let mut samples: Vec<T> = Vec::with_capacity(n_total);
        let mut init_position = position0;
        let mut init_momentum: T;
        let mut forward_position: T;
        let mut backward_position: T;
        let mut forward_momentum: T;
        let mut backward_momentum: T;
        let mut curr_sample_ix = 0;
        while samples.len() < n_total {
            samples.push(init_position);
            init_momentum = self.momentum_density.sample();
            let u = self.rng.gen_range(
                0.0..self
                    .log_hamiltonian_density(&init_position, &init_momentum)
                    .exp(),
            );
            forward_position = init_position;
            backward_position = init_position;
            forward_momentum = init_momentum;
            backward_momentum = init_momentum;
            let mut j = 0;
            let mut n = 1.;
            let mut n_prime: f64;
            let mut s = true;
            let mut s_prime: bool;
            let mut alpha = 0.;
            let mut n_alpha = 0.;
            while s {
                let v = if self.rng.gen_range(0.0..1.0) < 0.5_f64 {
                    -1
                } else {
                    1
                };
                if v == -1 {
                    (n_prime, s_prime, alpha, n_alpha) = self.build_tree(
                        &mut backward_position,
                        &mut backward_momentum,
                        u,
                        v,
                        j,
                        step_size,
                        &init_position,
                        &init_momentum,
                    );
                } else {
                    (n_prime, s_prime, alpha, n_alpha) = self.build_tree(
                        &mut forward_position,
                        &mut forward_momentum,
                        u,
                        v,
                        j,
                        step_size,
                        &init_position,
                        &init_momentum,
                    );
                }
                if s_prime {
                    let r = n_prime / n;
                    let acc_prob = if r > 1. { 1. } else { r };
                    if self.rng.gen_range(0.0..1.0) <= acc_prob {
                        if v == -1 {
                            samples[curr_sample_ix] = backward_position;
                        } else {
                            samples[curr_sample_ix] = forward_position;
                        }
                    }
                }
                n += n_prime;
                s = s_prime
                    && ((forward_momentum - backward_momentum).dotp(&backward_momentum) >= 0.0)
                    && ((forward_momentum - backward_momentum).dotp(&forward_momentum) >= 0.0);
                j += 1;
            }
            curr_sample_ix += 1;
            init_position = samples[curr_sample_ix];
            // dual averaging
            if curr_sample_ix < n_adapt {
                av_H = (1. - (1. / (curr_sample_ix + t0) as f64)) * av_H
                    + (1. / (curr_sample_ix + t0) as f64) * (av_acc_prob - alpha / n_alpha);
                let log_step_size =
                    shrinkage_target - ((curr_sample_ix as f64).sqrt() / gamma) * av_H;
                let ix_pow_neg_kappa = (curr_sample_ix as f64).powf(-kappa);
                log_av_step_size =
                    ix_pow_neg_kappa * log_step_size + (1 - ix_pow_neg_kappa) * log_av_step_size;
                step_size = log_av_step_size.exp();
            } else {
                step_size = log_av_step_size.exp();
            }
            }
        }
        samples
    }

    fn build_tree(
        &self,
        position: &mut T,
        momentum: &mut T,
        u: f64,
        v: isize,
        j: usize,
        step_size: f64,
        init_position: &T,
        init_momentum: &T,
    ) -> (f64, bool, f64, f64) {
        
        (0., true, 0., 0.)
    }

    fn find_reasonable_step_size(&mut self, mut position: T) -> f64 {
        let mut step_size = 1.;
        let mut momentum = self.momentum_density.sample();
        let initial_momentum = momentum;
        let initial_position = position;
        self.leapfrog(&mut position, &mut momentum, step_size);
        let mut acceptance_prob =
            self.acceptance_probability(&position, &momentum, &initial_position, &initial_momentum);
        let a: u32 = if acceptance_prob > 0.5 { 1 } else { 0 };
        while acceptance_prob.powf(a as f64) > (1. / 2usize.pow(a) as f64) {
            step_size *= 2usize.pow(a) as f64;
            self.leapfrog(&mut position, &mut momentum, step_size);
            acceptance_prob = self.acceptance_probability(
                &position,
                &momentum,
                &initial_position,
                &initial_momentum,
            );
        }
        step_size
    }

    fn leapfrog(&self, position: &mut T, momentum: &mut T, step_size: f64) {
        *momentum += self.target_density.log_density_gradient(position) * (step_size / 2.);
        *position += *momentum * step_size;
        *momentum += self.target_density.log_density_gradient(position) * (step_size / 2.);
        dbg!(position, momentum);
    }

    fn is_accepted(&mut self, acceptance_probability: f64) -> bool {
        self.rng.gen_range(0.0..1.0) < acceptance_probability
    }

    fn acceptance_probability(
        &self,
        new_position: &T,
        new_momentum: &T,
        initial_position: &T,
        initial_momentum: &T,
    ) -> f64 {
        let log_acc_probability = self.log_hamiltonian_density(new_position, new_momentum)
            - self.log_hamiltonian_density(initial_position, initial_momentum);
        if log_acc_probability >= 0. {
            return 1.;
        }
        log_acc_probability.exp()
    }

    fn log_hamiltonian_density(&self, position: &T, momentum: &T) -> f64 {
        self.target_density.log_density(position) + self.momentum_density.log_density(momentum)
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
