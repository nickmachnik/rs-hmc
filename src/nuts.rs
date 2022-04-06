use crate::momentum::Momentum;
use crate::target::Target;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::fmt::Debug;
use std::ops::{AddAssign, Mul};

use std::marker::PhantomData;

pub struct HMC<D, M, T>
where
    D: Target<T>,
    M: Momentum<T>,
    T: Clone + Copy + Mul<f64, Output = T> + AddAssign + Debug,
{
    target_density: D,
    momentum_density: M,
    data_type: PhantomData<T>,
    rng: ThreadRng,
}

impl<D, M, T> HMC<D, M, T>
where
    D: Target<T>,
    M: Momentum<T>,
    T: Clone + Copy + Mul<f64, Output = T> + AddAssign + Debug,
{
    pub fn new(target_density: D, momentum_density: M) -> Self {
        Self {
            target_density,
            momentum_density,
            data_type: PhantomData,
            rng: thread_rng(),
        }
    }

    pub fn sample(
        &mut self,
        position0: T,
        step_size: f64,
        integration_length: usize,
        n_samples: usize,
    ) -> Vec<T> {
        let mut samples: Vec<T> = Vec::with_capacity(n_samples);
        let mut position_m = position0;
        while samples.len() < n_samples {
            let mut position = position0;
            let momentum_m = self.momentum_density.sample();
            let mut momentum = momentum_m;
            for _ in 0..integration_length {
                self.leapfrog(&mut position, &mut momentum, step_size);
            }
            let acc_prob =
                self.acceptance_probability(&position, &momentum, &position_m, &momentum_m);
            if self.is_accepted(acc_prob) {
                samples.push(position);
                position_m = position;
            }
        }
        samples
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
//     fn test_hmc_univariate_normal() {
//         let mut hmc = HMC::new(
//             UnivariateStandardNormal::new(),
//             UnivariateStandardNormalMomentum::new(),
//         );
//         let samples = hmc.sample(0.1, 0.01, 100, 1000);
//         let mean = samples.iter().sum::<f64>() / samples.len() as f64;
//         let variance = samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
//         assert_abs_diff_eq!(mean, 0.0);
//         assert_abs_diff_eq!(variance, 1.0);
//     }
// }
