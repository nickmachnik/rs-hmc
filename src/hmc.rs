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
            let mut position = position_m;
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

    fn leapfrog(&self, position: &mut T, momentum: &mut T, step_size: f64) {
        *momentum += self.target_density.log_density_gradient(position) * (step_size / 2.);
        *position += *momentum * step_size;
        *momentum += self.target_density.log_density_gradient(position) * (step_size / 2.);
        // dbg!(position, momentum);
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
        let log_acc_probability = self.neg_hamiltonian(new_position, new_momentum)
            - self.neg_hamiltonian(initial_position, initial_momentum);
        if log_acc_probability >= 0. {
            return 1.;
        }
        log_acc_probability.exp()
    }

    // this is -H = (-U) + (-K)
    fn neg_hamiltonian(&self, position: &T, momentum: &T) -> f64 {
        self.target_density.log_density(position) + self.momentum_density.log_density(momentum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::UnivariateStandardNormalMomentum;
    use crate::target::UnivariateStandardNormal;

    #[test]
    fn test_hmc_univariate_normal() {
        let mut hmc = HMC::new(
            UnivariateStandardNormal::new(),
            UnivariateStandardNormalMomentum::new(),
        );
        let n_samples = 10000;
        let samples = hmc.sample(1.2, 0.1, 100, n_samples);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n_samples as f64;
        println!("{:?}", mean);
        println!("{:?}", variance - 1.0);
        assert!(mean.abs() < 0.01);
        assert!((variance - 1.0) < 0.1);
    }
}
