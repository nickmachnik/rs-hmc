use ndarray::Array1;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Sampling of momenta and computation of log densities.
pub trait Momentum<T> {
    /// Compute the logarithm of the momentum density function
    fn log_density(&self, position: &T) -> f64;
    fn sample(&mut self) -> T;
}

pub struct UnivariateStandardNormalMomentum {
    rng: rand::rngs::ThreadRng,
    log_sqrt_2_pi: f64,
}

impl UnivariateStandardNormalMomentum {
    pub fn new() -> Self {
        Self {
            rng: thread_rng(),
            log_sqrt_2_pi: (2.0 * std::f64::consts::PI).sqrt().ln(),
        }
    }
}

impl Momentum<f64> for UnivariateStandardNormalMomentum {
    fn sample(&mut self) -> f64 {
        self.rng.sample(StandardNormal)
    }

    fn log_density(&self, position: &f64) -> f64 {
        (-0.5 * position * position) - self.log_sqrt_2_pi
    }
}

pub struct MultivariaStandardNormalMomentum {
    rng: rand::rngs::ThreadRng,
    ndim: usize,
}

impl MultivariaStandardNormalMomentum {
    pub fn new(ndim: usize) -> Self {
        Self {
            rng: thread_rng(),
            ndim,
        }
    }
}

impl Momentum<Array1<f64>> for MultivariaStandardNormalMomentum {
    fn sample(&mut self) -> Array1<f64> {
        let mut res = Array1::from(vec![0.; self.ndim]);
        for ix in 0..self.ndim {
            res[ix] = self.rng.sample(StandardNormal);
        }
        res
    }

    fn log_density(&self, position: &Array1<f64>) -> f64 {
        -0.5 * position.dot(position)
    }
}
