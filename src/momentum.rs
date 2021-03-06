use ndarray::Array1;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Sampling of momenta and computation of log densities.
pub trait Momentum<T, F> {
    /// Compute the logarithm of the momentum density function
    fn log_density(&self, position: &T) -> F;
    fn sample(&mut self) -> T;
}

#[derive(Clone)]
pub struct UnivariateStandardNormalMomentum {
    rng: rand::rngs::ThreadRng,
}

impl UnivariateStandardNormalMomentum {
    pub fn new() -> Self {
        Self { rng: thread_rng() }
    }
}

impl Momentum<f64, f64> for UnivariateStandardNormalMomentum {
    fn sample(&mut self) -> f64 {
        self.rng.sample(StandardNormal)
    }

    fn log_density(&self, position: &f64) -> f64 {
        -0.5 * position * position
    }
}

#[derive(Clone)]
pub struct MultivariateStandardNormalMomentum {
    rng: rand::rngs::ThreadRng,
    ndim: usize,
}

impl MultivariateStandardNormalMomentum {
    pub fn new(ndim: usize) -> Self {
        Self {
            rng: thread_rng(),
            ndim,
        }
    }
}

impl Momentum<Array1<f64>, f64> for MultivariateStandardNormalMomentum {
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

impl Momentum<Array1<f32>, f32> for MultivariateStandardNormalMomentum {
    fn sample(&mut self) -> Array1<f32> {
        let mut res = Array1::from(vec![0.; self.ndim]);
        for ix in 0..self.ndim {
            res[ix] = self.rng.sample(StandardNormal);
        }
        res
    }

    fn log_density(&self, position: &Array1<f32>) -> f32 {
        -0.5 * position.dot(position)
    }
}
