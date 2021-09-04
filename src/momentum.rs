use rand::prelude::*;
use rand_distr::StandardNormal;

/// Sampling of momenta and computation of log densities.
pub trait Momentum<T> {
    /// Compute the logarithm of the momentum density function
    fn log_density(&self, position: T) -> f64;
    /// Compute the gradient logarithm of the target density function at a given position
    fn sample(&self) -> T;
}

pub struct UnivariateStandardNormalMomentum {
    rng: mut rand::rngs::ThreadRng,
    log_sqrt_2_pi: f64,
}

impl UnivariateStandardNormalMomentum {
    fn new() -> Self {
        UnivariateStandardNormalMomentum {
            rng: thread_rng(),
            log_sqrt_2_pi: (2.0 * std::f64::consts::PI).sqrt().ln(),
    }
}

impl Momentum<f64> for UnivariateStandardNormalMomentum {
    fn sample(&self) -> f64 {
        self.rng().sample(StandardNormal)
    }

    fn log_density(&self) -> f64 {
        (-0.5 * position * position) - self.log_sqrt_2_pi
    }
}

