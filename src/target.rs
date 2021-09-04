/// Distributions that can be targeted with the samplers in this crate
///
/// Any distribution targeted by the hmc or nuts sampler needs to implement
/// the `Target` trait.
///
/// ## Examples
///
/// ```
/// struct UnivariateStandardNormal {
///     log_sqrt_2_pi: f64,
/// }
///
/// impl UnivariateStandardNormal {
///     fn new() -> Self {
///         UnivariateStandardNormal {
///             log_sqrt_2_pi: (2.0 * std::f64::consts::PI).sqrt().ln(),
///         }
///     }
/// }
///
/// impl Target<f64> for UnivariateStandardNormal {
///     fn log_density(&self, position: f64) -> f64 {
///         (-0.5 * position * position) - self.log_sqrt_2_pi
///     }
///
///     fn log_density_gradient(&self, position: f64) -> f64 {
///         -position
///     }
/// }
/// ```
pub trait Target<T> {
    // Compute the logarithm of the target density function
    fn log_density(&self, position: T) -> f64;
    // Compute the gradient logarithm of the target density function at a given position
    fn log_density_gradient(&self, position: T) -> T;
}

struct UnivariateStandardNormal {
    log_sqrt_2_pi: f64,
}

impl UnivariateStandardNormal {
    fn new() -> Self {
        UnivariateStandardNormal {
            log_sqrt_2_pi: (2.0 * std::f64::consts::PI).sqrt().ln(),
        }
    }
}

impl Target<f64> for UnivariateStandardNormal {
    fn log_density(&self, position: f64) -> f64 {
        (-0.5 * position * position) - self.log_sqrt_2_pi
    }

    fn log_density_gradient(&self, position: f64) -> f64 {
        -position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_uv_standard_normal() {
        let d = UnivariateStandardNormal::new();
        assert_eq!(d.log_density_gradient(0.0), 0.0);
        assert!(d.log_density(0.0) > d.log_density(1.0));
        assert!(d.log_density(0.0) > d.log_density(-1.0));
    }
}
