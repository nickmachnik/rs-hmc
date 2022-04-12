use ndarray::Array1;

/// Distributions that can be targeted with the samplers in this crate
///
/// Any distribution targeted by the hmc or nuts sampler needs to implement
/// the `Target` trait.
///
/// ## Examples
///
/// ```
/// use rs_hmc::target::Target;
///
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
///     fn log_density(&self, position: &f64) -> f64 {
///         (-0.5 * position * position) - self.log_sqrt_2_pi
///     }
///
///     fn log_density_gradient(&self, position: &f64) -> f64 {
///         -position
///     }
/// }
/// ```
pub trait Target<T> {
    /// Compute the logarithm of the target density function
    fn log_density(&self, position: &T) -> f64;
    /// Compute the gradient logarithm of the target density function at a given position
    fn log_density_gradient(&self, position: &T) -> T;
}

#[derive(Copy, Clone)]
pub struct UnivariateStandardNormal {
    log_sqrt_2_pi: f64,
}

impl UnivariateStandardNormal {
    pub fn new() -> Self {
        Self {
            log_sqrt_2_pi: (2.0 * std::f64::consts::PI).sqrt().ln(),
        }
    }
}

impl Target<f64> for UnivariateStandardNormal {
    fn log_density(&self, position: &f64) -> f64 {
        (-0.5 * position * position) - self.log_sqrt_2_pi
    }

    fn log_density_gradient(&self, position: &f64) -> f64 {
        -position
    }
}

#[derive(Copy, Clone)]
pub struct MultivariateStandardNormal {}

impl MultivariateStandardNormal {
    pub fn new() -> Self {
        Self {}
    }
}

impl Target<Array1<f64>> for MultivariateStandardNormal {
    fn log_density(&self, position: &Array1<f64>) -> f64 {
        -0.5 * position.dot(position)
    }

    fn log_density_gradient(&self, position: &Array1<f64>) -> Array1<f64> {
        -position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;

    #[test]
    fn test_uv_standard_normal() {
        let d = UnivariateStandardNormal::new();
        assert_eq!(d.log_density_gradient(&0.0), 0.0);
        assert!(d.log_density(&0.0) > d.log_density(&1.0));
        assert!(d.log_density(&0.0) > d.log_density(&-1.0));
    }

    #[test]
    fn test_mv_standard_normal() {
        let d = MultivariateStandardNormal::new();
        assert_abs_diff_eq!(d.log_density_gradient(&arr1(&[0., 0.])), arr1(&[0., 0.]));
        assert!(d.log_density(&arr1(&[0., 0.])) > d.log_density(&arr1(&[1., 0.])));
        assert!(d.log_density(&arr1(&[0., 0.])) > d.log_density(&arr1(&[-1., 0.])));
    }
}
