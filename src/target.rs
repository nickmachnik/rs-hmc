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
    fn dim(&self) -> usize;
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

    fn dim(&self) -> usize {
        1
    }
}

#[derive(Copy, Clone)]
pub struct MultivariateStandardNormal {
    dim: usize,
}

impl MultivariateStandardNormal {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Target<Array1<f64>> for MultivariateStandardNormal {
    fn log_density(&self, position: &Array1<f64>) -> f64 {
        -0.5 * position.dot(position)
    }

    fn log_density_gradient(&self, position: &Array1<f64>) -> Array1<f64> {
        -position
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Clone)]
pub struct MultivariateBimodal {
    dim: usize,
    mu1: Array1<f64>,
    mu2: Array1<f64>,
}

impl MultivariateBimodal {
    pub fn new(dim: usize, mu1: Array1<f64>, mu2: Array1<f64>) -> Self {
        Self { dim, mu1, mu2 }
    }
}

impl Target<Array1<f64>> for MultivariateBimodal {
    fn log_density(&self, position: &Array1<f64>) -> f64 {
        let diff_a = position - &self.mu1;
        let diff_b = position - &self.mu2;
        ((-0.5 * diff_a.dot(&diff_a)).exp() + (-0.5 * diff_b.dot(&diff_b)).exp())
            .ln()
            .max(-100.)
    }

    fn log_density_gradient(&self, position: &Array1<f64>) -> Array1<f64> {
        let delta = 0.00001;
        let curr_d = self.log_density(position);
        let mut grad = Array1::zeros(self.dim());
        let mut position1 = position.clone();
        for ix in 0..self.dim() {
            position1[ix] = position[ix] + delta;
            grad[ix] = (self.log_density(&position1) - curr_d) / delta;
            position1[ix] = position[ix];
        }
        grad
    }

    fn dim(&self) -> usize {
        self.dim
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
        let d = MultivariateStandardNormal::new(2);
        assert_abs_diff_eq!(d.log_density_gradient(&arr1(&[0., 0.])), arr1(&[0., 0.]));
        assert!(d.log_density(&arr1(&[0., 0.])) > d.log_density(&arr1(&[1., 0.])));
        assert!(d.log_density(&arr1(&[0., 0.])) > d.log_density(&arr1(&[-1., 0.])));
    }
}
