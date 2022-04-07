use ndarray::Array1;

/// Ability to compute a dot product.
/// Used in NUTS to generalize over univariate and multivariate targets.
pub trait Dot<T> {
    fn dotp(&self, other: &T) -> f64;
}

impl Dot<Array1<f64>> for Array1<f64> {
    fn dotp(&self, other: &Array1<f64>) -> f64 {
        self.dot(other)
    }
}

impl Dot<f64> for f64 {
    fn dotp(&self, other: &f64) -> f64 {
        self * other
    }
}
