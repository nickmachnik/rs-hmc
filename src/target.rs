/// Distributions that can be targeted with the samplers in this crate
pub trait Target<T> {
    // Compute the logarithm of the target density function
    fn log_density(position: T) -> f64;
    // Compute the gradient logarithm of the target density function at a given position
    fn log_density_gradient(position: T) -> f64;
}
