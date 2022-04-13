use ndarray::arr1;
use rs_hmc::momentum::MultivariateStandardNormalMomentum;
use rs_hmc::nuts::NUTS;
use rs_hmc::target::MultivariateStandardNormal;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let mut nuts = NUTS::new(
        MultivariateStandardNormal::new(2),
        MultivariateStandardNormalMomentum::new(2),
        8,
    );
    nuts.sample(arr1(&[2., 5.]), 0, 10)
        .iter()
        .for_each(|v| println!("{v}"));
}
