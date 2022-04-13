use ndarray::arr1;
use rs_hmc::momentum::MultivariateStandardNormalMomentum;
use rs_hmc::nuts::NUTS;
use rs_hmc::target::MultivariateStandardNormal;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let mut nuts = NUTS::new(
        MultivariateStandardNormal::new(1),
        MultivariateStandardNormalMomentum::new(1),
        8,
    );
    nuts.sample(arr1(&[2.]), 10000, 1000)
        .iter()
        .for_each(|v| println!("{v}"));
}
