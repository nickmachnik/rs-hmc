use ndarray::arr1;
use rs_hmc::momentum::MultivariateStandardNormalMomentum;
use rs_hmc::nuts::NUTS;
use rs_hmc::target::MultivariateBimodal;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let dim = 2;
    let mut nuts = NUTS::new(
        MultivariateBimodal::new(dim, arr1(&[-2., -2.]), arr1(&[2., 2.])),
        MultivariateStandardNormalMomentum::new(dim),
        8,
    );
    nuts.sample(arr1(&[1., -1.]), 10000, 1000)
        .iter()
        .for_each(|v| println!("{v}"));
}
