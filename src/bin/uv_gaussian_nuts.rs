use rs_hmc::momentum::UnivariateStandardNormalMomentum;
use rs_hmc::nuts::NUTS;
use rs_hmc::target::UnivariateStandardNormal;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let mut nuts = NUTS::new(
        UnivariateStandardNormal::new(),
        UnivariateStandardNormalMomentum::new(),
        8,
    );
    nuts.sample(2., 10000, 1000)
        .iter()
        .for_each(|v| println!("{v}"));
}
