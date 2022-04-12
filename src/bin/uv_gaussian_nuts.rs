use rs_hmc::momentum::UnivariateStandardNormalMomentum;
use rs_hmc::nuts::NUTS;
use rs_hmc::target::UnivariateStandardNormal;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let mut nuts = NUTS::new(
        UnivariateStandardNormal::new(),
        UnivariateStandardNormalMomentum::new(),
        100,
    );
    nuts.sample(2., 200, 10)
        .iter()
        .for_each(|v| println!("{v}"));
}
