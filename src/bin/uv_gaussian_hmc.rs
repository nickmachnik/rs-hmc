use rs_hmc::hmc::HMC;
use rs_hmc::momentum::UnivariateStandardNormalMomentum;
use rs_hmc::target::UnivariateStandardNormal;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let mut hmc = HMC::new(
        UnivariateStandardNormal::new(),
        UnivariateStandardNormalMomentum::new(),
    );
    hmc.sample(2., 0.05, 200, 1000)
        .iter()
        .for_each(|v| println!("{v}"));
}
