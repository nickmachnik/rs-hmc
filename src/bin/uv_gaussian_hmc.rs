use rs_hmc::hmc::HMC;
use rs_hmc::momentum::UnivariateStandardNormalMomentum;
use rs_hmc::target::UnivariateStandardNormal;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let mut hmc = HMC::new(
        UnivariateStandardNormal::new(),
        UnivariateStandardNormalMomentum::new(),
    );
    let _samples = hmc.sample(0.1, 0.01, 100, 1000);
}
