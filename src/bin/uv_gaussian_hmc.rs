use rs_hmc::HMC;

/// Samples from a standard normal using the libs HMC implementation.
fn main() {
    let mut hmc = HMC::new(
        UnivariateStandardNormal::new(),
        UnivariateStandardNormalMomentum::new(),
    );
    let samples = hmc.sample(0.1, 0.01, 100, 1000);
}
