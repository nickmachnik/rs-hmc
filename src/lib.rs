//! Implementations of the standard Hamiltonian Monte Carlo and No U-Turn Sampler algorithms.

extern crate blas_src;
extern crate openblas_src;

pub mod hmc;
pub mod momentum;
pub mod nuts;
pub mod target;
