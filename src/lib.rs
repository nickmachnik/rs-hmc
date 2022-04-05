//! Implementations of the standard Hamiltonian Monte Carlo and No U-Turn Sampler algorithms.

extern crate blas_src;
extern crate openblas_src;

mod hmc;
mod momentum;
mod nuts;
mod target;
