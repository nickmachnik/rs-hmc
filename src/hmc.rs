use crate::target::Target;

use std::marker::PhantomData;

struct HMC<D, T>
where
    D: Target<T>,
{
    target: D,
    data_type: PhantomData<T>,
}

impl<D, T> HMC<D, T>
where
    D: Target<T>,
{
    fn new(target: D) -> Self {
        Self {
            target,
            data_type: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{Target, UnivariateStandardNormal};
    use ndarray::{arr1, Array1};

    #[test]
    fn test_new_hmc() {
        let target = UnivariateStandardNormal::new();
        let hmc = HMC::new(target);
        assert_eq!(1, 2);
    }
}
