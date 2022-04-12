pub fn is_pow2(num: usize) -> bool {
    (num != 0) && ((num & (num - 1)) == 0)
}

pub fn mod_pow2(num: usize) -> usize {
    num - (1 << (log2(num)))
}

pub fn log2(num: usize) -> usize {
    if num == 0 {
        panic!("attempted to take log2 of 0!");
    }
    let mut c = num;
    let mut log2 = 0;
    while c > 0 {
        c >>= 1;
        log2 += 1;
    }
    log2 - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_pow2() {
        assert!(is_pow2(16));
        assert!(is_pow2(1024));
        assert!(!is_pow2(15));
        assert!(!is_pow2(1025));
    }

    #[test]
    fn test_mod_pow2() {
        assert_eq!(mod_pow2(16), 0);
        assert_eq!(mod_pow2(1048), 24);
        assert_eq!(mod_pow2(20), 4);
    }

    #[test]
    fn test_log2() {
        assert_eq!(log2(2), 1);
        assert_eq!(log2(16), 4);
        assert_eq!(log2(2048), 11);
    }
}
