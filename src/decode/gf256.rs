//! GF(2^8) arithmetic over the QR primitive polynomial x^8 + x^4 + x^3 + x^2 + 1.
//!
//! Polynomials are stored little-endian: `p[i]` is the coefficient of x^i.

const PRIMITIVE: u16 = 0x11D;

const fn build_tables() -> ([u8; 510], [u8; 256]) {
    let mut exp = [0u8; 510];
    let mut log = [0u8; 256];
    let mut x: u16 = 1;
    let mut i = 0;
    while i < 255 {
        exp[i] = x as u8;
        exp[i + 255] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= PRIMITIVE;
        }
        i += 1;
    }
    (exp, log)
}

static TABLES: ([u8; 510], [u8; 256]) = build_tables();

/// α^i, for any non-negative exponent.
#[inline]
pub fn exp(i: usize) -> u8 {
    TABLES.0[i % 255]
}

/// Discrete logarithm base α. `x` must be non-zero.
#[inline]
pub fn log(x: u8) -> usize {
    debug_assert!(x != 0, "log of zero");
    TABLES.1[x as usize] as usize
}

#[inline]
pub fn mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { 0 } else { TABLES.0[log(a) + log(b)] }
}

/// `a / b`. `b` must be non-zero.
#[inline]
pub fn div(a: u8, b: u8) -> u8 {
    debug_assert!(b != 0, "division by zero");
    if a == 0 { 0 } else { TABLES.0[log(a) + 255 - log(b)] }
}

#[inline]
pub fn inv(a: u8) -> u8 {
    div(1, a)
}

pub fn pow(a: u8, n: usize) -> u8 {
    if n == 0 {
        1
    } else if a == 0 {
        0
    } else {
        exp(log(a) * (n % 255))
    }
}

pub fn poly_eval(p: &[u8], x: u8) -> u8 {
    p.iter().rev().fold(0, |acc, &c| mul(acc, x) ^ c)
}

pub fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u8; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            out[i + j] ^= mul(x, y);
        }
    }
    out
}

/// Degree of `p`, treating the zero polynomial as degree 0.
pub fn poly_degree(p: &[u8]) -> usize {
    p.iter().rposition(|&c| c != 0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_axioms_hold_exhaustively() {
        for a in 1..=255u8 {
            assert_eq!(mul(a, inv(a)), 1);
            assert_eq!(exp(log(a)), a);
            for b in 1..=255u8 {
                let p = mul(a, b);
                assert_eq!(div(p, b), a);
                assert_eq!(mul(b, a), p);
            }
        }
        assert_eq!(mul(0, 7), 0);
        assert_eq!(div(0, 7), 0);
    }

    #[test]
    fn multiplication_matches_carryless_reduction() {
        fn slow(mut a: u16, mut b: u16) -> u8 {
            let mut r = 0u16;
            while b != 0 {
                if b & 1 != 0 {
                    r ^= a;
                }
                a <<= 1;
                if a & 0x100 != 0 {
                    a ^= PRIMITIVE;
                }
                b >>= 1;
            }
            r as u8
        }
        for a in 0..=255u16 {
            for b in 0..=255u16 {
                assert_eq!(mul(a as u8, b as u8), slow(a, b));
            }
        }
    }

    #[test]
    fn pow_and_eval() {
        assert_eq!(pow(2, 8), 0x1D);
        assert_eq!(pow(0, 0), 1);
        assert_eq!(pow(0, 3), 0);
        assert_eq!(pow(exp(200), 3), exp(600));
        // (1 + x)(1 + 2x) = 1 + 3x + 2x^2
        let p = poly_mul(&[1, 1], &[1, 2]);
        assert_eq!(p, vec![1, 3, 2]);
        assert_eq!(poly_eval(&p, 1), 0);
        assert_eq!(poly_eval(&p, inv(2)), 0);
        assert_eq!(poly_degree(&[1, 0, 0]), 0);
        assert_eq!(poly_degree(&[1, 3, 2, 0]), 2);
    }
}
