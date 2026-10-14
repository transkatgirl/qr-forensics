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

const fn build_products() -> [[u8; 256]; 256] {
    let (exp, log) = build_tables();
    let mut out = [[0u8; 256]; 256];
    let mut a = 1;
    while a < 256 {
        let mut b = 1;
        while b < 256 {
            out[a][b] = exp[log[a] as usize + log[b] as usize];
            b += 1;
        }
        a += 1;
    }
    out
}

/// Every product, `PRODUCTS[a][b] = a·b`: a multiplication is one lookup,
/// with no logarithm taken and no test for nought, and a loop that
/// multiplies many elements by one reads one row of it.
static PRODUCTS: [[u8; 256]; 256] = build_products();

/// α^i, for any non-negative exponent.
#[inline]
pub fn exp(i: usize) -> u8 {
    TABLES.0[i % 255]
}

/// α^i for an exponent below 510: the sum of two logarithms, with no
/// reduction.
#[inline]
pub fn exp_of_sum(i: usize) -> u8 {
    debug_assert!(i < 510);
    TABLES.0[i]
}

/// Discrete logarithm base α. `x` must be non-zero.
#[inline]
pub fn log(x: u8) -> usize {
    debug_assert!(x != 0, "log of zero");
    TABLES.1[x as usize] as usize
}

#[inline]
pub fn mul(a: u8, b: u8) -> u8 {
    PRODUCTS[a as usize][b as usize]
}

/// The products of `a` with every element: `products(a)[b]` is `a·b`.
#[inline]
pub fn products(a: u8) -> &'static [u8; 256] {
    &PRODUCTS[a as usize]
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

/// Multiplication by one element, eight elements a word at a time.
/// Multiplying by `c` is linear over the bits, `c · t` the sum of
/// `c · 2^b` over the bits `b` of `t`: each bit of every byte of a word
/// is widened to a byte of ones, and `c · 2^b` kept where it is set.
pub struct Scale {
    by_bit: [u64; 8],
    times_c: &'static [u8; 256],
}

impl Scale {
    const ONES: u64 = 0x0101_0101_0101_0101;

    pub fn new(c: u8) -> Self {
        Self { by_bit: std::array::from_fn(|b| u64::from(mul(c, 1 << b)) * Self::ONES), times_c: products(c) }
    }

    /// The eight elements of `w`, one a byte, each times `c`.
    #[inline]
    pub fn word(&self, w: u64) -> u64 {
        let mut out = 0;
        for (b, &times) in self.by_bit.iter().enumerate() {
            let set = w >> b & Self::ONES;
            out ^= (set << 8).wrapping_sub(set) & times;
        }
        out
    }

    /// One element times `c`.
    #[inline]
    pub fn byte(&self, t: u8) -> u8 {
        self.times_c[usize::from(t)]
    }

    /// `acc[i] += c · src[i]` for every `i`, the shorter of the two long.
    pub fn add_into(&self, acc: &mut [u8], src: &[u8]) {
        let len = acc.len().min(src.len());
        let (acc_words, acc_rest) = acc[..len].as_chunks_mut::<8>();
        let (src_words, src_rest) = src[..len].as_chunks::<8>();
        for (a, &s) in acc_words.iter_mut().zip(src_words) {
            *a = (u64::from_le_bytes(*a) ^ self.word(u64::from_le_bytes(s))).to_le_bytes();
        }
        for (a, &s) in acc_rest.iter_mut().zip(src_rest) {
            *a ^= self.byte(s);
        }
    }
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
    fn scaling_a_word_multiplies_each_byte() {
        let mut word = 0x0123_4567_89AB_CDEFu64;
        for c in 0..=255u8 {
            let scale = Scale::new(c);
            for _ in 0..64 {
                word = word.rotate_left(13).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(c);
                let bytes = word.to_le_bytes();
                let want: Vec<u8> = bytes.iter().map(|&t| mul(c, t)).collect();
                assert_eq!(scale.word(word).to_le_bytes().to_vec(), want, "{c} times {word:#x}");
            }
            // Every length, across a word's end and short of one.
            for len in 0..20usize {
                let src: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37) ^ c).collect();
                let mut acc: Vec<u8> = (0..len + 3).map(|i| (i as u8).wrapping_mul(91)).collect();
                let want: Vec<u8> = acc.iter().enumerate().map(|(i, &a)| if i < len { a ^ mul(c, src[i]) } else { a }).collect();
                scale.add_into(&mut acc, &src);
                assert_eq!(acc, want, "{c}, {len} long");
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
