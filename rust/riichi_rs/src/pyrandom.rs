//! Bit-exact re-implementation of CPython's `random` module (MT19937 core,
//! int seeding via init_by_array, random(), getrandbits, _randbelow, shuffle,
//! randrange, choice, uniform, gauss, choices). Needed so a Rust table built
//! from `random.seed(deal_seed)` deals the SAME wall as PyMahjongTable —
//! paired-wall evaluations and dup_k group baselines depend on it.
use pyo3::prelude::*;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; N],
    mti: usize,
    pub gauss_next: Option<f64>,
}

impl Mt19937 {
    pub fn new() -> Self {
        let mut r = Mt19937 { mt: [0u32; N], mti: N + 1, gauss_next: None };
        r.init_genrand(19650218);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let klen = key.len().max(1);
        let mut k = if N > klen { N } else { klen };
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(if key.is_empty() { 0 } else { key[j] })
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= klen {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
        self.mti = N;
    }

    /// `random.seed(int)`: CPython splits |n| into 32-bit words, little end first.
    pub fn seed_u64(&mut self, n: u64) {
        let mut key = Vec::new();
        let mut v = n;
        if v == 0 {
            key.push(0);
        }
        while v > 0 {
            key.push((v & 0xffff_ffff) as u32);
            v >>= 32;
        }
        self.init_by_array(&key);
        self.gauss_next = None;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            let mut kk = 0usize;
            while kk < N - M {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
                kk += 1;
            }
            while kk < N - 1 {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
                kk += 1;
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`: 53-bit float in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `random.getrandbits(k)` for 0 < k <= 64.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k > 0 && k <= 64);
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        // CPython fills 32-bit words little end first; the LAST word is truncated.
        let lo = self.genrand_u32() as u64;
        let hi = (self.genrand_u32() >> (64 - k)) as u64;
        lo | (hi << 32)
    }

    /// `random._randbelow_with_getrandbits(n)`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    pub fn randrange(&mut self, n: u64) -> u64 {
        self.randbelow(n)
    }

    /// `random.shuffle(x)`: Fisher-Yates from the back, j = randbelow(i+1).
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        let n = x.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.randbelow((i + 1) as u64) as usize;
            x.swap(i, j);
        }
    }

    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// `random.gauss(mu, sigma)` with CPython's cached second value.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let z = match self.gauss_next.take() {
            Some(z) => z,
            None => {
                let x2pi = self.random() * std::f64::consts::TAU;
                let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
                let z = x2pi.cos() * g2rad;
                self.gauss_next = Some(x2pi.sin() * g2rad);
                z
            }
        };
        mu + z * sigma
    }

    /// `random.choices(population, weights)[0]`: bisect_right on cumulative weights.
    pub fn choices_index(&mut self, weights: &[f64]) -> usize {
        let mut cum = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for w in weights {
            acc += w;
            cum.push(acc);
        }
        let total = acc;
        let x = self.random() * total;
        let hi = weights.len() - 1;
        // bisect.bisect_right(cum, x, 0, hi)
        let (mut lo, mut hi) = (0usize, hi);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x < cum[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }
}

impl Default for Mt19937 {
    fn default() -> Self {
        Self::new()
    }
}

/// Python-facing handle used by the parity tests.
#[pyclass(name = "PyRandom")]
pub struct PyRandom {
    inner: Mt19937,
}

#[pymethods]
impl PyRandom {
    #[new]
    fn new(seed: u64) -> Self {
        let mut inner = Mt19937::new();
        inner.seed_u64(seed);
        PyRandom { inner }
    }
    fn seed(&mut self, seed: u64) {
        self.inner.seed_u64(seed);
    }
    fn random(&mut self) -> f64 {
        self.inner.random()
    }
    fn getrandbits(&mut self, k: u32) -> u64 {
        self.inner.getrandbits(k)
    }
    fn randrange(&mut self, n: u64) -> u64 {
        self.inner.randrange(n)
    }
    fn shuffle(&mut self, x: Vec<i64>) -> Vec<i64> {
        let mut v = x;
        self.inner.shuffle(&mut v);
        v
    }
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        self.inner.uniform(a, b)
    }
    fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        self.inner.gauss(mu, sigma)
    }
    fn choices_index(&mut self, weights: Vec<f64>) -> usize {
        self.inner.choices_index(&weights)
    }
}
