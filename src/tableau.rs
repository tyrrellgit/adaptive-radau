//! Radau IIA tableau builder.
//!
//! All coefficients are computed numerically from the canonical definition
//! of Radau IIA collocation rather than typed by hand. This avoids the
//! truncated / mis-transcribed entries that previously plagued the s=5 and
//! s=7 tableaux, and lets us extend trivially to higher odd orders.
//!
//! Construction:
//!   1. Nodes `c_i` are the roots of
//!         d^{s-1}/dx^{s-1} [ x^{s-1} (x-1)^s ]
//!      (the canonical right-Radau polynomial, see Hairer & Wanner II.5.21).
//!   2. The collocation A matrix is the unique s×s matrix satisfying
//!         sum_j A_{ij} c_j^{k-1} = c_i^k / k    for k = 1..=s
//!      — i.e. each row is the Birkhoff/quadrature weights that integrate
//!      degree (s-1) polynomials exactly over [0, c_i].
//!   3. b_i := A_{s i} because c_s = 1.
//!   4. b_hat is constructed from Hairer's ESTRAD error estimator:
//!         b_hat = b - (last row of A^{-1}) / u1
//!      where u1 is the unique real eigenvalue of A^{-1}. The estimate
//!      `y_next - y_emb = h * sum_i (a_inv_last_row[i] / u1) f_i` is the
//!      raw, unsmoothed Hairer error proxy. (Stiff problems additionally
//!      smooth it with `(I − h γ J)^{-1}`; this is wired into the integrator.)

use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::error::{RadauError, Result};
use crate::transform::BlockDiagTransform;

#[derive(Debug, Clone)]
pub struct RadauTableau {
    pub order: usize,
    pub stages: usize,
    pub c: Vec<f64>,
    pub a: DMatrix<f64>,
    pub a_inv: DMatrix<f64>,
    pub b: Vec<f64>,
    /// Hairer's ESTRAD coefficients (`DD_i` in radau5.f).
    /// Uniquely determined by the moment conditions
    ///   sum_i DD_i * c_i^k = 0   for k = 2..s-1
    ///   sum_i DD_i * c_i     = -1
    ///   DD_s = -1/s            (normalisation fixing the remaining degree of freedom)
    /// The smoothed-form error estimate uses
    ///   err_raw[j] = sum_i (DD_i / h) * (Y_i - y_n)[j]
    ///   err[j]     = (I - h γ J)^{-1} * (err_raw + f(y_n))[j]
    /// where γ = 1 / u1 and u1 is the real eigenvalue of A^{-1}.
    pub dd: Vec<f64>,
    /// Real eigenvalue of A^{-1}. Used as γ = 1/u1 in the stiff-error smoothing step.
    pub u1: f64,
    /// Order of the embedded method when used with smoothing: `s`.
    pub embedded_order: usize,
    pub transform: BlockDiagTransform,
}

#[derive(Debug, Default)]
pub struct TableauCache {
    cache: HashMap<usize, RadauTableau>,
}

impl TableauCache {
    pub fn get(&mut self, order: usize) -> Result<&RadauTableau> {
        if !self.cache.contains_key(&order) {
            let t = RadauTableau::for_order(order)?;
            self.cache.insert(order, t);
        }
        Ok(self.cache.get(&order).expect("cache entry exists"))
    }
}

impl RadauTableau {
    /// Build the Radau IIA tableau of classical order `2s − 1`.
    /// Supported orders: any odd integer ≥ 3 (i.e. s ≥ 2). Higher orders
    /// remain numerically clean up to s ≈ 9 with standard f64 root-finding.
    pub fn for_order(order: usize) -> Result<Self> {
        if order < 3 || order % 2 == 0 {
            return Err(RadauError::InvalidOrder(order));
        }
        let s = (order + 1) / 2;
        Ok(build_tableau(s))
    }
}

fn build_tableau(s: usize) -> RadauTableau {
    let c = radau_iia_nodes(s);
    let a = build_a_matrix(&c);
    let a_inv = a.clone().try_inverse().expect("Radau A must be invertible");
    let b: Vec<f64> = a.row(s - 1).iter().copied().collect();
    let u1 = real_eigenvalue_of_a_inv(&a_inv);
    let dd = hairer_dd_coefficients(&c);
    let transform = BlockDiagTransform::from_a_inv(&a_inv);

    RadauTableau {
        order: 2 * s - 1,
        stages: s,
        c,
        a,
        a_inv,
        b,
        dd,
        u1,
        embedded_order: s,
        transform,
    }
}

/// Hairer ESTRAD coefficients.
///
/// We solve the linear system
///   DD_s                              = -1/s          (normalisation)
///   sum_j DD_j * c_j                   = -1
///   sum_j DD_j * c_j^k                 =  0   for k = 2 .. s-1
///
/// For the 3-stage tableau this recovers Hairer's published values
///   DD_1 = -(13 + 7√6)/3, DD_2 = (-13 + 7√6)/3, DD_3 = -1/3.
fn hairer_dd_coefficients(c: &[f64]) -> Vec<f64> {
    let s = c.len();
    let mut m = DMatrix::<f64>::zeros(s, s);
    let mut rhs = nalgebra::DVector::<f64>::zeros(s);

    // Row 0: DD_s = -1/s
    m[(0, s - 1)] = 1.0;
    rhs[0] = -1.0 / s as f64;

    // Row 1: sum DD_j * c_j = -1
    for j in 0..s {
        m[(1, j)] = c[j];
    }
    rhs[1] = -1.0;

    // Rows 2..s-1: sum DD_j * c_j^k = 0   for k = 2..s-1
    for row in 2..s {
        let k = row as i32;
        for j in 0..s {
            m[(row, j)] = c[j].powi(k);
        }
    }

    let lu = m.lu();
    let dd = lu.solve(&rhs).expect("DD linear system must be solvable");
    (0..s).map(|i| dd[i]).collect()
}

// ---------------------------------------------------------------------------
// Radau IIA nodes: roots of d^{s-1}/dx^{s-1} [ x^{s-1} (x-1)^s ].
// ---------------------------------------------------------------------------

fn radau_iia_nodes(s: usize) -> Vec<f64> {
    // Build coefficients of p(x) = x^{s-1} (x-1)^s in a monomial basis.
    let deg_p = 2 * s - 1;
    let mut p = vec![0.0_f64; deg_p + 1];
    // (x - 1)^s expansion shifted by x^{s-1}
    for k in 0..=s {
        let coeff = (if k % 2 == 0 { 1.0 } else { -1.0 }) * binomial(s, k) as f64;
        // coefficient of x^{(s) - k} in (x - 1)^s = (-1)^k C(s, k) x^{s-k}.
        // After multiplying by x^{s-1}: degree (s - 1) + (s - k) = 2s - 1 - k.
        p[2 * s - 1 - k] += coeff;
    }

    // Differentiate (s - 1) times.
    let mut q = p;
    for _ in 0..s - 1 {
        q = differentiate(&q);
    }
    // q is now the degree-s Radau polynomial.

    let roots = roots_real_in_unit_interval(&q, s);
    debug_assert_eq!(roots.len(), s);
    roots
}

fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num: u128 = 1;
    let mut den: u128 = 1;
    for i in 0..k {
        num *= (n - i) as u128;
        den *= (i + 1) as u128;
    }
    num / den
}

fn differentiate(p: &[f64]) -> Vec<f64> {
    if p.len() <= 1 {
        return vec![0.0];
    }
    p.iter()
        .enumerate()
        .skip(1)
        .map(|(i, &c)| c * i as f64)
        .collect()
}

fn poly_eval(p: &[f64], x: f64) -> f64 {
    // Horner
    let mut acc = 0.0;
    for &c in p.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Find all `s` real roots of `p` in [0, 1], to machine precision.
/// We assume the roots are simple and well-separated, which is true for
/// Radau IIA nodes. Strategy:
///   1. Bracket via sign changes on a fine grid.
///   2. Refine each bracket with bisection.
///   3. Polish with Newton.
fn roots_real_in_unit_interval(p: &[f64], expected: usize) -> Vec<f64> {
    let dp = differentiate(p);
    // Dense scan to find sign changes. Radau nodes for s up to ~20 are
    // well-separated; 4000 samples easily resolves them.
    const SAMPLES: usize = 4000;
    let mut prev_x = 0.0;
    let mut prev_y = poly_eval(p, prev_x);
    let mut brackets: Vec<(f64, f64)> = Vec::new();
    for k in 1..=SAMPLES {
        let x = k as f64 / SAMPLES as f64;
        let y = poly_eval(p, x);
        if prev_y == 0.0 {
            brackets.push((prev_x, prev_x));
        } else if prev_y.signum() != y.signum() {
            brackets.push((prev_x, x));
        }
        prev_x = x;
        prev_y = y;
    }
    // Final endpoint x=1 is a root for Radau IIA — include if not already.
    if poly_eval(p, 1.0).abs() < 1e-10 && brackets.last().map_or(true, |&(_, hi)| (hi - 1.0).abs() > 1e-12) {
        brackets.push((1.0, 1.0));
    }
    assert_eq!(
        brackets.len(),
        expected,
        "expected {} root brackets, found {}",
        expected,
        brackets.len()
    );

    let mut roots = Vec::with_capacity(expected);
    for (lo, hi) in brackets {
        let mut a = lo;
        let mut b = hi;
        let mut fa = poly_eval(p, a);

        // Bisection until interval is small.
        if a != b {
            for _ in 0..200 {
                if (b - a) < 1e-15 {
                    break;
                }
                let m = 0.5 * (a + b);
                let fm = poly_eval(p, m);
                if fm == 0.0 {
                    a = m;
                    b = m;
                    break;
                }
                if fa.signum() * fm.signum() < 0.0 {
                    b = m;
                } else {
                    a = m;
                    fa = fm;
                }
            }
        }
        let mut x = 0.5 * (a + b);

        // Newton polish.
        for _ in 0..30 {
            let f = poly_eval(p, x);
            let df = poly_eval(&dp, x);
            if df == 0.0 {
                break;
            }
            let dx = f / df;
            x -= dx;
            if dx.abs() < f64::EPSILON * x.abs().max(1.0) {
                break;
            }
        }
        roots.push(x.clamp(0.0, 1.0));
    }
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    roots
}

// ---------------------------------------------------------------------------
// Collocation A matrix.
//
// For each row i, solve V^T a_i = rhs_i where V_{kj} = c_j^{k-1},
// rhs_k = c_i^k / k. Vandermonde systems with distinct nodes are non-singular.
// We solve via LU through nalgebra.
// ---------------------------------------------------------------------------

fn build_a_matrix(c: &[f64]) -> DMatrix<f64> {
    let s = c.len();
    // V_{k-1, j} = c_j^{k-1}  →  V is row-power, columns indexed by stage.
    let mut v = DMatrix::<f64>::zeros(s, s);
    for k in 1..=s {
        for j in 0..s {
            v[(k - 1, j)] = c[j].powi((k - 1) as i32);
        }
    }
    let lu = v.lu();

    let mut a = DMatrix::<f64>::zeros(s, s);
    for i in 0..s {
        let rhs = nalgebra::DVector::from_iterator(
            s,
            (1..=s).map(|k| c[i].powi(k as i32) / k as f64),
        );
        let row = lu.solve(&rhs).expect("Vandermonde solve must succeed");
        for j in 0..s {
            a[(i, j)] = row[j];
        }
    }
    a
}

// ---------------------------------------------------------------------------
// Real eigenvalue of A^{-1}.
//
// For Radau IIA, A^{-1} has exactly one real eigenvalue (u1) and
// (s−1)/2 complex conjugate pairs. We extract them via the Schur form.
// ---------------------------------------------------------------------------

fn real_eigenvalue_of_a_inv(a_inv: &DMatrix<f64>) -> f64 {
    let schur = a_inv.clone().schur();
    let (_q, t) = schur.unpack();
    let n = t.nrows();

    let mut i = 0;
    let mut real_eigs = Vec::new();
    while i < n {
        let is_complex_block = i + 1 < n && t[(i + 1, i)].abs() > 1e-12;
        if is_complex_block {
            i += 2;
        } else {
            real_eigs.push(t[(i, i)]);
            i += 1;
        }
    }
    assert_eq!(
        real_eigs.len(),
        1,
        "Radau IIA A^-1 should have exactly one real eigenvalue, got {}",
        real_eigs.len()
    );
    real_eigs[0]
}
