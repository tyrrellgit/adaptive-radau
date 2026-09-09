//! The block factorisation must reproduce the direct Kronecker Newton solve
//! exactly — it is only a cheaper factorisation of the same operator.

use nalgebra::{DMatrix, DVector};

use crate::tableau::RadauTableau;
use crate::transform::BlockFactors;

/// Deterministic pseudo-random fill, so failures are reproducible.
fn pseudo_random(n: usize, seed: u64) -> DMatrix<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    DMatrix::from_fn(n, n, |_, _| next())
}

/// `a_inv == Q T Qᵀ` — pins nalgebra's `Schur::unpack` convention.
#[test]
fn schur_factors_reconstruct_a_inv() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        let tr = &t.transform;
        let recon = &tr.q * &tr.t * tr.q.transpose();
        for i in 0..t.stages {
            for j in 0..t.stages {
                let diff = (recon[(i, j)] - t.a_inv[(i, j)]).abs();
                assert!(diff < 1e-12, "order {} ({},{}) diff {}", order, i, j, diff);
            }
        }
    }
}

/// Q must be orthogonal.
#[test]
fn schur_vectors_are_orthogonal() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        let qtq = t.transform.q.transpose() * &t.transform.q;
        for i in 0..t.stages {
            for j in 0..t.stages {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[(i, j)] - expect).abs() < 1e-12, "order {} not orthogonal", order);
            }
        }
    }
}

/// Block sizes must sum to s, with exactly (s-1)/2 complex pairs.
#[test]
fn block_layout_matches_radau_spectrum() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        let blocks = &t.transform.blocks;
        let total: usize = blocks.iter().map(|b| b.size).sum();
        assert_eq!(total, t.stages, "order {}", order);
        let pairs = blocks.iter().filter(|b| b.size == 2).count();
        assert_eq!(pairs, (t.stages - 1) / 2, "order {}", order);
        assert_eq!(blocks.iter().filter(|b| b.size == 1).count(), 1, "order {}", order);
    }
}

/// The heart of it: for random J and random residual r, the block solve must
/// return the same Δz as factorising `I − h(A ⊗ J)` directly.
#[test]
fn block_solve_matches_kronecker_solve() {
    for order in [5, 9, 13] {
        let tab = RadauTableau::for_order(order).unwrap();
        let s = tab.stages;

        for (n, seed) in [(1usize, 1u64), (2, 2), (3, 3), (6, 4)] {
            for &h in &[1e-4_f64, 1e-2, 0.5] {
                let jac = pseudo_random(n, seed);
                let sn = s * n;

                // Direct: (I − h A⊗J) dz = −r
                let mut m = DMatrix::<f64>::identity(sn, sn);
                for i in 0..s {
                    for q in 0..s {
                        let scale = h * tab.a[(i, q)];
                        for r in 0..n {
                            for c in 0..n {
                                m[(i * n + r, q * n + c)] -= scale * jac[(r, c)];
                            }
                        }
                    }
                }
                let resid: Vec<f64> =
                    (0..sn).map(|k| ((k as f64) * 0.37 + 0.11).sin()).collect();
                let neg = DVector::from_column_slice(&resid).map(|x| -x);
                let expected = m.lu().solve(&neg).expect("kronecker solve");

                // Block-factorised path.
                let factors = BlockFactors::build(&tab.transform, &jac, h);
                let rhs = tab.transform.transform_residual(&resid, n);
                let w = factors.solve(&tab.transform, &rhs).expect("block solve");
                let got = tab.transform.back_transform(&w, n);

                let scale = expected.amax().max(1.0);
                for k in 0..sn {
                    let diff = (got[k] - expected[k]).abs();
                    assert!(
                        diff < 1e-9 * scale,
                        "order {} n {} h {}: index {} block {} vs kronecker {} (diff {})",
                        order, n, h, k, got[k], expected[k], diff
                    );
                }
            }
        }
    }
}
