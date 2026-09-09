//! Real-Schur block factorisation of the Radau Newton system.
//!
//! The Newton system for one Radau step is
//!
//! ```text
//!     (I_s ⊗ I_n − h A ⊗ J) Δz = −r
//! ```
//!
//! which costs one `(s·n)×(s·n)` LU. Left-multiplying by `(A⁻¹ ⊗ I)` gives
//!
//! ```text
//!     (A⁻¹ ⊗ I − h I_s ⊗ J) Δz = −(A⁻¹ ⊗ I) r
//! ```
//!
//! so the stage coupling now lives entirely in the small `s×s` matrix `A⁻¹`.
//! Taking its real Schur decomposition `A⁻¹ = Q T Qᵀ` (`Q` orthogonal, `T`
//! quasi-upper-triangular with 1×1 and 2×2 diagonal blocks) and substituting
//! `Δz = (Q ⊗ I) w`:
//!
//! ```text
//!     (T ⊗ I − h I_s ⊗ J) w = −(T Qᵀ ⊗ I) r,      Δz = (Q ⊗ I) w
//! ```
//!
//! `T ⊗ I − h I ⊗ J` is block upper triangular, so `w` follows by block
//! back-substitution: one `n×n` LU per real eigenvalue and one `2n×2n` LU per
//! complex pair, versus a single `(s·n)×(s·n)` LU. For `s = 7` that is roughly
//! `8.3 n³` against `114 n³` — the win grows with both `s` and `n`.
//!
//! Note that the real Schur form is only quasi-*triangular*, not block
//! diagonal: the strictly-upper coupling entries of `T` are real and must be
//! carried through the back-substitution. They cost an `O(s² n)` gemv, which is
//! negligible beside the factorisations.

use nalgebra::{DMatrix, DVector, Dyn, LU};

/// A diagonal block of the Schur form: `size` is 1 (real eigenvalue) or 2
/// (complex conjugate pair), starting at row/column `start` of `T`.
#[derive(Debug, Clone, Copy)]
pub struct SchurBlock {
    pub start: usize,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct BlockDiagTransform {
    /// Orthogonal Schur vectors, `a_inv == q * t * q.transpose()`.
    pub q: DMatrix<f64>,
    /// Quasi-upper-triangular Schur form.
    pub t: DMatrix<f64>,
    /// Precomputed `T · Qᵀ`, the operator applied to the Newton residual.
    tqt: DMatrix<f64>,
    /// Diagonal blocks of `T`, in increasing `start` order.
    pub blocks: Vec<SchurBlock>,
}

impl BlockDiagTransform {
    /// Build the transform from the real Schur decomposition of `A⁻¹`.
    pub fn from_a_inv(a_inv: &DMatrix<f64>) -> Self {
        let s = a_inv.nrows();
        let (q, t) = a_inv.clone().schur().unpack();

        // Identify the 1×1 / 2×2 diagonal blocks by scanning the sub-diagonal.
        // The scale-relative threshold keeps a genuinely tiny but nonzero
        // sub-diagonal entry from being mistaken for a block boundary.
        let scale = a_inv.amax().max(1.0);
        let tol = 1e-12 * scale;

        let mut blocks = Vec::new();
        let mut i = 0;
        while i < s {
            if i + 1 < s && t[(i + 1, i)].abs() > tol {
                blocks.push(SchurBlock { start: i, size: 2 });
                i += 2;
            } else {
                blocks.push(SchurBlock { start: i, size: 1 });
                i += 1;
            }
        }

        let tqt = &t * q.transpose();

        Self { q, t, tqt, blocks }
    }

    /// Number of stages `s`.
    pub fn stages(&self) -> usize {
        self.t.nrows()
    }

    /// Right-hand side of the transformed system: `−(T Qᵀ ⊗ I) r`.
    pub fn transform_residual(&self, r: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.stages() * n];
        kron_apply(&self.tqt, r, n, &mut out);
        for v in out.iter_mut() {
            *v = -*v;
        }
        out
    }

    /// Map the transformed solution back to stage coordinates: `(Q ⊗ I) w`.
    pub fn back_transform(&self, w: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.stages() * n];
        kron_apply(&self.q, w, n, &mut out);
        out
    }
}

/// `out += (M ⊗ I_n) v`, with `v` and `out` stage-stacked.
fn kron_apply(m: &DMatrix<f64>, v: &[f64], n: usize, out: &mut [f64]) {
    let s = m.nrows();
    for i in 0..s {
        for j in 0..s {
            let mij = m[(i, j)];
            if mij == 0.0 {
                continue;
            }
            for k in 0..n {
                out[i * n + k] += mij * v[j * n + k];
            }
        }
    }
}

/// LU factors of the diagonal blocks of `T ⊗ I − h I ⊗ J`.
pub struct BlockFactors {
    factors: Vec<LU<f64, Dyn, Dyn>>,
    n: usize,
}

impl BlockFactors {
    pub fn build(transform: &BlockDiagTransform, jac: &DMatrix<f64>, h: f64) -> Self {
        let n = jac.nrows();
        let t = &transform.t;

        let factors = transform
            .blocks
            .iter()
            .map(|blk| {
                let i = blk.start;
                let bn = blk.size * n;
                let mut m = DMatrix::<f64>::zeros(bn, bn);

                // The 2×2 diagonal blocks of a real Schur form are not
                // guaranteed to be in the standardised [[α, −β], [β, α]] shape,
                // so take the four entries of T as they come.
                for br in 0..blk.size {
                    for bc in 0..blk.size {
                        let t_entry = t[(i + br, i + bc)];
                        for r in 0..n {
                            for c in 0..n {
                                let mut val = if br == bc { -h * jac[(r, c)] } else { 0.0 };
                                if r == c {
                                    val += t_entry;
                                }
                                m[(br * n + r, bc * n + c)] = val;
                            }
                        }
                    }
                }
                m.lu()
            })
            .collect();

        Self { factors, n }
    }

    /// Solve `(T ⊗ I − h I ⊗ J) w = rhs` by block back-substitution.
    pub fn solve(&self, transform: &BlockDiagTransform, rhs: &[f64]) -> Option<Vec<f64>> {
        let n = self.n;
        let t = &transform.t;
        let mut w = vec![0.0_f64; rhs.len()];

        // T is upper quasi-triangular, so the last block is decoupled; work
        // upward, subtracting the contribution of the blocks already solved.
        for (bi, blk) in transform.blocks.iter().enumerate().rev() {
            let bn = blk.size * n;
            let mut local = vec![0.0_f64; bn];

            for br in 0..blk.size {
                let row = blk.start + br;
                for k in 0..n {
                    local[br * n + k] = rhs[row * n + k];
                }
                // Couplings to every column right of this block.
                for col in (blk.start + blk.size)..transform.stages() {
                    let t_entry = t[(row, col)];
                    if t_entry == 0.0 {
                        continue;
                    }
                    for k in 0..n {
                        local[br * n + k] -= t_entry * w[col * n + k];
                    }
                }
            }

            let sol = self.factors[bi].solve(&DVector::from_column_slice(&local))?;
            for br in 0..blk.size {
                let row = blk.start + br;
                for k in 0..n {
                    w[row * n + k] = sol[br * n + k];
                }
            }
        }

        Some(w)
    }
}
