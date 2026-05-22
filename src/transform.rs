//! Block-diagonal transform support for Radau IIA.
//!
//! The Schur decomposition of A^{-1} reveals its eigenstructure:
//! - One real eigenvalue (u1)
//! - (s-1)/2 complex conjugate pairs
//!
//! This module provides transformation matrices that permute stages into
//! block-diagonal form, allowing the Newton solver to work with separated
//! real and complex subsystems.

use nalgebra::{DMatrix, DVector, Dyn, LU};

#[derive(Debug, Clone)]
pub struct ComplexPair {
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Debug, Clone)]
pub struct BlockDiagTransform {
    pub t_mat: DMatrix<f64>,
    pub t_inv: DMatrix<f64>,
    pub lambda_real: f64,
    pub complex_pairs: Vec<ComplexPair>,
}

impl BlockDiagTransform {
    /// Build transformation from Schur decomposition of A^{-1}.
    ///
    /// Decomposes A_inv = Q T Q^T where T is quasi-triangular with:
    /// - 1×1 blocks for real eigenvalues
    /// - 2×2 blocks for complex conjugate pairs
    ///
    /// Returns a transform that:
    /// - Extracts the real eigenvalue λ₁ and complex pairs {αⱼ ± iβⱼ}
    /// - Builds transformation matrices to permute stages into block-diagonal form:
    ///   stage-stacked vector → [real block stages; complex pair stages...]
    pub fn from_a_inv(a_inv: &DMatrix<f64>) -> Self {
        let s = a_inv.nrows();
        
        // Compute Schur decomposition: A_inv = Q T Q^T
        let schur = a_inv.clone().schur();
        let (_q, t) = schur.unpack();

        // Scan quasi-triangular form to identify real eigenvalue and complex pairs
        let mut real_idx: Option<usize> = None;
        let mut complex_blocks: Vec<(usize, usize)> = Vec::new();
        
        let mut i = 0;
        while i < s {
            let is_complex = i + 1 < s && t[(i + 1, i)].abs() > 1e-12;
            if is_complex {
                complex_blocks.push((i, i + 1));
                i += 2;
            } else {
                if real_idx.is_none() {
                    real_idx = Some(i);
                }
                i += 1;
            }
        }

        assert_eq!(complex_blocks.len(), (s - 1) / 2,
                   "Expected {} complex pairs, got {}", (s - 1) / 2, complex_blocks.len());

        let real_idx = real_idx.expect("Radau IIA A^-1 must have a real eigenvalue");
        let lambda_real = t[(real_idx, real_idx)];

        // Extract complex pairs: for each 2×2 block in T, extract α ± iβ
        let mut complex_pairs = Vec::new();
        for (i1, i2) in &complex_blocks {
            let alpha = t[(*i1, *i1)];
            let beta = t[(*i2, *i1)];  // sub-diagonal element
            complex_pairs.push(ComplexPair { alpha, beta });
        }

        // Build permutation that orders: [real_stage, complex_pairs_stages...]
        let mut perm = Vec::new();
        perm.push(real_idx);
        for (i1, i2) in &complex_blocks {
            perm.push(*i1);
            perm.push(*i2);
        }

        // T_mat permutes from physical to block-diagonal coordinates
        let mut t_mat = DMatrix::zeros(s, s);
        for (bd_idx, phys_idx) in perm.iter().enumerate() {
            t_mat[(bd_idx, *phys_idx)] = 1.0;
        }

        // T_inv_mat is the inverse permutation
        let mut t_inv = DMatrix::zeros(s, s);
        for (bd_idx, phys_idx) in perm.iter().enumerate() {
            t_inv[(*phys_idx, bd_idx)] = 1.0;
        }

        Self {
            t_mat,
            t_inv,
            lambda_real,
            complex_pairs,
        }
    }

    /// Apply (T^{-1} \otimes I) to a stage-stacked vector.
    pub fn transform_residual(&self, f: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let s = self.t_inv.nrows();
        let mut tf = vec![0.0_f64; s * n];

        for i in 0..s {
            for j in 0..s {
                let tij = self.t_inv[(i, j)];
                if tij == 0.0 {
                    continue;
                }
                for k in 0..n {
                    tf[i * n + k] += tij * f[j * n + k];
                }
            }
        }

        let r_real = tf[0..n].to_vec();
        let mut r_complex = Vec::new();
        for k in 0..self.complex_pairs.len() {
            let start = (1 + 2 * k) * n;
            r_complex.push(tf[start..start + 2 * n].to_vec());
        }

        (r_real, r_complex)
    }

    /// Apply (T \otimes I) to go back to physical stage coordinates.
    pub fn back_transform(&self, z_real: &[f64], z_complex: &[Vec<f64>], n: usize) -> Vec<f64> {
        let s = self.t_mat.nrows();
        let mut z_tilde = vec![0.0_f64; s * n];
        z_tilde[0..n].copy_from_slice(z_real);

        for k in 0..self.complex_pairs.len() {
            let start = (1 + 2 * k) * n;
            z_tilde[start..start + 2 * n].copy_from_slice(&z_complex[k]);
        }

        let mut dz = vec![0.0_f64; s * n];
        for i in 0..s {
            for j in 0..s {
                let tij = self.t_mat[(i, j)];
                if tij == 0.0 {
                    continue;
                }
                for k in 0..n {
                    dz[i * n + k] += tij * z_tilde[j * n + k];
                }
            }
        }

        dz
    }
}

pub struct BlockFactors {
    pub real_lu: LU<f64, Dyn, Dyn>,
    pub complex_lus: Vec<LU<f64, Dyn, Dyn>>,
}

impl BlockFactors {
    pub fn build(transform: &BlockDiagTransform, jac: &DMatrix<f64>, h: f64) -> Self {
        let n = jac.nrows();
        let ident = DMatrix::<f64>::identity(n, n);

        let real_mat = ident.scale(transform.lambda_real) - jac.scale(h);
        let real_lu = real_mat.lu();

        let mut complex_lus = Vec::new();
        for cp in &transform.complex_pairs {
            let mut m = DMatrix::<f64>::zeros(2 * n, 2 * n);
            for r in 0..n {
                for c in 0..n {
                    let val = -h * jac[(r, c)];
                    let diag_add = if r == c { cp.alpha } else { 0.0 };
                    let beta_add = if r == c { cp.beta } else { 0.0 };

                    m[(r, c)] = val + diag_add;
                    m[(r + n, c + n)] = val + diag_add;
                    m[(r, c + n)] = -beta_add;
                    m[(r + n, c)] = beta_add;
                }
            }
            complex_lus.push(m.lu());
        }

        Self { real_lu, complex_lus }
    }

    pub fn solve(
        &self,
        r_real: &[f64],
        r_complex: &[Vec<f64>],
    ) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let rr = DVector::from_column_slice(r_real);
        let z_real = self.real_lu.solve(&rr)?.iter().copied().collect();

        let mut z_complex = Vec::new();
        for (lu, rc) in self.complex_lus.iter().zip(r_complex.iter()) {
            let rcv = DVector::from_column_slice(rc);
            let sol = lu.solve(&rcv)?;
            z_complex.push(sol.iter().copied().collect());
        }

        Some((z_real, z_complex))
    }
}