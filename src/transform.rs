//! Block-diagonal transform support for Radau IIA.
//!
//! For now this file only provides the data structures and block-factor
//! builder needed by the adaptive Newton solver. The current implementation
//! uses a conservative identity transform placeholder, which keeps the code
//! correct while allowing the solver to switch between direct Kronecker
//! solves and future block-diagonal solves.

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
    /// Placeholder transform.
    ///
    /// This keeps the API stable while we build out the true Hairer-style
    /// block-diagonal decomposition. For odd-stage Radau IIA, the eventual
    /// implementation will populate one real block and (s-1)/2 complex pairs.
    pub fn from_a_inv(a_inv: &DMatrix<f64>) -> Self {
        let s = a_inv.nrows();
        Self {
            t_mat: DMatrix::identity(s, s),
            t_inv: DMatrix::identity(s, s),
            lambda_real: 1.0,
            complex_pairs: Vec::new(),
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