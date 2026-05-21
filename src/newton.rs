use nalgebra::{DMatrix, DVector};

use crate::error::{RadauError, Result};
use crate::problem::OdeProblem;
use crate::tableau::RadauTableau;
use crate::transform::BlockFactors;

#[derive(Debug, Clone, Copy)]
pub enum NewtonLinearAlgebra {
    KroneckerFull,
    BlockDiagonal,
}

#[derive(Debug, Clone)]
pub struct CachedJacobian {
    pub jac: DMatrix<f64>,
    pub age: usize,
    pub max_age: usize,
}

impl CachedJacobian {
    pub fn new(n: usize) -> Self {
        Self { jac: DMatrix::zeros(n, n), age: usize::MAX, max_age: 20 }
    }

    pub fn is_stale(&self) -> bool { self.age >= self.max_age }
    pub fn increment_age(&mut self) { self.age += 1; }
    pub fn mark_stale(&mut self) { self.age = self.max_age; }

    pub fn refresh(&mut self, problem: &dyn OdeProblem, t: f64, y: &[f64]) {
        let n = y.len();
        let mut f0 = vec![0.0; n];
        problem.rhs(t, y, &mut f0);
        for col in 0..n {
            let h = f64::EPSILON.sqrt() * y[col].abs().max(1.0);
            let mut yp = y.to_vec();
            yp[col] += h;
            let mut fp = vec![0.0; n];
            problem.rhs(t, &yp, &mut fp);
            for row in 0..n {
                self.jac[(row, col)] = (fp[row] - f0[row]) / h;
            }
        }
        self.age = 0;
    }
}

/// Result of the Newton solve for one Radau step.
///
/// Stages are stored as **increments** `Z_i = Y_i - y_n` rather than absolute
/// `Y_i`. This is essential for numerical accuracy at small step sizes:
/// computing `Y_i - y_n` by subtraction loses precision when `|Y_i| ≈ |y_n|`
/// and the difference is much smaller than either operand. The downstream
/// ESTRAD error estimator divides these increments by `h`, so any loss of
/// significance here would be amplified into the error norm.
#[derive(Debug, Clone)]
pub struct NewtonState {
    /// Stage increments `Z_i = Y_i - y_n` (shape: stages × n).
    pub increments: Vec<Vec<f64>>,
    /// Stage absolute values `Y_i = y_n + Z_i` (recomputed at convergence so
    /// integrator code can read them without recomputing).
    pub stages: Vec<Vec<f64>>,
    pub iter_count: usize,
    pub converged: bool,
}

impl NewtonState {
    pub fn new(stage_count: usize, n: usize) -> Self {
        Self {
            increments: vec![vec![0.0; n]; stage_count],
            stages: vec![vec![0.0; n]; stage_count],
            iter_count: 0,
            converged: false,
        }
    }
}

pub fn solve_step(
    problem: &dyn OdeProblem,
    tableau: &RadauTableau,
    t: f64,
    h: f64,
    y: &[f64],
    jac_cache: &mut CachedJacobian,
    tol: f64,
    max_iter: usize,
    linear_algebra: NewtonLinearAlgebra,
) -> Result<NewtonState> {
    if jac_cache.is_stale() {
        jac_cache.refresh(problem, t, y);
    }

    match linear_algebra {
        NewtonLinearAlgebra::KroneckerFull => {
            solve_kronecker(problem, tableau, t, h, y, jac_cache, tol, max_iter)
        }
        NewtonLinearAlgebra::BlockDiagonal => {
            solve_block_diag(problem, tableau, t, h, y, jac_cache, tol, max_iter)
        }
    }
}

fn solve_kronecker(
    problem: &dyn OdeProblem,
    tableau: &RadauTableau,
    t: f64,
    h: f64,
    y: &[f64],
    jac_cache: &mut CachedJacobian,
    tol: f64,
    max_iter: usize,
) -> Result<NewtonState> {
    let n = y.len();
    let s = tableau.stages;
    let sn = s * n;

    let mut m = DMatrix::<f64>::identity(sn, sn);
    for i in 0..s {
        for q in 0..s {
            let a_iq = tableau.a[(i, q)];
            if a_iq == 0.0 {
                continue;
            }
            let scale = h * a_iq;
            for row in 0..n {
                for col in 0..n {
                    m[(i * n + row, q * n + col)] -= scale * jac_cache.jac[(row, col)];
                }
            }
        }
    }
    let lu = m.lu();

    newton_loop(problem, tableau, t, h, y, jac_cache, tol, max_iter, |resid| {
        let neg = DVector::from_column_slice(resid).map(|x| -x);
        lu.solve(&neg).map(|v| v.iter().copied().collect())
    })
}

fn solve_block_diag(
    problem: &dyn OdeProblem,
    tableau: &RadauTableau,
    t: f64,
    h: f64,
    y: &[f64],
    jac_cache: &mut CachedJacobian,
    tol: f64,
    max_iter: usize,
) -> Result<NewtonState> {
    let n = y.len();
    let factors = BlockFactors::build(&tableau.transform, &jac_cache.jac, h);

    newton_loop(problem, tableau, t, h, y, jac_cache, tol, max_iter, |resid| {
        let (r_real, r_complex) = tableau.transform.transform_residual(resid, n);
        let (z_real, z_complex) = factors.solve(&r_real, &r_complex)?;
        Some(tableau.transform.back_transform(&z_real, &z_complex, n))
    })
}

fn newton_loop(
    problem: &dyn OdeProblem,
    tableau: &RadauTableau,
    t: f64,
    h: f64,
    y: &[f64],
    jac_cache: &mut CachedJacobian,
    tol: f64,
    max_iter: usize,
    linear_solve: impl Fn(&[f64]) -> Option<Vec<f64>>,
) -> Result<NewtonState> {
    let n = y.len();
    let s = tableau.stages;
    let sn = s * n;

    let mut state = NewtonState::new(s, n);
    let mut fvals = vec![vec![0.0; n]; s];
    // Y_i = y + Z_i; updated whenever Z is corrected.
    let mut y_stage = vec![0.0; n];

    for iter in 0..max_iter {
        for i in 0..s {
            let ti = t + tableau.c[i] * h;
            for j in 0..n {
                y_stage[j] = y[j] + state.increments[i][j];
            }
            problem.rhs(ti, &y_stage, &mut fvals[i]);
        }

        // Collocation residual in increment form:
        //   r_i = Z_i - h * sum_q A_{iq} f(t + c_q h, y + Z_q)
        // (The y-term cancels because each stage subtracts y from Y_i = y + Z_i.)
        let mut resid = vec![0.0_f64; sn];
        for i in 0..s {
            for j in 0..n {
                let mut acc = state.increments[i][j];
                for q in 0..s {
                    acc -= h * tableau.a[(i, q)] * fvals[q][j];
                }
                resid[i * n + j] = acc;
            }
        }

        let dz = linear_solve(&resid).ok_or(RadauError::LinearSolveFailed)?;

        let mut max_corr = 0.0_f64;
        for i in 0..s {
            for j in 0..n {
                let corr = dz[i * n + j];
                state.increments[i][j] += corr;
                // Scale by stage size to weight in the convergence test.
                let sc = 1.0_f64 + state.increments[i][j].abs();
                max_corr = max_corr.max((corr / sc).abs());
            }
        }

        state.iter_count = iter + 1;

        if max_corr < tol {
            state.converged = true;
            for i in 0..s {
                for j in 0..n {
                    state.stages[i][j] = y[j] + state.increments[i][j];
                }
            }
            jac_cache.increment_age();
            return Ok(state);
        }

        if iter >= 4 && max_corr > 0.5 {
            jac_cache.mark_stale();
        }
    }

    jac_cache.mark_stale();
    Err(RadauError::NewtonFailed)
}