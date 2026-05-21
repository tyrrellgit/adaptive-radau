//! Newton solver — automatically selects linear algebra strategy:
//!
//!   n <= KRONECKER_THRESHOLD:  one sn×sn LU  (fast for small systems)
//!   n >  KRONECKER_THRESHOLD:  block-diagonal via Schur transform  (fast for large systems)

use nalgebra::{DMatrix, DVector};

use crate::error::{RadauError, Result};
use crate::problem::OdeProblem;
use crate::tableau::RadauTableau;
use crate::transform::BlockFactors;

/// Below this ODE dimension, the direct Kronecker sn×sn LU is faster.
/// Above it, the block-diagonal transform is used.
const KRONECKER_THRESHOLD: usize = 50;

// ── Cached Jacobian ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CachedJacobian {
    pub jac:     DMatrix<f64>,
    pub age:     usize,
    pub max_age: usize,
}

impl CachedJacobian {
    pub fn new(n: usize) -> Self {
        Self { jac: DMatrix::zeros(n, n), age: usize::MAX, max_age: 20 }
    }
    pub fn is_stale(&self)      -> bool { self.age >= self.max_age }
    pub fn increment_age(&mut self)     { self.age += 1; }
    pub fn mark_stale(&mut self)        { self.age = self.max_age; }

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

// ── Newton state ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NewtonState {
    pub stages:     Vec<Vec<f64>>,
    pub iter_count: usize,
    pub converged:  bool,
}

impl NewtonState {
    pub fn new(stage_count: usize, y0: &[f64]) -> Self {
        Self { stages: vec![y0.to_vec(); stage_count], iter_count: 0, converged: false }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn solve_step(
    problem:   &dyn OdeProblem,
    tableau:   &RadauTableau,
    t:         f64,
    h:         f64,
    y:         &[f64],
    jac_cache: &mut CachedJacobian,
    tol:       f64,
    max_iter:  usize,
) -> Result<NewtonState> {
    if jac_cache.is_stale() {
        jac_cache.refresh(problem, t, y);
    }

    if y.len() <= KRONECKER_THRESHOLD {
        solve_kronecker(problem, tableau, t, h, y, jac_cache, tol, max_iter)
    } else {
        solve_block_diag(problem, tableau, t, h, y, jac_cache, tol, max_iter)
    }
}

// ── Strategy A: direct sn×sn Kronecker LU ────────────────────────────────────
//
// Best for small n (Robertson: n=3, sn=9).
// One LU factorisation per step, one triangular solve per Newton iteration.

fn solve_kronecker(
    problem:   &dyn OdeProblem,
    tableau:   &RadauTableau,
    t:         f64,
    h:         f64,
    y:         &[f64],
    jac_cache: &mut CachedJacobian,
    tol:       f64,
    max_iter:  usize,
) -> Result<NewtonState> {
    let n  = y.len();
    let s  = tableau.stages;
    let sn = s * n;

    // Build M = I_{sn} - h*(A ⊗ J)  and factor once
    let mut m = DMatrix::<f64>::identity(sn, sn);
    for i in 0..s {
        for q in 0..s {
            let a_iq = tableau.a[(i, q)];
            if a_iq == 0.0 { continue; }
            let scale = h * a_iq;
            for row in 0..n {
                for col in 0..n {
                    m[(i * n + row, q * n + col)] -= scale * jac_cache.jac[(row, col)];
                }
            }
        }
    }
    let lu = m.lu();

    newton_loop(problem, tableau, t, h, y, jac_cache, tol, max_iter,
        |resid| {
            let neg = DVector::from_column_slice(resid).map(|x| -x);
            lu.solve(&neg).map(|v| v.iter().copied().collect())
        })
}

// ── Strategy B: block-diagonal Schur transform ───────────────────────────────
//
// Best for large n (n > 50).
// Cost O(s·n³) vs O(s³·n³) for the Kronecker approach.

fn solve_block_diag(
    problem:   &dyn OdeProblem,
    tableau:   &RadauTableau,
    t:         f64,
    h:         f64,
    y:         &[f64],
    jac_cache: &mut CachedJacobian,
    tol:       f64,
    max_iter:  usize,
) -> Result<NewtonState> {
    let n = y.len();
    let factors = BlockFactors::build(&tableau.transform, &jac_cache.jac, h);

    newton_loop(problem, tableau, t, h, y, jac_cache, tol, max_iter,
        |resid| {
            let (r_real, r_complex) = tableau.transform.transform_residual(resid, n);
            let (z_real, z_complex) = factors.solve(&r_real, &r_complex)?;
            Some(tableau.transform.back_transform(&z_real, &z_complex, n))
        })
}

// ── Shared Newton iteration loop ─────────────────────────────────────────────
//
// Takes a closure `linear_solve: &[f64] -> Option<Vec<f64>>`
// that maps the residual to the correction ΔZ.

fn newton_loop(
    problem:      &dyn OdeProblem,
    tableau:      &RadauTableau,
    t:            f64,
    h:            f64,
    y:            &[f64],
    jac_cache:    &mut CachedJacobian,
    tol:          f64,
    max_iter:     usize,
    linear_solve: impl Fn(&[f64]) -> Option<Vec<f64>>,
) -> Result<NewtonState> {
    let n  = y.len();
    let s  = tableau.stages;
    let sn = s * n;

    let mut state = NewtonState::new(s, y);
    let mut fvals = vec![vec![0.0; n]; s];

    for iter in 0..max_iter {
        // Evaluate RHS at current stage values
        for i in 0..s {
            let ti = t + tableau.c[i] * h;
            problem.rhs(ti, &state.stages[i], &mut fvals[i]);
        }

        // Build residual F_{i,j} = Y_{i,j} - y_j - h Σ_q a_{iq} f_q(Y_q)_j
        let mut resid = vec![0.0_f64; sn];
        for i in 0..s {
            for j in 0..n {
                let mut acc = state.stages[i][j] - y[j];
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
                state.stages[i][j] += corr;
                max_corr = max_corr.max(corr.abs());
            }
        }

        state.iter_count = iter + 1;

        if max_corr < tol {
            state.converged = true;
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