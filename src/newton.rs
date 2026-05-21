use nalgebra::{DMatrix, DVector};

use crate::error::{RadauError, Result};
use crate::problem::OdeProblem;
use crate::tableau::RadauTableau;

// ── Cached Jacobian ──────────────────────────────────────────────────────────

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

// ── Newton state ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NewtonState {
    pub stages: Vec<Vec<f64>>,
    pub iter_count: usize,
    pub converged: bool,
}

impl NewtonState {
    pub fn new(stage_count: usize, y0: &[f64]) -> Self {
        Self {
            stages: vec![y0.to_vec(); stage_count],
            iter_count: 0,
            converged: false,
        }
    }
}

// ── Main solver ───────────────────────────────────────────────────────────────
//
// Solves the coupled sn×sn Newton system in one shot:
//
//   (I_sn  -  h * A⊗J) * ΔZ  =  -F
//
// M is factored ONCE per step and the same LU is reused
// for every Newton iteration — only the RHS changes.

pub fn solve_step(
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

    if jac_cache.is_stale() {
        jac_cache.refresh(problem, t, y);
    }

    // Build sn×sn Newton matrix  M = I - h*(A⊗J)
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

    // Factor ONCE — reused for every Newton iteration
    let lu = m.lu();

    let mut state = NewtonState::new(s, y);
    let mut fvals = vec![vec![0.0; n]; s];

    for iter in 0..max_iter {
        // Evaluate RHS at current stage values
        for i in 0..s {
            let ti = t + tableau.c[i] * h;
            problem.rhs(ti, &state.stages[i], &mut fvals[i]);
        }

        // Build residual F (length sn)
        let mut resid = DVector::<f64>::zeros(sn);
        for i in 0..s {
            for j in 0..n {
                let mut acc = state.stages[i][j] - y[j];
                for q in 0..s {
                    acc -= h * tableau.a[(i, q)] * fvals[q][j];
                }
                resid[i * n + j] = acc;
            }
        }

        // Single solve per iteration
        let Some(dz) = lu.solve(&-resid) else {
            return Err(RadauError::LinearSolveFailed);
        };

        let mut max_corr: f64 = 0.0;
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

        // Stagnating after several iterations → force refresh next step
        if iter >= 4 && max_corr > 0.5 {
            jac_cache.mark_stale();
        }
    }

    jac_cache.mark_stale();
    Err(RadauError::NewtonFailed)
}