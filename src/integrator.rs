use nalgebra::{DMatrix, DVector};

use crate::dense_output::DenseOutput;
use crate::error::{RadauError, Result};
use crate::newton::{solve_step, CachedJacobian, NewtonLinearAlgebra};
use crate::order_control::{OrderChange, OrderController};
use crate::problem::OdeProblem;
use crate::step_control::{propose_step, wrms_norm};
use crate::tableau::TableauCache;

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub t: f64,
    pub y: Vec<f64>,
    pub steps_accepted: usize,
    pub steps_rejected: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum LinearSolveStrategy {
    Auto,
    Kronecker,
    BlockDiagonal,
}

#[derive(Debug, Clone)]
pub struct IntegratorOptions {
    pub rtol: f64,
    pub atol: f64,
    pub initial_step: f64,
    pub min_step: f64,
    pub max_step: f64,
    pub initial_order: usize,
    pub min_order: usize,
    pub max_order: usize,
    pub newton_tol: f64,
    pub max_newton_iter: usize,

    pub linear_solver: LinearSolveStrategy,
    pub block_diag_warn_below: usize,
}

impl Default for IntegratorOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            initial_step: 1e-3,
            min_step: 1e-12,
            max_step: 1.0,
            initial_order: 5,
            min_order: 5,
            max_order: 13,
            newton_tol: 1e-10,
            max_newton_iter: 12,

            linear_solver: LinearSolveStrategy::Auto,
            block_diag_warn_below: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepOutcome {
    pub t: f64,
    pub y: Vec<f64>,
    pub dense: DenseOutput,
    pub accepted: bool,
    pub newton_iters: usize,
    pub order_change: OrderChange,
}

pub struct RadauIntegrator {
    pub t: f64,
    pub y: Vec<f64>,
    pub h: f64,
    pub options: IntegratorOptions,
    pub order_control: OrderController,
    pub tableaux: TableauCache,
    pub jac_cache: CachedJacobian,
    warned_block_diag_small: bool,
}

impl RadauIntegrator {
    pub fn new(t0: f64, y0: Vec<f64>, options: IntegratorOptions) -> Self {
        let h0 = options.initial_step;
        let (io, lo, ho) = (options.initial_order, options.min_order, options.max_order);
        let n = y0.len();

        Self {
            t: t0,
            y: y0,
            h: h0,
            order_control: OrderController::new(io, lo, ho),
            options,
            tableaux: TableauCache::default(),
            jac_cache: CachedJacobian::new(n),
            warned_block_diag_small: false,
        }
    }

    pub fn solve(
        &mut self,
        t_end: f64,
        problem: &impl OdeProblem,
    ) -> Result<SolveResult> {
        assert!(t_end > self.t, "t_end must be greater than initial t");

        let mut steps_accepted = 0;
        let mut steps_rejected = 0;

        while t_end - self.t > 1e-14 * t_end.abs().max(1.0) {
            self.h = self.h.min(t_end - self.t).max(self.options.min_step);

            let step = self.step(problem)?;

            if step.accepted {
                steps_accepted += 1;
            } else {
                steps_rejected += 1;
            }
        }

        Ok(SolveResult {
            t: self.t,
            y: self.y.clone(),
            steps_accepted,
            steps_rejected,
        })
    }

    fn choose_linear_algebra(&mut self, n: usize) -> NewtonLinearAlgebra {
        match self.options.linear_solver {
            LinearSolveStrategy::Auto => {
                if n <= self.options.block_diag_warn_below {
                    NewtonLinearAlgebra::KroneckerFull
                } else {
                    NewtonLinearAlgebra::BlockDiagonal
                }
            }
            LinearSolveStrategy::Kronecker => NewtonLinearAlgebra::KroneckerFull,
            LinearSolveStrategy::BlockDiagonal => {
                if n < self.options.block_diag_warn_below && !self.warned_block_diag_small {
                    eprintln!(
                        "warning: block-diagonal linear solver selected for dim={}, below recommended threshold {}; this may be slower than the direct Kronecker solve",
                        n, self.options.block_diag_warn_below
                    );
                    self.warned_block_diag_small = true;
                }
                NewtonLinearAlgebra::BlockDiagonal
            }
        }
    }

    pub fn step(&mut self, problem: &dyn OdeProblem) -> Result<StepOutcome> {
        loop {
            let tableau = self.tableaux.get(self.order_control.current_order)?.clone();
            let n = self.y.len();
            let linear_algebra = self.choose_linear_algebra(n);

            let newton = match solve_step(
                problem,
                &tableau,
                self.t,
                self.h,
                &self.y,
                &mut self.jac_cache,
                self.options.newton_tol,
                self.options.max_newton_iter,
                linear_algebra,
            ) {
                Ok(n) => n,
                Err(RadauError::NewtonFailed) => {
                    self.h *= 0.5;
                    if self.h < self.options.min_step {
                        return Err(RadauError::StepUnderflow);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };

            let s = tableau.stages;
            let mut fvals = vec![vec![0.0; n]; s];
            for i in 0..s {
                let ti = self.t + tableau.c[i] * self.h;
                problem.rhs(ti, &newton.stages[i], &mut fvals[i]);
            }

            // y_next from the main weights b (stiffly-accurate: b == last row of A,
            // so y_next == stages[s-1], but we keep the explicit form for clarity).
            let mut y_next = self.y.clone();
            for j in 0..n {
                for i in 0..s {
                    y_next[j] += self.h * tableau.b[i] * fvals[i][j];
                }
            }

            // Hairer's ESTRAD error estimator (dc_decsol.f, subroutine ESTRAD).
            //   F2[j]   = sum_i (DD_i / h) * (Y_i - y_n)[j]    // stage *increments*
            //   CONT[j] = F2[j] + f(t_n, y_n)[j]               // add f at y_n
            //   err[j]  = (I - h γ J)^{-1} CONT                 // smooth
            // The DD coefficients satisfy moment cancellation against the leading
            // collocation residual: F2 ≈ -f(y_n) to (s-1)th order, so CONT is
            // O(h^s) and matches the embedded-method order naturally.
            let mut f_yn = vec![0.0_f64; n];
            problem.rhs(self.t, &self.y, &mut f_yn);

            // Use stage *increments* Z_i = Y_i - y_n directly from the Newton
            // state, not Y_i - y_n via subtraction. Subtraction would lose
            // precision when |Y_i| ≈ |y_n| but Z_i is small, which is exactly
            // the regime where ESTRAD's 1/h scaling amplifies any noise into
            // the error estimate.
            let mut cont = vec![0.0_f64; n];
            for j in 0..n {
                let mut f2 = 0.0_f64;
                for i in 0..s {
                    f2 += tableau.dd[i] * newton.increments[i][j];
                }
                cont[j] = f2 / self.h + f_yn[j];
            }
            let err_smoothed = smooth_error(&cont, &self.jac_cache.jac, self.h, tableau.u1);

            let err_norm = wrms_norm(
                &err_smoothed,
                &self.y,
                &y_next,
                self.options.rtol,
                self.options.atol,
            );

            let (h_new, accepted) = propose_step(self.h, err_norm, tableau.embedded_order);

            if accepted {
                let dense = DenseOutput {
                    t_n: self.t,
                    h: self.h,
                    y_n: self.y.clone(),
                    stage_vals: newton.stages.clone(),
                    c: tableau.c.clone(),
                };

                self.t += self.h;
                self.y = y_next.clone();
                self.h = h_new.clamp(self.options.min_step, self.options.max_step);

                let order_change = self.order_control.update(newton.iter_count);
                if order_change != OrderChange::Unchanged {
                    self.jac_cache.mark_stale();
                }

                return Ok(StepOutcome {
                    t: self.t,
                    y: y_next,
                    dense,
                    accepted: true,
                    newton_iters: newton.iter_count,
                    order_change,
                });
            }

            // Step rejected: shrink h, keep Jacobian (still valid at current
            // y), and retry without updating the order controller.
            self.h = h_new.clamp(self.options.min_step, self.options.max_step);
            if self.h <= self.options.min_step {
                return Err(RadauError::StepUnderflow);
            }
        }
    }
}

/// Hairer ESTRAD smoothing: solve (I − h γ J) · err_smoothed = err.
/// Cost: one extra n×n LU per step. We could cache this factor along the
/// real block of the block-diagonal solver; left as a future optimisation.
fn smooth_error(err: &[f64], jac: &DMatrix<f64>, h: f64, u1: f64) -> Vec<f64> {
    let n = err.len();
    let gamma = 1.0 / u1;
    let mut m = DMatrix::<f64>::identity(n, n);
    for r in 0..n {
        for c in 0..n {
            m[(r, c)] -= h * gamma * jac[(r, c)];
        }
    }
    let rhs = DVector::from_column_slice(err);
    match m.lu().solve(&rhs) {
        Some(sol) => sol.iter().copied().collect(),
        None => err.to_vec(), // singular: fall back to raw err
    }
}
