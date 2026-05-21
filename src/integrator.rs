use crate::dense_output::DenseOutput;
use crate::error::{RadauError, Result};
use crate::newton::{solve_step, CachedJacobian};
use crate::order_control::{OrderChange, OrderController};
use crate::problem::OdeProblem;
use crate::step_control::{propose_step, wrms_norm};
use crate::tableau::TableauCache;

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
}

impl Default for IntegratorOptions {
    fn default() -> Self {
        Self {
            rtol:             1e-6,
            atol:             1e-9,
            initial_step:     1e-3,
            min_step:         1e-12,
            max_step:         1.0,
            initial_order:    5,
            min_order:        5,
            max_order:        13,
            newton_tol:       1e-10,
            max_newton_iter:  12,
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
        }
    }

    pub fn step(&mut self, problem: &dyn OdeProblem) -> Result<StepOutcome> {
        loop {
            let tableau = self.tableaux.get(self.order_control.current_order)?.clone();

            let newton = match solve_step(
                problem,
                &tableau,
                self.t,
                self.h,
                &self.y,
                &mut self.jac_cache,
                self.options.newton_tol,
                self.options.max_newton_iter,
            ) {
                Ok(n) => n,
                Err(RadauError::NewtonFailed) => {
                    // Jacobian already marked stale inside solve_step.
                    // Halve the step and retry.
                    self.h *= 0.5;
                    if self.h < self.options.min_step {
                        return Err(RadauError::StepUnderflow);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };

            let n = self.y.len();
            let s = tableau.stages;
            let mut fvals = vec![vec![0.0; n]; s];
            for i in 0..s {
                let ti = self.t + tableau.c[i] * self.h;
                problem.rhs(ti, &newton.stages[i], &mut fvals[i]);
            }

            // High-order solution
            let mut y_next = self.y.clone();
            // Embedded lower-order solution for error estimate
            let mut y_emb  = self.y.clone();
            for j in 0..n {
                for i in 0..s {
                    y_next[j] += self.h * tableau.b[i]     * fvals[i][j];
                    y_emb[j]  += self.h * tableau.b_hat[i] * fvals[i][j];
                }
            }

            let err: Vec<f64> = y_next.iter().zip(y_emb.iter())
                .map(|(a, b)| a - b)
                .collect();
            let err_norm = wrms_norm(&err, &y_next, self.options.rtol, self.options.atol);

            let order_change = self.order_control.update(newton.iter_count);

            // If order changed, invalidate Jacobian cache so we refactor
            // with the correct stage structure on the next step
            if order_change != OrderChange::Unchanged {
                self.jac_cache.age = self.jac_cache.max_age;
            }

            let (h_new, accepted) = propose_step(
                self.h,
                err_norm,
                self.order_control.current_order,
                newton.iter_count,
            );

            if accepted {
                let dense = DenseOutput {
                    t_n:        self.t,
                    h:          self.h,
                    y_n:        self.y.clone(),
                    stage_vals: newton.stages.clone(),
                    c:          tableau.c.clone(),
                };

                self.t += self.h;
                self.y  = y_next.clone();
                self.h  = h_new.clamp(self.options.min_step, self.options.max_step);

                return Ok(StepOutcome {
                    t:            self.t,
                    y:            y_next,
                    dense,
                    accepted:     true,
                    newton_iters: newton.iter_count,
                    order_change,
                });
            }

            // Step rejected — shrink step and mark Jacobian stale
            self.jac_cache.age = self.jac_cache.max_age;
            self.h = h_new.clamp(self.options.min_step, self.options.max_step);
            if self.h <= self.options.min_step {
                return Err(RadauError::StepUnderflow);
            }
        }
    }
}