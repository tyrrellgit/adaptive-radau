//! Equivalence tests for the block-diagonal (Schur) Newton linear algebra.
//!
//! The block-diagonal path must solve the *same* Newton system as the direct
//! Kronecker path — it is only a cheaper factorisation of it. So for any
//! problem, both strategies must produce the same trajectory to round-off.

use radau_rs::{IntegratorOptions, LinearSolveStrategy, OdeProblem, RadauIntegrator};

struct Decay {
    lambda: f64,
}
impl OdeProblem for Decay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = self.lambda * y[0];
    }
}

/// 2D system with complex Jacobian eigenvalues, exercising the 2x2 blocks.
struct Spiral;
impl OdeProblem for Spiral {
    fn dim(&self) -> usize { 2 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.1 * y[0] + 10.0 * y[1];
        dydt[1] = -10.0 * y[0] - 0.1 * y[1];
    }
}

fn opts(strategy: LinearSolveStrategy, order: usize) -> IntegratorOptions {
    IntegratorOptions {
        initial_step: 1e-5,
        max_step: 0.05,
        rtol: 1e-10,
        atol: 1e-12,
        linear_solver: strategy,
        // Keep the order fixed so both runs use an identical tableau.
        initial_order: order,
        min_order: order,
        max_order: order,
        ..Default::default()
    }
}

fn solve_with<P: OdeProblem>(
    p: &P,
    y0: Vec<f64>,
    t_end: f64,
    s: LinearSolveStrategy,
    order: usize,
) -> Vec<f64> {
    let mut solver = RadauIntegrator::new(0.0, y0, opts(s, order));
    solver.solve(t_end, p).expect("solve failed").y
}

/// Orders 5/9/13 give one, two and three 2x2 Schur blocks respectively, so
/// this covers the strictly-upper coupling entries that a pure block-diagonal
/// treatment would silently drop.
#[test]
fn block_diag_matches_kronecker_on_scalar_decay() {
    let p = Decay { lambda: -1.0 };
    for order in [5, 9, 13] {
        let kron = solve_with(&p, vec![1.0], 1.0, LinearSolveStrategy::Kronecker, order);
        let block = solve_with(&p, vec![1.0], 1.0, LinearSolveStrategy::BlockDiagonal, order);
        let diff = (kron[0] - block[0]).abs();
        assert!(
            diff < 1e-10,
            "order {}: kronecker {} vs block-diagonal {} (diff {})",
            order, kron[0], block[0], diff
        );
    }
}

#[test]
fn block_diag_matches_exact_solution_on_scalar_decay() {
    let p = Decay { lambda: -1.0 };
    let exact = (-1.0f64).exp();
    for order in [5, 9, 13] {
        let got = solve_with(&p, vec![1.0], 1.0, LinearSolveStrategy::BlockDiagonal, order);
        assert!(
            (got[0] - exact).abs() < 1e-8,
            "order {}: got {}, exact {}", order, got[0], exact
        );
    }
}

#[test]
fn block_diag_matches_kronecker_on_complex_spectrum() {
    let t = 0.3_f64;
    for order in [5, 9, 13] {
        let kron = solve_with(&Spiral, vec![1.0, 0.0], t, LinearSolveStrategy::Kronecker, order);
        let block = solve_with(&Spiral, vec![1.0, 0.0], t, LinearSolveStrategy::BlockDiagonal, order);
        for i in 0..2 {
            let diff = (kron[i] - block[i]).abs();
            assert!(
                diff < 1e-9,
                "order {} component {}: kronecker {} vs block {} (diff {})",
                order, i, kron[i], block[i], diff
            );
        }
    }
}

#[test]
fn block_diag_matches_exact_solution_on_complex_spectrum() {
    let t = 0.3_f64;
    let decay = (-0.1 * t).exp();
    let exact = [decay * (10.0 * t).cos(), -decay * (10.0 * t).sin()];
    for order in [5, 9, 13] {
        let got = solve_with(&Spiral, vec![1.0, 0.0], t, LinearSolveStrategy::BlockDiagonal, order);
        for i in 0..2 {
            assert!(
                (got[i] - exact[i]).abs() < 1e-7,
                "order {} component {}: got {}, exact {}", order, i, got[i], exact[i]
            );
        }
    }
}
