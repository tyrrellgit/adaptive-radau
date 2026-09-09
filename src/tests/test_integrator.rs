use crate::integrator::{IntegratorOptions, RadauIntegrator};
use crate::problem::OdeProblem;

fn assert_close(x: f64, y: f64, atol: f64, rtol: f64) {
    let diff = (x - y).abs();
    let scale = atol + rtol * x.abs().max(y.abs());
    assert!(
        diff <= scale,
        "|{} - {}| = {} > {} (atol={}, rtol={})",
        x,
        y,
        diff,
        scale,
        atol,
        rtol
    );
}

struct Decay;
impl OdeProblem for Decay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[0];
    }
}

#[test]
fn takes_one_accepted_step() {
    let mut solver = RadauIntegrator::new(0.0, vec![1.0], IntegratorOptions {
        initial_step: 1e-3,
        max_step: 0.1,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    });

    let step = solver.step(&Decay).unwrap();
    assert!(step.accepted);
    assert!(step.t > 0.0);
    assert!(step.y[0] < 1.0);
    assert!(step.newton_iters >= 1);
}

#[test]
fn integrates_decay_to_t1_within_tolerance() {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-3,
            max_step: 0.2,
            rtol: 1e-8,
            atol: 1e-10,
            ..Default::default()
        },
    );

    let result = solver.solve(1.0, &Decay).unwrap();
    let exact = (-1.0f64).exp();
    assert_close(result.y[0], exact, 1e-8, 1e-6);
}

#[test]
fn dense_output_reproduces_left_endpoint() {
    let mut solver = RadauIntegrator::new(0.0, vec![1.0], IntegratorOptions {
        initial_step: 1e-2,
        max_step: 1e-2,
        ..Default::default()
    });

    let step = solver.step(&Decay).unwrap();
    assert_close(step.dense.evaluate(step.dense.t_n)[0], 1.0, 1e-12, 1e-12);
}

#[test]
fn dense_output_reproduces_right_endpoint() {
    let mut solver = RadauIntegrator::new(0.0, vec![1.0], IntegratorOptions {
        initial_step: 1e-2,
        max_step: 1e-2,
        rtol: 1e-9,
        atol: 1e-12,
        ..Default::default()
    });

    let step = solver.step(&Decay).unwrap();
    let t_end = step.dense.t_n + step.dense.h;
    assert_close(step.dense.evaluate(t_end)[0], step.y[0], 1e-10, 1e-8);
}

#[test]
fn dense_output_matches_decay_midpoint() {
    let mut solver = RadauIntegrator::new(0.0, vec![1.0], IntegratorOptions {
        initial_step: 1e-2,
        max_step: 1e-2,
        rtol: 1e-10,
        atol: 1e-12,
        ..Default::default()
    });

    let step = solver.step(&Decay).unwrap();
    let t_mid = step.dense.t_n + 0.5 * step.dense.h;
    let exact = (-t_mid).exp();
    let y_mid = step.dense.evaluate(t_mid)[0];

    assert_close(y_mid, exact, 1e-9, 1e-6);
}

// ============================================================================
// Stiff problems with analytical solutions
// ============================================================================

struct StiffDecay {
    lambda: f64,
}

impl OdeProblem for StiffDecay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.lambda * y[0];
    }
}

#[test]
fn stiff_decay_lambda_100_midtime_matches_exact() {
    let lambda = 100.0_f64;
    let t_end = 0.05_f64;
    let exact_y = (-lambda * t_end).exp();

    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-5,
            max_step: 0.05,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(t_end, &StiffDecay { lambda }).unwrap();
    assert_close(result.y[0], exact_y, 1e-10, 1e-6);
}

#[test]
fn stiff_decay_lambda_100_to_t1_is_small() {
    let lambda = 100.0_f64;

    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-5,
            max_step: 0.1,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(1.0, &StiffDecay { lambda }).unwrap();
    assert!(result.y[0].abs() < 1e-10, "expected near-zero solution, got {}", result.y[0]);
}

#[test]
fn stiff_decay_lambda_1000_midtime_matches_exact() {
    let lambda = 1000.0_f64;
    let t_end = 0.01_f64;
    let exact_y = (-lambda * t_end).exp();

    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: 0.01,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(t_end, &StiffDecay { lambda }).unwrap();
    assert_close(result.y[0], exact_y, 1e-10, 1e-6);
}

#[test]
fn stiff_decay_lambda_1000_to_t1_is_tiny() {
    let lambda = 1000.0_f64;

    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: 0.1,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(1.0, &StiffDecay { lambda }).unwrap();
    assert!(result.y[0].abs() < 1e-12, "expected tiny solution, got {}", result.y[0]);
}

/// Prothero-Robinson equation:
///     y' = lambda * (y - cos(t)) - sin(t)
/// Exact solution with y(0)=1 is y(t)=cos(t).
///
/// This is a standard stiff order-reduction test.
struct ProtheroRobinson {
    lambda: f64,
}

impl OdeProblem for ProtheroRobinson {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = self.lambda * (y[0] - t.cos()) - t.sin();
    }
}

#[test]
fn prothero_robinson_stiff_solution_tracks_exact() {
    let lambda = -1000.0_f64;
    let t_end = 1.0_f64;
    let exact_y = t_end.cos();

    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-5,
            max_step: 0.05,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(t_end, &ProtheroRobinson { lambda }).unwrap();
    assert_close(result.y[0], exact_y, 1e-6, 1e-4);
}

#[test]
fn prothero_robinson_intermediate_points_accurate() {
    let lambda = -100.0_f64;
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-4,
            max_step: 0.05,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let t_eval = [0.2_f64, 0.5, 0.8, 1.0];
    for &t_end in &t_eval {
        let result = solver.solve(t_end, &ProtheroRobinson { lambda }).unwrap();
        let exact = t_end.cos();
        assert_close(result.y[0], exact, 1e-6, 5e-4);
    }
}

/// 2D stiff linear system:
///     y1' = -y1
///     y2' = -1000*y2
/// Exact:
///     y1(t) = exp(-t)
///     y2(t) = exp(-1000*t)
struct MixedTimescale;

impl OdeProblem for MixedTimescale {
    fn dim(&self) -> usize { 2 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[0];
        dydt[1] = -1000.0 * y[1];
    }
}

#[test]
fn mixed_timescale_slow_component_and_fast_component_at_t1() {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 1.0],
        IntegratorOptions {
            initial_step: 1e-5,
            max_step: 0.2,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(1.0, &MixedTimescale).unwrap();
    let exact_y1 = (-1.0f64).exp();

    assert_close(result.y[0], exact_y1, 1e-10, 1e-6);
    assert!(result.y[1].abs() < 1e-12, "expected tiny fast component, got {}", result.y[1]);
}

#[test]
fn mixed_timescale_fast_component_decays_early() {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 1.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: 0.01,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(0.01, &MixedTimescale).unwrap();
    let exact_y1 = (-0.01f64).exp();
    let exact_y2 = (-10.0f64).exp();

    assert_close(result.y[0], exact_y1, 1e-10, 1e-6);
    assert_close(result.y[1], exact_y2, 1e-10, 1e-6);
}

/// Oscillatory damped linear system:
///     y1' = -0.1*y1 + 10*y2
///     y2' = -10*y1 - 0.1*y2
///
/// With y(0) = [1, 0], the exact solution is:
///     y1(t) = exp(-0.1 t) cos(10 t)
///     y2(t) = -exp(-0.1 t) sin(10 t)
///
/// The amplitude decays exponentially; the squared norm decays like exp(-0.2 t).
struct ComplexEigenvaluesStiff;

impl OdeProblem for ComplexEigenvaluesStiff {
    fn dim(&self) -> usize { 2 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.1 * y[0] + 10.0 * y[1];
        dydt[1] = -10.0 * y[0] - 0.1 * y[1];
    }
}

#[test]
fn complex_eigenvalues_matches_exact_solution() {
    let t = 0.3_f64;

    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 0.0],
        IntegratorOptions {
            initial_step: 1e-5,
            max_step: 0.02,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(t, &ComplexEigenvaluesStiff).unwrap();
    let decay = (-0.1 * t).exp();
    let exact_y1 = decay * (10.0 * t).cos();
    let exact_y2 = -decay * (10.0 * t).sin();

    assert_close(result.y[0], exact_y1, 1e-9, 1e-6);
    assert_close(result.y[1], exact_y2, 1e-9, 1e-6);
}

#[test]
fn complex_eigenvalues_energy_decay() {
    let t = 1.0_f64;

    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 0.0],
        IntegratorOptions {
            initial_step: 1e-5,
            max_step: 0.05,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(t, &ComplexEigenvaluesStiff).unwrap();
    let energy = result.y[0] * result.y[0] + result.y[1] * result.y[1];
    let exact_energy = (-0.2 * t).exp();

    assert_close(energy, exact_energy, 1e-8, 1e-6);
}

/// Slow-fast diagonal system:
///     y1' = -y1
///     y2' = -(1/epsilon) y2, epsilon = 1e-2
struct SlowFastSystem;

impl OdeProblem for SlowFastSystem {
    fn dim(&self) -> usize { 2 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let epsilon = 1e-2;
        dydt[0] = -y[0];
        dydt[1] = -(1.0 / epsilon) * y[1];
    }
}

#[test]
fn slow_fast_system_long_time_integration() {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 1.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: 0.5,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        },
    );

    let result = solver.solve(5.0, &SlowFastSystem).unwrap();
    let exact_y1 = (-5.0f64).exp();

    assert_close(result.y[0], exact_y1, 1e-10, 1e-6);
    assert!(result.y[1].abs() < 1e-12, "expected tiny fast component, got {}", result.y[1]);
}