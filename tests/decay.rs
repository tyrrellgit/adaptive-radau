use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

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

struct Decay {
    lambda: f64,
}

impl OdeProblem for Decay {
    fn dim(&self) -> usize { 1 }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = self.lambda * y[0];
    }
}

fn solve_decay(lambda: f64, t_end: f64, rtol: f64, atol: f64) -> f64 {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: 0.1,
            rtol,
            atol,
            ..Default::default()
        },
    );

    let result = solver.solve(t_end, &Decay { lambda }).unwrap();
    result.y[0]
}

#[test]
fn scalar_decay_loose_tolerance() {
    let got = solve_decay(-1.0, 1.0, 1e-4, 1e-6);
    let exact = (-1.0f64).exp();
    assert_close(got, exact, 1e-8, 1e-3);
}

#[test]
fn scalar_decay_tight_tolerance() {
    let got = solve_decay(-1.0, 1.0, 1e-10, 1e-12);
    let exact = (-1.0f64).exp();
    assert_close(got, exact, 1e-10, 1e-8);
}

#[test]
fn scalar_growth_positive_lambda() {
    let got = solve_decay(1.0, 1.0, 1e-10, 1e-12);
    let exact = 1.0f64.exp();
    assert_close(got, exact, 1e-10, 1e-8);
}

#[test]
fn stiff_fast_decay_lambda_100_midtime_matches_exact() {
    let got = solve_decay(-100.0, 0.05, 1e-10, 1e-12);
    let exact = (-5.0f64).exp();
    assert_close(got, exact, 1e-10, 1e-6);
}

#[test]
fn stiff_fast_decay_lambda_1000_midtime_matches_exact() {
    let got = solve_decay(-1000.0, 0.01, 1e-10, 1e-12);
    let exact = (-10.0f64).exp();
    assert_close(got, exact, 1e-10, 1e-6);
}

#[test]
fn stiff_fast_decay_lambda_1000_long_time_is_tiny() {
    let got = solve_decay(-1000.0, 1.0, 1e-8, 1e-10);
    assert!(got.abs() < 1e-10, "expected tiny solution, got {}", got);
}

#[test]
fn zero_lambda_stays_constant() {
    let got = solve_decay(0.0, 10.0, 1e-10, 1e-12);
    assert_close(got, 1.0, 1e-12, 1e-12);
}

#[test]
fn medium_and_tight_tolerances_agree() {
    let y_med = solve_decay(-1.0, 1.0, 1e-6, 1e-8);
    let y_tight = solve_decay(-1.0, 1.0, 1e-10, 1e-12);
    assert_close(y_med, y_tight, 1e-10, 1e-5);
}