use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

fn assert_close(x: f64, y: f64, tol: f64) {
    let diff = (x - y).abs();
    assert!(diff <= tol, "|{} - {}| = {} > {}", x, y, diff, tol);
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

fn integrate(lambda: f64, t_end: f64, rtol: f64, atol: f64) -> f64 {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0],
        IntegratorOptions {
            initial_step: 1e-3,
            max_step: 0.1,
            rtol,
            atol,
            ..Default::default()
        },
    );
    while solver.t < t_end {
        let h = (t_end - solver.t).min(solver.h);
        solver.h = h.max(solver.options.min_step);
        solver.step(&Decay { lambda }).unwrap();
    }
    solver.y[0]
}

#[test]
fn scalar_decay_loose_tolerance() {
    let got = integrate(-1.0, 1.0, 1e-4, 1e-6);
    assert_close(got, (-1.0f64).exp(), 1e-3);
}

#[test]
fn scalar_decay_tight_tolerance() {
    let got = integrate(-1.0, 1.0, 1e-10, 1e-12);
    assert_close(got, (-1.0f64).exp(), 1e-8);
}

#[test]
fn stiff_fast_decay_lambda_1000() {
    let got = integrate(-1000.0, 1.0, 1e-6, 1e-9);
    assert_close(got, (-1000.0f64).exp(), 1e-5);
}

#[test]
fn grows_correctly_for_positive_lambda() {
    let got = integrate(1.0, 1.0, 1e-8, 1e-10);
    assert_close(got, 1.0f64.exp(), 1e-5);
}