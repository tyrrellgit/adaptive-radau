use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

struct VanDerPol {
    mu: f64,
}

impl OdeProblem for VanDerPol {
    fn dim(&self) -> usize { 2 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = y[1];
        dydt[1] = self.mu * ((1.0 - y[0] * y[0]) * y[1] - y[0]);
    }
}

fn integrate(mu: f64, t_end: f64, rtol: f64, atol: f64) -> Vec<f64> {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![2.0, 0.0],
        IntegratorOptions {
            initial_step: 1e-3,
            max_step: t_end * 0.01,
            rtol,
            atol,
            ..Default::default()
        },
    );
    while solver.t < t_end {
        let h = (t_end - solver.t).min(solver.h);
        solver.h = h.max(solver.options.min_step);
        solver.step(&VanDerPol { mu }).unwrap();
    }
    solver.y.clone()
}

#[test]
fn mildly_stiff_mu1_stays_bounded() {
    let y = integrate(1.0, 10.0, 1e-6, 1e-8);
    assert!(y[0].abs() < 3.0, "y[0] = {}", y[0]);
    assert!(y[1].abs() < 10.0, "y[1] = {}", y[1]);
}

#[test]
fn highly_stiff_mu1000_completes_and_stays_bounded() {
    let y = integrate(1000.0, 2000.0, 1e-4, 1e-6);
    assert!(y[0].abs() < 3.0, "y[0] = {}", y[0]);
}

#[test]
fn solution_is_periodic_for_mu1() {
    // One full period of the mu=1 limit cycle is approximately T≈6.66
    let y_one_period = integrate(1.0, 6.66, 1e-8, 1e-10);
    let y_two_periods = integrate(1.0, 13.32, 1e-8, 1e-10);
    let diff = (y_one_period[0] - y_two_periods[0]).abs();
    assert!(diff < 0.1, "solution not periodic: diff = {}", diff);
}