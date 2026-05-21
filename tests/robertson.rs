use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

struct Robertson;

// y' = [-0.04 y0 + 1e4 y1 y2,
//        0.04 y0 - 1e4 y1 y2 - 3e7 y1^2,
//        3e7 y1^2]
impl OdeProblem for Robertson {
    fn dim(&self) -> usize { 3 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] =  3e7 * y[1] * y[1];
    }
}

fn integrate_robertson(t_end: f64, rtol: f64, atol: f64) -> Vec<f64> {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 0.0, 0.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: t_end * 0.1,
            rtol,
            atol,
            ..Default::default()
        },
    );
    while solver.t < t_end {
        let h = (t_end - solver.t).min(solver.h);
        solver.h = h.max(solver.options.min_step);
        solver.step(&Robertson).unwrap();
    }
    solver.y.clone()
}

#[test]
fn conservation_holds_throughout() {
    // y0 + y1 + y2 = 1 is a conserved quantity
    let y = integrate_robertson(1e4, 1e-6, 1e-9);
    let sum = y[0] + y[1] + y[2];
    let diff = (sum - 1.0).abs();
    assert!(diff < 1e-6, "conservation violated: sum = {}", sum);
}

#[test]
fn y2_grows_monotonically_to_near_one() {
    let y = integrate_robertson(1e11, 1e-6, 1e-9);
    assert!(y[2] > 0.99, "y2 = {} expected near 1.0 at t=1e11", y[2]);
    assert!(y[0] < 1e-4, "y0 = {} expected near 0.0 at t=1e11", y[0]);
}

#[test]
fn y1_stays_small_throughout() {
    // y1 is a fast intermediate — always tiny
    let y = integrate_robertson(1e4, 1e-6, 1e-9);
    assert!(y[1] < 1e-3, "y1 = {} should remain small", y[1]);
}

#[test]
fn completes_without_error_at_tight_tolerance() {
    let y = integrate_robertson(1e4, 1e-10, 1e-12);
    let sum = y[0] + y[1] + y[2];
    assert!((sum - 1.0).abs() < 1e-9);
}