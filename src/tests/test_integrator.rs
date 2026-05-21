use crate::integrator::{IntegratorOptions, RadauIntegrator};
use crate::problem::OdeProblem;

fn assert_close(x: f64, y: f64, tol: f64) {
    let diff = (x - y).abs();
    assert!(diff <= tol, "|{} - {}| = {} > {}", x, y, diff, tol);
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

    assert_close(result.y[0], (-1.0f64).exp(), 5e-4);
}

#[test]
fn dense_output_reproduces_left_endpoint() {
    let mut solver = RadauIntegrator::new(0.0, vec![1.0], IntegratorOptions {
        initial_step: 1e-2,
        max_step: 1e-2,
        ..Default::default()
    });
    let step = solver.step(&Decay).unwrap();
    assert_close(step.dense.evaluate(step.dense.t_n)[0], 1.0, 1e-10);
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
    assert_close(step.dense.evaluate(t_end)[0], step.y[0], 1e-8);
}