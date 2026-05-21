use radau_rs::{DenseOutput, IntegratorOptions, OdeProblem, OrderChange,
               OrderController, RadauIntegrator, RadauTableau};
use radau_rs::step_control::{propose_step, wrms_norm};

struct ExpDecay;
impl OdeProblem for ExpDecay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) { dydt[0] = -y[0]; }
}

#[test]
fn order_controller_raises_when_iterations_are_low() {
    let mut ctrl = OrderController::new(5, 5, 25);
    assert_eq!(ctrl.update(2), OrderChange::Raised);
    assert_eq!(ctrl.current_order, 9);
}

#[test]
fn order_controller_lowers_when_iterations_are_high() {
    let mut ctrl = OrderController::new(13, 5, 25);
    ctrl.histiter = 8.5;
    assert_eq!(ctrl.update(10), OrderChange::Lowered);
    assert_eq!(ctrl.current_order, 9);
}

#[test]
fn wrms_norm_is_zero_for_zero_error() {
    assert_eq!(wrms_norm(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], 1e-6, 1e-9), 0.0);
}

#[test]
fn propose_step_accepts_small_error_rejects_large() {
    let (_, ok1) = propose_step(0.1, 0.5, 5, 4);
    assert!(ok1);
    let (h2, ok2) = propose_step(0.1, 2.0, 5, 4);
    assert!(!ok2);
    assert!(h2 < 0.1);
}

#[test]
fn dense_output_reproduces_endpoints() {
    let d = DenseOutput {
        t_n: 0.0, h: 1.0,
        y_n: vec![1.0],
        stage_vals: vec![vec![1.2], vec![1.6], vec![2.0]],
        c: vec![0.2, 0.6, 1.0],
    };
    assert!((d.evaluate(0.0)[0] - 1.0).abs() < 1e-12);
    assert!((d.evaluate(1.0)[0] - 2.0).abs() < 1e-12);
}

#[test]
fn order5_tableau_shape_and_weights() {
    let tab = RadauTableau::for_order(5).unwrap();
    assert_eq!(tab.stages, 3);
    assert_eq!(tab.b.len(), 3);
    let sum_b: f64 = tab.b.iter().sum();
    assert!((sum_b - 1.0).abs() < 1e-12);
}

#[test]
fn integrator_smoke_test_exp_decay() {
    let mut solver = RadauIntegrator::new(
        0.0, vec![1.0],
        IntegratorOptions { initial_step: 1e-3, ..Default::default() },
    );
    let s = solver.step(&ExpDecay).unwrap();
    assert!(s.accepted);
    assert!(s.y[0] < 1.0 && s.y[0].is_finite());
}