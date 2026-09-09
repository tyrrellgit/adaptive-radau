//! Order check on the embedded error estimator.
//!
//! For an s-stage Radau IIA method Hairer's ESTRAD estimator has order s, so
//! the estimate must scale as `h^(s+1)`. This pins the scaling of the
//! smoothing solve: using the normalised `(I - (h/u1) J)` in place of radau5's
//! `E1 = (u1/h) I - J` costs exactly one power of h and shows up here as a
//! slope of `s` instead of `s+1`.

use crate::integrator::{IntegratorOptions, RadauIntegrator};
use crate::problem::OdeProblem;

struct Decay;
impl OdeProblem for Decay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) { dydt[0] = -y[0]; }
}

/// One step of size `h` at fixed order; returns the error-estimate norm.
/// `rtol = 1`, `atol = 0` makes `scal = |y| ~ 1`, so the norm is the estimate.
fn estimate_for(order: usize, h: f64) -> f64 {
    let mut solver = RadauIntegrator::new(0.0, vec![1.0], IntegratorOptions {
        initial_step: h,
        min_step: h,
        max_step: h,
        rtol: 1.0,
        atol: 0.0,
        initial_order: order,
        min_order: order,
        max_order: order,
        ..Default::default()
    });
    solver.step(&Decay).expect("step").err_norm
}

#[test]
fn error_estimate_scales_as_h_to_the_s_plus_one() {
    for order in [5usize, 9, 13] {
        let s = (order + 1) / 2;
        let expected = (s + 1) as f64;

        // Stay in the range where the estimate is well above f64 noise: the
        // order-13 estimate hits ~1e-16 by h = 0.05.
        let (h1, h2) = match order {
            5 => (0.2, 0.1),
            9 => (0.2, 0.1),
            _ => (0.4, 0.2),
        };
        let (e1, e2) = (estimate_for(order, h1), estimate_for(order, h2));
        let slope = (e2 / e1).ln() / (h2 / h1).ln();

        assert!(
            (slope - expected).abs() < 0.35,
            "order {}: estimate slope {:.3}, expected ~{} (est {:.3e} at h={}, {:.3e} at h={})",
            order, slope, expected, e1, h1, e2, h2
        );
    }
}

/// The estimate must be a genuine bound-ish proxy: comfortably larger than the
/// true local error (it is the lower-order embedded estimate) but not absurdly
/// so at a moderate step.
#[test]
fn error_estimate_exceeds_true_local_error() {
    let h = 0.1_f64;
    let mut solver = RadauIntegrator::new(0.0, vec![1.0], IntegratorOptions {
        initial_step: h, min_step: h, max_step: h,
        rtol: 1.0, atol: 0.0,
        initial_order: 5, min_order: 5, max_order: 5,
        ..Default::default()
    });
    let out = solver.step(&Decay).expect("step");
    let true_local = (out.y[0] - (-h).exp()).abs();
    assert!(
        out.err_norm > true_local,
        "estimate {:.3e} should exceed true local error {:.3e}",
        out.err_norm, true_local
    );
}
