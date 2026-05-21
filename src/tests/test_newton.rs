use crate::newton::CachedJacobian;
use crate::problem::OdeProblem;

struct Decay;
impl OdeProblem for Decay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[0];
    }
}

#[test]
fn cached_jacobian_starts_stale() {
    let jac = CachedJacobian::new(3);
    assert!(jac.is_stale());
}

#[test]
fn cached_jacobian_not_stale_after_refresh() {
    let mut jac = CachedJacobian::new(1);
    jac.refresh(&Decay, 0.0, &[1.0]);
    assert!(!jac.is_stale());
}

#[test]
fn cached_jacobian_becomes_stale_after_max_age_increments() {
    let mut jac = CachedJacobian::new(1);
    jac.refresh(&Decay, 0.0, &[1.0]);
    for _ in 0..jac.max_age {
        jac.increment_age();
    }
    assert!(jac.is_stale());
}

#[test]
fn cached_jacobian_finite_difference_is_reasonable_for_decay() {
    let mut jac = CachedJacobian::new(1);
    jac.refresh(&Decay, 0.0, &[1.0]);
    let j00 = jac.jac[(0, 0)];
    let diff = (j00 - (-1.0)).abs();
    assert!(diff < 1e-6, "expected J≈-1, got {}", j00);
}