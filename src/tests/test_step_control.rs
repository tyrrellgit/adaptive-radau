use crate::step_control::{propose_step, wrms_norm};

fn assert_close(x: f64, y: f64, tol: f64) {
    let diff = (x - y).abs();
    assert!(diff <= tol, "|{} - {}| = {} > {}", x, y, diff, tol);
}

#[test]
fn wrms_norm_of_zero_error_is_zero() {
    assert_close(
        wrms_norm(&[0.0, 0.0], &[1.0, 2.0], &[1.0, 2.0], 1e-6, 1e-9),
        0.0,
        1e-16,
    );
}

#[test]
fn wrms_norm_is_positive_for_nonzero_error() {
    assert!(wrms_norm(&[1e-6, -2e-6], &[1.0, 1.0], &[1.0, 1.0], 1e-6, 1e-9) > 0.0);
}

#[test]
fn propose_step_accepts_small_error_and_grows_h() {
    let (h_new, accepted) = propose_step(1e-3, 1e-2, 3);
    assert!(accepted);
    assert!(h_new > 1e-3);
}

#[test]
fn propose_step_rejects_large_error_and_shrinks_h() {
    let (h_new, accepted) = propose_step(1e-3, 10.0, 3);
    assert!(!accepted);
    assert!(h_new < 1e-3);
}

#[test]
fn propose_step_caps_growth() {
    let (h_new, accepted) = propose_step(1e-3, 0.0, 3);
    assert!(accepted);
    // capped at MAX_FACTOR * h
    assert!(h_new <= 1e-3 * 8.0 + 1e-15);
    assert!(h_new > 1e-3);
}

#[test]
fn propose_step_error_exactly_one_is_accepted() {
    let (_, accepted) = propose_step(1e-3, 1.0, 3);
    assert!(accepted);
}
