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

struct Robertson;

// y' = [
//   -0.04 y0 + 1e4 y1 y2,
//    0.04 y0 - 1e4 y1 y2 - 3e7 y1^2,
//    3e7 y1^2
// ]
impl OdeProblem for Robertson {
    fn dim(&self) -> usize { 3 }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] =  3e7 * y[1] * y[1];
    }
}

fn solve_robertson(t_end: f64, rtol: f64, atol: f64) -> Vec<f64> {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 0.0, 0.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: (0.1 * t_end).max(1e-6),
            rtol,
            atol,
            ..Default::default()
        },
    );

    let result = solver.solve(t_end, &Robertson).unwrap();
    result.y
}

fn assert_physical_state(y: &[f64]) {
    assert!(y.len() == 3);
    assert!(y[0] >= -1e-12, "y0 negative: {}", y[0]);
    assert!(y[1] >= -1e-12, "y1 negative: {}", y[1]);
    assert!(y[2] >= -1e-12, "y2 negative: {}", y[2]);

    let sum = y[0] + y[1] + y[2];
    assert_close(sum, 1.0, 1e-12, 1e-9);
}

#[test]
fn conservation_holds_at_t1e4() {
    let y = solve_robertson(1e4, 1e-8, 1e-11);
    let sum = y[0] + y[1] + y[2];
    assert_close(sum, 1.0, 1e-12, 1e-9);
}

#[test]
fn solution_remains_physical_at_characteristic_times() {
    let times = [1e-6_f64, 1e-2, 1.0, 1e4, 1e9];

    for &t in &times {
        let y = solve_robertson(t, 1e-8, 1e-11);
        assert_physical_state(&y);
    }
}

#[test]
fn robertson_early_time_is_still_mostly_in_y0() {
    let y = solve_robertson(1e-6, 1e-8, 1e-11);
    assert_physical_state(&y);
    assert!(y[0] > 0.999, "expected y0 near 1 early, got {}", y[0]);
    assert!(y[2] < 1e-5, "expected y2 tiny early, got {}", y[2]);
}

#[test]
fn robertson_mid_time_creates_product_species() {
    let y_early = solve_robertson(1e-6, 1e-8, 1e-11);
    let y_mid = solve_robertson(1e-2, 1e-8, 1e-11);
    assert_physical_state(&y_mid);
    assert!(y_mid[2] > y_early[2], "expected y2 growth by t=1e-2");
    assert!(y_mid[0] < y_early[0], "expected y0 decay by t=1e-2");
    assert!(y_mid[1] < 1e-2, "expected y1 still small at t=1e-2, got {}", y_mid[1]);
}

#[test]
fn robertson_late_time_is_physically_plausible() {
    let y = solve_robertson(1e4, 1e-8, 1e-11);
    assert_physical_state(&y);

    assert!(y[2] >= y[0], "expected y2 to dominate y0, got y2={} y0={}", y[2], y[0]);
    assert!(y[2] >= y[1], "expected y2 to dominate y1, got y2={} y1={}", y[2], y[1]);
    assert!(y[1] < 1e-4, "expected y1 tiny at t=1e4, got {}", y[1]);
}

#[test]
fn y2_increases_across_characteristic_times() {
    let times = [1e-6_f64, 1e-4, 1e-2, 1.0, 1e4];
    let mut prev_y2 = -1.0_f64;

    for &t in &times {
        let y = solve_robertson(t, 1e-8, 1e-11);
        assert!(
            y[2] >= prev_y2 - 1e-12,
            "y2 not monotone at t={}: previous {}, current {}",
            t,
            prev_y2,
            y[2]
        );
        prev_y2 = y[2];
    }
}

#[test]
fn y1_remains_small_at_late_times() {
    let checkpoints = [1.0_f64, 1e2, 1e4, 1e9];

    for &t in &checkpoints {
        let y = solve_robertson(t, 1e-8, 1e-11);
        assert!(
            y[1] < 1e-4,
            "expected y1 to be small at t={}, got {}",
            t,
            y[1]
        );
    }
}

/// Late-time Robertson decays with `y0 ~ 2.0833e-8 * (1e11 / t)`, so the
/// standard published reference at `t = 1e11` is
/// `y = (2.0839e-8, 8.3359e-14, 1.0)`. The previous form of this test asserted
/// `y0 < 1e-8`, which the reference value itself violates — it only passed
/// while the error estimator was mis-scaled and the solver was overshooting.
///
/// Checked at `t = 1e9` (reference `y0 ~ 2.0833e-6`): the physics is the same
/// and the run is affordable. See `robertson_late_time_step_growth` for why
/// `t = 1e11` is currently impractical.
#[test]
fn final_state_approaches_published_reference_at_t1e9() {
    let y = solve_robertson(1e9, 1e-8, 1e-11);

    assert_physical_state(&y);

    assert!(y[2] > 0.9999, "expected y2 very near 1 at t=1e9, got {}", y[2]);
    assert_close(y[0], 2.0833e-6, 0.0, 1e-3);
    assert!(y[1] < 1e-10, "expected y1 tiny at t=1e9, got {}", y[1]);
}

#[test]
fn tighter_tolerance_still_preserves_conservation() {
    let y = solve_robertson(1e4, 1e-10, 1e-13);
    let sum = y[0] + y[1] + y[2];
    assert_close(sum, 1.0, 1e-13, 1e-10);
}

/// Two tolerances must agree to roughly the looser one. `y[1]` is ~4.8e-7 at
/// t=1e4, so with `atol = 1e-11` its error is atol-dominated: the achievable
/// agreement is `atol/y1 ~ 2e-5` relative, not `rtol`. The bound below
/// reflects that rather than the tighter `rtol` figure.
#[test]
fn medium_and_tight_tolerances_agree_at_t1e4() {
    let y_med = solve_robertson(1e4, 1e-8, 1e-11);
    let y_tight = solve_robertson(1e4, 1e-10, 1e-13);

    assert_close(y_med[0], y_tight[0], 1e-10, 1e-5);
    assert_close(y_med[1], y_tight[1], 1e-12, 1e-4);
    assert_close(y_med[2], y_tight[2], 1e-10, 1e-5);
}

/// Guards the step-size behaviour that three separate bugs on this branch were
/// hiding. Robertson is a conservation law, so `J` is singular and the ESTRAD
/// smoothing matrix `E1 = (u1/h)I - J` becomes ill-conditioned as `h` grows;
/// any residual left by an under-converged Newton solve is amplified into the
/// slow direction and caps the step size. If this regresses, step counts
/// explode by orders of magnitude rather than the answer going wrong.
#[test]
fn late_time_integration_step_count_stays_bounded() {
    let mut solver = radau_rs::RadauIntegrator::new(
        0.0,
        vec![1.0, 0.0, 0.0],
        IntegratorOptions {
            initial_step: 1e-6,
            max_step: 1e8,
            rtol: 1e-8,
            atol: 1e-11,
            ..Default::default()
        },
    );
    let r = solver.solve(1e9, &Robertson).unwrap();
    assert!(
        r.steps_accepted < 20_000,
        "step count regressed: {} accepted, {} rejected",
        r.steps_accepted, r.steps_rejected
    );
    assert_close(r.y[0], 2.0833e-6, 0.0, 1e-3);
}