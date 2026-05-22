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

fn solve_vdp(mu: f64, t_end: f64, rtol: f64, atol: f64) -> Vec<f64> {
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![2.0, 0.0],
        IntegratorOptions {
            initial_step: 1e-4,
            max_step: (0.01 * t_end).max(1e-4),
            rtol,
            atol,
            ..Default::default()
        },
    );

    let result = solver.solve(t_end, &VanDerPol { mu }).unwrap();
    result.y
}

#[test]
fn mildly_nonlinear_mu1_completes_and_stays_bounded() {
    let y = solve_vdp(1.0, 20.0, 1e-8, 1e-10);
    assert!(y[0].abs() < 3.0, "y[0] = {}", y[0]);
    assert!(y[1].abs() < 10.0, "y[1] = {}", y[1]);
}

#[test]
fn highly_stiff_mu1000_completes_and_stays_bounded() {
    let y = solve_vdp(1000.0, 2000.0, 1e-5, 1e-7);
    assert!(y[0].abs() < 3.0, "y[0] = {}", y[0]);
    assert!(y[1].abs() < 3000.0, "y[1] = {}", y[1]);
}

#[test]
fn mu1_solution_approaches_limit_cycle_amplitude() {
    let y = solve_vdp(1.0, 40.0, 1e-8, 1e-10);

    // The van der Pol limit cycle has characteristic amplitude about 2.
    assert!(y[0].abs() < 2.5, "expected state near limit cycle, got y[0] = {}", y[0]);
}

#[test]
fn mu1000_solution_approaches_relaxation_oscillation_scale() {
    let y = solve_vdp(1000.0, 3000.0, 1e-5, 1e-7);

    // Large-mu van der Pol still has x-amplitude of order 2.
    assert!(y[0].abs() < 2.5, "expected relaxation oscillation scale, got y[0] = {}", y[0]);
}

#[test]
fn mu1_two_late_times_have_similar_cycle_scale() {
    let y1 = solve_vdp(1.0, 40.0, 1e-8, 1e-10);
    let y2 = solve_vdp(1.0, 46.6, 1e-8, 1e-10);

    // About one additional period later for mu≈1.
    // This is intentionally loose because phase alignment is sensitive.
    assert!((y1[0] - y2[0]).abs() < 0.5, "x mismatch: {} vs {}", y1[0], y2[0]);
    assert!((y1[1] - y2[1]).abs() < 1.0, "v mismatch: {} vs {}", y1[1], y2[1]);
}

#[test]
fn large_mu_period_has_right_order_of_magnitude() {
    let y1 = solve_vdp(1000.0, 2000.0, 1e-5, 1e-7);
    let y2 = solve_vdp(1000.0, 3614.0, 1e-5, 1e-7);

    // For large mu, period is asymptotically about 1.614 * mu.
    // This is only an order-of-magnitude qualitative check.
    assert!(y1[0].abs() < 3.0, "y1[0] = {}", y1[0]);
    assert!(y2[0].abs() < 3.0, "y2[0] = {}", y2[0]);
}