//! Step-count / timing benchmark across orders.
//!
//! For each test problem and each fixed order, integrate to t_end and report
//! - accepted steps
//! - rejected steps
//! - wall-clock time
//! - max relative error vs reference solution

use std::time::Instant;

use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

struct Decay { lambda: f64 }
impl OdeProblem for Decay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = self.lambda * y[0];
    }
}

struct Robertson;
impl OdeProblem for Robertson {
    fn dim(&self) -> usize { 3 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] =  3e7 * y[1] * y[1];
    }
}

fn bench<P: OdeProblem>(
    name: &str,
    t0: f64,
    y0: Vec<f64>,
    t_end: f64,
    rtol: f64,
    atol: f64,
    fixed_order: usize,
    problem: &P,
    reference: Option<&[f64]>,
) {
    let opts = IntegratorOptions {
        rtol, atol,
        initial_step: 1e-6,
        max_step: t_end * 0.5,
        initial_order: fixed_order,
        min_order: fixed_order,
        max_order: fixed_order,
        ..Default::default()
    };
    let mut solver = RadauIntegrator::new(t0, y0.clone(), opts);
    let tic = Instant::now();
    let result = solver.solve(t_end, problem).expect("solve failed");
    let elapsed = tic.elapsed();

    let err_str = match reference {
        Some(r) => {
            let max_rel: f64 = result.y.iter().zip(r.iter())
                .map(|(a, b)| (a - b).abs() / (b.abs().max(1e-12)))
                .fold(0.0_f64, f64::max);
            format!("max-rel-err={:.2e}", max_rel)
        }
        None => "".to_string(),
    };

    println!(
        "{:18}  order={:2}  accepted={:5}  rejected={:4}  time={:>7.2}ms  {}",
        name, fixed_order, result.steps_accepted, result.steps_rejected,
        elapsed.as_secs_f64() * 1e3, err_str,
    );
}

fn main() {
    println!("\n=== decay y'=-y, t in [0,1], rtol=1e-8 atol=1e-11 ===");
    let dref = vec![(-1.0_f64).exp()];
    for o in [5, 9, 13] {
        bench("decay-lambda-1", 0.0, vec![1.0], 1.0, 1e-8, 1e-11, o, &Decay{lambda:-1.0}, Some(&dref));
    }

    println!("\n=== stiff decay y'=-1000y, t in [0,1], rtol=1e-8 atol=1e-11 ===");
    let sref = vec![(-1000.0_f64).exp()];
    for o in [5, 9, 13] {
        bench("decay-lambda-1000", 0.0, vec![1.0], 1.0, 1e-8, 1e-11, o, &Decay{lambda:-1000.0}, Some(&sref));
    }

    println!("\n=== Robertson, t in [0,1e4], rtol=1e-6 atol=1e-9 ===");
    for o in [5, 9, 13] {
        bench("robertson-1e4", 0.0, vec![1.0,0.0,0.0], 1e4, 1e-6, 1e-9, o, &Robertson, None);
    }

    println!("\n=== Robertson, t in [0,1e4], rtol=1e-10 atol=1e-12 (tight) ===");
    for o in [5, 9, 13] {
        bench("robertson-tight", 0.0, vec![1.0,0.0,0.0], 1e4, 1e-10, 1e-12, o, &Robertson, None);
    }

    println!("\n=== Adaptive order (5..13), Robertson t in [0,1e4], rtol=1e-8 atol=1e-11 ===");
    let opts = IntegratorOptions {
        rtol: 1e-8, atol: 1e-11,
        initial_step: 1e-6,
        max_step: 1e3,
        initial_order: 5,
        min_order: 5,
        max_order: 13,
        ..Default::default()
    };
    let mut solver = RadauIntegrator::new(0.0, vec![1.0,0.0,0.0], opts);
    let tic = Instant::now();
    let result = solver.solve(1e4, &Robertson).unwrap();
    let elapsed = tic.elapsed();
    println!("adaptive            order=A   accepted={:5}  rejected={:4}  time={:>7.2}ms",
        result.steps_accepted, result.steps_rejected, elapsed.as_secs_f64() * 1e3);
}
