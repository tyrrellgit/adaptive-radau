use std::time::Instant;
use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

struct Robertson;

impl OdeProblem for Robertson {
    fn dim(&self) -> usize { 3 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
        dydt[2] =  3.0e7 * y[1] * y[1];
    }
}

fn main() {
    let problem = Robertson;
    let mut solver = RadauIntegrator::new(
        0.0,
        vec![1.0, 0.0, 0.0],
        IntegratorOptions {
            initial_step:    1e-4,
            min_step:        1e-14,
            max_step:        1e6,
            rtol:            1e-6,
            atol:            1e-10,
            max_newton_iter: 10,
            ..Default::default()
        },
    );

    let output_times: Vec<f64> = ((-4_i32)..=5).map(|e| 10f64.powi(e)).collect();
    let mut out_idx = 0;
    let mut steps = 0usize;
    let mut rejected = 0usize;

    let wall_start = Instant::now();
    let mut last_wall = wall_start;

    println!("{:<12} {:<16} {:<16} {:<16} {:>10} {:>5} {:>5} {:>10}",
             "t", "y1", "y2", "y3", "sum", "ord", "nitr", "wall_ms");
    println!("{}", "-".repeat(95));

    while out_idx < output_times.len() {
        let t_target = output_times[out_idx];
        let h_clamped = solver.h.min(t_target - solver.t);
        if h_clamped <= 0.0 {
            let sum = solver.y[0] + solver.y[1] + solver.y[2];
            let ms = last_wall.elapsed().as_secs_f64() * 1000.0;
            last_wall = Instant::now();
            println!("{:<12.4e} {:<16.6e} {:<16.6e} {:<16.6e} {:>10.8} {:>5} {:>5} {:>9.2}ms",
                     solver.t, solver.y[0], solver.y[1], solver.y[2],
                     sum, solver.order_control.current_order, "-", ms);
            out_idx += 1;
            continue;
        }
        solver.h = h_clamped;

        match solver.step(&problem) {
            Ok(s) => {
                steps += 1;
                if s.t >= t_target {
                    let sum = s.y[0] + s.y[1] + s.y[2];
                    let ms = last_wall.elapsed().as_secs_f64() * 1000.0;
                    last_wall = Instant::now();
                    println!("{:<12.4e} {:<16.6e} {:<16.6e} {:<16.6e} {:>10.8} {:>5} {:>5} {:>9.2}ms",
                             s.t, s.y[0], s.y[1], s.y[2],
                             sum, solver.order_control.current_order,
                             s.newton_iters, ms);
                    out_idx += 1;
                }
            }
            Err(e) => {
                rejected += 1;
                if rejected > 1000 {
                    eprintln!("Too many failures at t={:.4e} h={:.4e}: {}", solver.t, solver.h, e);
                    break;
                }
            }
        }
    }

    let total_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    println!("\nDone: {} accepted steps, {} rejections, {:.2}ms total.", steps, rejected, total_ms);
}