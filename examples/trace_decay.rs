use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

struct Decay;
impl OdeProblem for Decay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[0];
    }
}

fn main() {
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

    let t_end = 1.0;
    let max_steps = 200;
    for step_i in 0..max_steps {
        if solver.t >= t_end { break; }
        let h_before = solver.h;
        let t_before = solver.t;
        let outcome = solver.step(&Decay).unwrap();
        eprintln!(
            "step {:>3}: t {:.6e} -> {:.6e}, h {:.3e} (new {:.3e}), order={}, iters={}, accepted={}",
            step_i, t_before, solver.t, h_before, solver.h,
            solver.order_control.current_order, outcome.newton_iters, outcome.accepted,
        );
        if step_i > 150 {
            eprintln!("=== too many steps, stopping ===");
            break;
        }
    }
    println!("final t={}, y={}, target exp(-1)={}", solver.t, solver.y[0], (-1.0_f64).exp());
}
