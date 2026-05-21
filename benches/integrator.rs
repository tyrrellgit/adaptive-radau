//! Criterion benchmarks for the adaptive Radau IIA integrator.
//!
//! Runs a curated set of stiff and non-stiff test problems at fixed orders
//! (5, 9, 13) and at full adaptive order (5..=13). Each benchmark integrates
//! end-to-end; Criterion handles warm-up, sample sizing, and statistical
//! reporting.
//!
//! Run with:
//!   cargo bench --bench integrator
//! Filter to a single group:
//!   cargo bench --bench integrator -- decay/non_stiff
//! Generate HTML reports (default with the `html_reports` feature):
//!   open target/criterion/report/index.html

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use radau_rs::{IntegratorOptions, OdeProblem, RadauIntegrator};

// --- Test problems --------------------------------------------------------

struct Decay {
    lambda: f64,
}
impl OdeProblem for Decay {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = self.lambda * y[0];
    }
}

/// Robertson stiff chemical kinetics — the canonical stiff test problem.
struct Robertson;
impl OdeProblem for Robertson {
    fn dim(&self) -> usize {
        3
    }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] = 3e7 * y[1] * y[1];
    }
}

/// Van der Pol oscillator. `mu` controls stiffness (mu=1 mild, mu=1000 highly stiff).
struct VanDerPol {
    mu: f64,
}
impl OdeProblem for VanDerPol {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = y[1];
        dydt[1] = self.mu * ((1.0 - y[0] * y[0]) * y[1] - y[0]);
    }
}

// --- Bench helpers --------------------------------------------------------

fn fixed_order_options(order: usize, rtol: f64, atol: f64, max_step: f64) -> IntegratorOptions {
    IntegratorOptions {
        rtol,
        atol,
        initial_step: 1e-6,
        max_step,
        initial_order: order,
        min_order: order,
        max_order: order,
        ..Default::default()
    }
}

fn adaptive_options(rtol: f64, atol: f64, max_step: f64) -> IntegratorOptions {
    IntegratorOptions {
        rtol,
        atol,
        initial_step: 1e-6,
        max_step,
        initial_order: 5,
        min_order: 5,
        max_order: 13,
        ..Default::default()
    }
}

fn solve<P: OdeProblem>(
    t0: f64,
    y0: &[f64],
    t_end: f64,
    opts: IntegratorOptions,
    problem: &P,
) -> radau_rs::SolveResult {
    let mut solver = RadauIntegrator::new(t0, y0.to_vec(), opts);
    solver.solve(t_end, problem).expect("solve failed")
}

// --- Decay benchmarks -----------------------------------------------------

fn bench_decay_non_stiff(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay/non_stiff");
    group.measurement_time(Duration::from_secs(3));

    let problem = Decay { lambda: -1.0 };
    let y0 = vec![1.0];

    for &order in &[5usize, 9, 13] {
        group.bench_with_input(
            BenchmarkId::new("fixed_order", order),
            &order,
            |b, &order| {
                b.iter(|| {
                    solve(
                        0.0,
                        &y0,
                        1.0,
                        fixed_order_options(order, 1e-8, 1e-11, 0.5),
                        black_box(&problem),
                    )
                });
            },
        );
    }

    group.bench_function("adaptive", |b| {
        b.iter(|| {
            solve(
                0.0,
                &y0,
                1.0,
                adaptive_options(1e-8, 1e-11, 0.5),
                black_box(&problem),
            )
        });
    });

    group.finish();
}

fn bench_decay_stiff(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay/stiff_lambda_1000");
    group.measurement_time(Duration::from_secs(5));

    let problem = Decay { lambda: -1000.0 };
    let y0 = vec![1.0];

    for &order in &[5usize, 9, 13] {
        group.bench_with_input(
            BenchmarkId::new("fixed_order", order),
            &order,
            |b, &order| {
                b.iter(|| {
                    solve(
                        0.0,
                        &y0,
                        1.0,
                        fixed_order_options(order, 1e-8, 1e-11, 0.5),
                        black_box(&problem),
                    )
                });
            },
        );
    }

    group.bench_function("adaptive", |b| {
        b.iter(|| {
            solve(
                0.0,
                &y0,
                1.0,
                adaptive_options(1e-8, 1e-11, 0.5),
                black_box(&problem),
            )
        });
    });

    group.finish();
}

// --- Robertson benchmarks -------------------------------------------------

fn bench_robertson(c: &mut Criterion) {
    let mut group = c.benchmark_group("robertson");
    group.measurement_time(Duration::from_secs(5));

    let problem = Robertson;
    let y0 = vec![1.0, 0.0, 0.0];
    let t_end = 1e4_f64;

    // Throughput per simulated second so Criterion's reports show
    // "simulated time integrated per wall-clock second".
    group.throughput(Throughput::Elements(t_end as u64));

    for &(rtol, atol, label) in &[
        (1e-6_f64, 1e-9_f64, "loose"),
        (1e-8, 1e-11, "medium"),
        (1e-10, 1e-12, "tight"),
    ] {
        for &order in &[5usize, 9, 13] {
            group.bench_with_input(
                BenchmarkId::new(format!("fixed_order_{}", label), order),
                &order,
                |b, &order| {
                    b.iter(|| {
                        solve(
                            0.0,
                            &y0,
                            t_end,
                            fixed_order_options(order, rtol, atol, t_end * 0.1),
                            black_box(&problem),
                        )
                    });
                },
            );
        }

        group.bench_function(format!("adaptive_{}", label), |b| {
            b.iter(|| {
                solve(
                    0.0,
                    &y0,
                    t_end,
                    adaptive_options(rtol, atol, t_end * 0.1),
                    black_box(&problem),
                )
            });
        });
    }

    group.finish();
}

// --- Van der Pol benchmarks -----------------------------------------------

fn bench_van_der_pol(c: &mut Criterion) {
    let mut group = c.benchmark_group("van_der_pol");
    group.measurement_time(Duration::from_secs(8));
    // Van der Pol with mu=1000 is the slowest case; cap sample size so the
    // run still completes in a reasonable wall-clock budget.
    group.sample_size(20);

    for &(mu, t_end, label) in &[(1.0_f64, 20.0_f64, "mild_mu1"), (1000.0, 3000.0, "stiff_mu1000")] {
        let problem = VanDerPol { mu };
        let y0 = vec![2.0, 0.0];

        for &order in &[5usize, 9, 13] {
            group.bench_with_input(
                BenchmarkId::new(format!("fixed_order_{}", label), order),
                &order,
                |b, &order| {
                    b.iter(|| {
                        solve(
                            0.0,
                            &y0,
                            t_end,
                            fixed_order_options(order, 1e-6, 1e-9, t_end),
                            black_box(&problem),
                        )
                    });
                },
            );
        }

        group.bench_function(format!("adaptive_{}", label), |b| {
            b.iter(|| {
                solve(
                    0.0,
                    &y0,
                    t_end,
                    adaptive_options(1e-6, 1e-9, t_end),
                    black_box(&problem),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_decay_non_stiff,
    bench_decay_stiff,
    bench_robertson,
    bench_van_der_pol,
);
criterion_main!(benches);