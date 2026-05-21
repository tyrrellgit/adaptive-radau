//! Microbenchmarks for tableau construction.
//!
//! The integrator builds a `RadauTableau` lazily on first use of each order
//! (cached via `TableauCache`). The cost is dominated by:
//!   - root-finding for the Radau polynomial (bracket + bisection + Newton),
//!   - the Vandermonde solve that produces the A matrix, and
//!   - the Schur decomposition that extracts u1 = real eigenvalue of A^{-1}.
//!
//! These benches show the cold-start cost per order. They are not on any
//! steady-state hot path, but spikes here would matter if the integrator is
//! frequently restarted with a fresh `TableauCache`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use radau_rs::RadauTableau;

fn bench_tableau_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("tableau/build");

    for &order in &[5usize, 9, 13] {
        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, &order| {
            b.iter(|| RadauTableau::for_order(black_box(order)).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tableau_build);
criterion_main!(benches);