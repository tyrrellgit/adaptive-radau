# Adaptive RADAU IIA Solver

This is a basic implementation of the adaptive order RADAU IIA method presented in this [paper](https://arxiv.org/abs/2412.14362).

We make the slight change of making it an embedded method allowing for adaptive step size control and generating dense output.

## Benchmarks

Criterion benchmarks live under `benches/`:

- `benches/integrator.rs` — end-to-end solves on decay (non-stiff and stiff),
  Robertson, and Van der Pol, at fixed orders 5/9/13 and full adaptive order.
- `benches/tableau.rs` — cold-start tableau-construction microbench.

Run everything:

```bash
cargo bench
```

Run a single group or filter:

```bash
cargo bench --bench integrator -- robertson
cargo bench --bench integrator -- 'decay/non_stiff/adaptive'
cargo bench --bench tableau
```

HTML reports (with the default `html_reports` feature enabled) are written
to `target/criterion/report/index.html`.
