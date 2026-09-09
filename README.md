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

## Linear algebra for the Newton iteration

Each Radau step solves `(I - h A (x) J) dz = -r`. Two strategies are available
via `IntegratorOptions::linear_solver`:

- **Kronecker** — one direct `(s*n) x (s*n)` LU.
- **Block-diagonal (Schur)** — real Schur decomposition `A^-1 = Q T Q^T`
  reduces this to one `n x n` LU plus `(s-1)/2` `2n x 2n` LUs, solved by block
  back-substitution.

`Auto` (the default) picks between them on dimension. Measured on a
method-of-lines heat equation at order 9, block-diagonal versus Kronecker:

| n | 2 | 6 | 10 | 25 | 50 | 100 | 200 |
|---|---|---|----|----|----|-----|-----|
| speedup | 0.86x | 1.11x | 1.40x | 3.36x | 5.78x | 6.88x | 6.85x |

## Other stuff
Below are a few features or ideas being planned for the library, in no particular order of priority:

- IMEX RK methods / operator splitting
- DRIK / ESDIRK methods
- Explicit RK steppers (embedded and dense)
- Krylov‑based method for newton iterations
- Fast jacobians from compile time automatic differentiation (AD)
- General Compatability with compile time AD (differentiable stiff solvers)
- Demonstrations for various common examples 
- Python bindings
