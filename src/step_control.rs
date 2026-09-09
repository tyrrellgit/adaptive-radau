/// Weighted root-mean-square norm used by the PI step-size controller.
pub fn wrms_norm(err: &[f64], y: &[f64], y_new: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = err.len().max(1) as f64;
    let sum = err
        .iter()
        .zip(y.iter())
        .zip(y_new.iter())
        .map(|((e, yi), yi_new)| {
            let sc = atol + rtol * yi.abs().max(yi_new.abs());
            (e / sc).powi(2)
        })
        .sum::<f64>();
    (sum / n).sqrt()
}

/// Standard Hairer I-controller (Solving ODEs II, p.124).
///
/// `embedded_order` is the order of the embedded method (`p_hat`). For a
/// classical p-th order main method with order p_hat embedded, the error is
/// estimated to be O(h^{p_hat+1}) and we pick the step from
///     h_new = h · safety · err_norm^{-1/(p_hat + 1)}
/// clamped to `[min_factor, max_factor]`.
pub fn propose_step(h: f64, err_norm: f64, embedded_order: usize) -> (f64, bool) {
    const SAFETY: f64 = 0.9;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 8.0;

    let exponent = 1.0 / (embedded_order as f64 + 1.0);

    let accepted = err_norm <= 1.0;

    let factor = if err_norm > 0.0 {
        let raw = SAFETY * err_norm.powf(-exponent);
        // When rejecting, do not grow the step.
        let upper = if accepted { MAX_FACTOR } else { 1.0 };
        raw.clamp(MIN_FACTOR, upper)
    } else {
        MAX_FACTOR
    };

    (h * factor, accepted)
}
