pub fn wrms_norm(err: &[f64], y: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = err.len().max(1) as f64;
    let sum = err
        .iter()
        .zip(y.iter())
        .map(|(e, yi)| {
            let sc = atol + rtol * yi.abs();
            (e / sc).powi(2)
        })
        .sum::<f64>();
    (sum / n).sqrt()
}

/// Standard Hairer PI controller (Solving ODEs II, p.124).
/// h_new = h * clamp(0.9 * err^(-1/(p+1)), 0.2, 10.0)
pub fn propose_step(h: f64, err_norm: f64, order: usize, _iter: usize) -> (f64, bool) {
    let p = order as f64;
    let factor = if err_norm > 0.0 {
        (0.9 * err_norm.powf(-1.0 / (p + 1.0))).clamp(0.2, 10.0)
    } else {
        10.0
    };
    (h * factor, err_norm <= 1.0)
}