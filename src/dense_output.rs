#[derive(Debug, Clone)]
pub struct DenseOutput {
    pub t_n: f64,
    pub h: f64,
    pub y_n: Vec<f64>,
    pub stage_vals: Vec<Vec<f64>>,
    pub c: Vec<f64>,
}

impl DenseOutput {
    pub fn evaluate(&self, t: f64) -> Vec<f64> {
        let theta = (t - self.t_n) / self.h;
        let n = self.y_n.len();
        let mut out = vec![0.0; n];

        let l0 = lagrange_basis_zero(theta, &self.c);
        for j in 0..n {
            out[j] += l0 * self.y_n[j];
        }

        for (i, yi) in self.stage_vals.iter().enumerate() {
            let li = lagrange_basis_stage(theta, i, &self.c);
            for j in 0..n {
                out[j] += li * yi[j];
            }
        }
        out
    }
}

fn lagrange_basis_zero(theta: f64, c: &[f64]) -> f64 {
    c.iter().fold(1.0, |acc, &ci| acc * (theta - ci) / (0.0 - ci))
}

fn lagrange_basis_stage(theta: f64, i: usize, c: &[f64]) -> f64 {
    let ci = c[i];
    let mut acc = theta / ci;
    for (j, &cj) in c.iter().enumerate() {
        if j != i {
            acc *= (theta - cj) / (ci - cj);
        }
    }
    acc
}
