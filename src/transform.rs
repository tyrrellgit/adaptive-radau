use nalgebra::DMatrix;

#[derive(Debug, Clone)]
pub struct BlockDiagTransform {
    pub t_mat: DMatrix<f64>,
    pub t_inv: DMatrix<f64>,
    pub lambda_real: f64,
    pub complex_pairs: Vec<(f64, f64)>,
}

impl BlockDiagTransform {
    pub fn identity(stages: usize) -> Self {
        Self {
            t_mat: DMatrix::identity(stages, stages),
            t_inv: DMatrix::identity(stages, stages),
            lambda_real: 1.0,
            complex_pairs: Vec::new(),
        }
    }
}