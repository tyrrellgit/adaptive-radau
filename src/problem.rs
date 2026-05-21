use nalgebra::DMatrix;

pub trait OdeProblem {
    fn dim(&self) -> usize;
    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]);

    fn jacobian(&self, _t: f64, _y: &[f64], _jac: &mut DMatrix<f64>) -> bool {
        false
    }
}
