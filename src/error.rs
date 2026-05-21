use thiserror::Error;

#[derive(Debug, Error)]
pub enum RadauError {
    #[error("linear solve failed")]
    LinearSolveFailed,
    #[error("newton iteration failed to converge")]
    NewtonFailed,
    #[error("invalid order {0}; supported orders are 5, 9, 13, 17, 21, 25")]
    InvalidOrder(usize),
    #[error("step size underflow")]
    StepUnderflow,
    #[error("dimension mismatch")]
    DimensionMismatch,
    #[error("tableau generation is not implemented for order {0}")]
    TableauNotImplemented(usize),
}

pub type Result<T> = std::result::Result<T, RadauError>;
