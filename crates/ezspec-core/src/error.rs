use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CoreError {
    /// Input arrays have inconsistent lengths.
    LengthMismatch { expected: usize, got: usize },
    /// Too few points for the requested operation.
    TooFewPoints { needed: usize, got: usize },
    /// Matrix of the Whittaker system is not positive definite (e.g. fewer
    /// points with non-zero weight than the difference order).
    NotPositiveDefinite { row: usize },
    /// x values must be strictly increasing.
    NotIncreasing,
    /// A parameter is outside its admissible range.
    InvalidParameter(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            CoreError::TooFewPoints { needed, got } => {
                write!(f, "too few points: need at least {needed}, got {got}")
            }
            CoreError::NotPositiveDefinite { row } => write!(
                f,
                "linear system is not positive definite (row {row}); \
                 too few points with non-zero weight for this difference order?"
            ),
            CoreError::NotIncreasing => write!(f, "x values must be strictly increasing"),
            CoreError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for CoreError {}

pub(crate) fn check_len(expected: usize, got: usize) -> Result<(), CoreError> {
    if expected != got {
        Err(CoreError::LengthMismatch { expected, got })
    } else {
        Ok(())
    }
}

pub(crate) fn check_increasing(x: &[f64]) -> Result<(), CoreError> {
    if x.windows(2).all(|w| w[1] > w[0]) {
        Ok(())
    } else {
        Err(CoreError::NotIncreasing)
    }
}
