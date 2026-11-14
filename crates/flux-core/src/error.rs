use thiserror::Error;

pub type Result<T> = std::result::Result<T, FluxError>;

#[derive(Error, Debug)]
pub enum FluxError {
    #[error("shape error: {0}")]
    Shape(String),

    #[error("dtype error: {0}")]
    DType(String),

    #[error("device error: {0}")]
    Device(String),

    #[error("autograd error: {0}")]
    Autograd(String),

    #[error("io error: {0}")]
    Io(String),
}
