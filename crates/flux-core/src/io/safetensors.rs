use crate::{FluxError, Result};

pub fn load_file(_path: &str) -> Result<()> {
    Err(FluxError::Io(
        "safetensors loader not implemented yet".into(),
    ))
}
