use crate::{FluxError, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn assert_rank(&self, r: usize) -> Result<()> {
        if self.rank() != r {
            return Err(FluxError::Shape(format!(
                "expected rank {r}, got {}",
                self.rank()
            )));
        }
        Ok(())
    }

    pub fn dim2(&self) -> Result<(usize, usize)> {
        self.assert_rank(2)?;
        Ok((self.dims[0], self.dims[1]))
    }
}
