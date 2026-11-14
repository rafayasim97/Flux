use crate::autograd::tape::{NodeId, Tape};
use crate::device::{Device, DeviceBuffer};
use crate::{DType, FluxError, Result, Shape};
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor {
    pub shape: Shape,
    pub dtype: DType,
    pub device: Arc<dyn Device>,
    pub buf: DeviceBuffer,

    pub requires_grad: bool,
    pub grad: Option<DeviceBuffer>,
    pub node: Option<NodeId>,
}

impl Tensor {
    pub fn zeros(shape: Shape, dtype: DType, device: Arc<dyn Device>) -> Result<Self> {
        let bytes = shape.numel() * dtype.size_bytes();
        let buf = device.alloc(bytes)?;
        Ok(Self {
            shape,
            dtype,
            device,
            buf,
            requires_grad: false,
            grad: None,
            node: None,
        })
    }

    pub fn from_f32(shape: Shape, device: Arc<dyn Device>, data: &[f32]) -> Result<Self> {
        if shape.numel() != data.len() {
            return Err(FluxError::Shape("from_f32: numel mismatch".into()));
        }
        let mut t = Self::zeros(shape, DType::F32, device)?;
        let bytes: &[u8] = bytemuck::cast_slice(data);
        t.device.copy_host_to_device(&mut t.buf, bytes)?;
        Ok(t)
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(FluxError::DType(
                "to_vec_f32 only supports f32 right now".into(),
            ));
        }
        let mut out = vec![0u8; self.shape.numel() * 4];
        self.device.copy_device_to_host(&mut out, &self.buf)?;
        Ok(bytemuck::cast_slice::<u8, f32>(&out).to_vec())
    }

    pub fn with_grad(mut self, tape: &mut Tape) -> Self {
        self.requires_grad = true;
        let id = tape.register_leaf(self.clone());
        self.node = Some(id);
        self
    }

    pub fn ensure_grad_buffer(&mut self) -> Result<()> {
        if !self.requires_grad {
            return Ok(());
        }
        if self.grad.is_none() {
            let bytes = self.shape.numel() * self.dtype.size_bytes();
            self.grad = Some(self.device.alloc(bytes)?);
        }
        Ok(())
    }
}
