mod cpu;
mod metal;

use crate::{DType, FluxError, Result};
use std::sync::Arc;

pub use cpu::CpuDevice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Metal,
}

pub trait Device: Send + Sync {
    fn kind(&self) -> DeviceKind;

    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer>;
    fn free(&self, _buf: DeviceBuffer) -> Result<()> {
        Ok(())
    }

    fn copy_host_to_device(&self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<()>;
    fn copy_device_to_host(&self, dst: &mut [u8], src: &DeviceBuffer) -> Result<()>;

    fn supports_dtype(&self, _dt: DType) -> bool {
        true
    }
}

#[derive(Clone)]
pub struct DeviceBuffer {
    pub bytes: usize,
    pub inner: Arc<dyn std::any::Any + Send + Sync>,
}

impl DeviceBuffer {
    pub fn new(bytes: usize, inner: Arc<dyn std::any::Any + Send + Sync>) -> Self {
        Self { bytes, inner }
    }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }

    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.inner)?.downcast_mut::<T>()
    }
}

pub fn default_device() -> Arc<dyn Device> {
    Arc::new(CpuDevice::new())
}

pub fn metal_device_if_available() -> Result<Option<Arc<dyn Device>>> {
    #[cfg(target_os = "macos")]
    {
        let d = metal::MetalDevice::try_new()?;
        Ok(d.map(|x| Arc::new(x) as Arc<dyn Device>))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(None)
    }
}

pub fn require_rank2(m: &crate::Shape) -> Result<(usize, usize)> {
    m.dim2().map_err(|e| FluxError::Shape(format!("{e}")))
}
