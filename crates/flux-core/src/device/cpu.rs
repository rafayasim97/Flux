use super::{Device, DeviceBuffer, DeviceKind};
use crate::{FluxError, Result};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Default)]
pub struct CpuDevice;

#[derive(Default)]
struct CpuBuf {
    data: Mutex<Vec<u8>>,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self
    }
}

impl Device for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        let buf = CpuBuf {
            data: Mutex::new(vec![0u8; bytes]),
        };
        Ok(DeviceBuffer::new(bytes, Arc::new(buf)))
    }

    fn copy_host_to_device(&self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<()> {
        let Some(b) = dst.downcast_ref::<CpuBuf>() else {
            return Err(FluxError::Device("cpu buffer downcast failed".into()));
        };
        let mut lock = b.data.lock();
        if lock.len() != src.len() {
            return Err(FluxError::Device("copy size mismatch".into()));
        }
        lock.copy_from_slice(src);
        Ok(())
    }

    fn copy_device_to_host(&self, dst: &mut [u8], src: &DeviceBuffer) -> Result<()> {
        let Some(b) = src.downcast_ref::<CpuBuf>() else {
            return Err(FluxError::Device("cpu buffer downcast failed".into()));
        };
        let lock = b.data.lock();
        if lock.len() != dst.len() {
            return Err(FluxError::Device("copy size mismatch".into()));
        }
        dst.copy_from_slice(&lock);
        Ok(())
    }
}
