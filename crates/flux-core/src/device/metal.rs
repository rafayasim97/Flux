#[cfg(target_os = "macos")]
use super::{Device, DeviceBuffer, DeviceKind};
#[cfg(target_os = "macos")]
use crate::{FluxError, Result};
#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
pub struct MetalDevice {
    _device: metal::Device,
}

#[cfg(target_os = "macos")]
impl MetalDevice {
    pub fn try_new() -> Result<Option<Self>> {
        let dev = metal::Device::system_default()
            .or_else(|| metal::Device::all().into_iter().next());
        let Some(d) = dev else {
            return Ok(None);
        };
        Ok(Some(Self { _device: d }))
    }
}

#[cfg(target_os = "macos")]
impl Device for MetalDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        let buf = self
            ._device
            .new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared);
        Ok(DeviceBuffer::new(bytes, Arc::new(buf)))
    }

    fn copy_host_to_device(&self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<()> {
        let Some(b) = dst.downcast_ref::<metal::Buffer>() else {
            return Err(FluxError::Device("metal buffer downcast failed".into()));
        };
        if src.len() != dst.bytes {
            return Err(FluxError::Device("copy size mismatch".into()));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), b.contents() as *mut u8, src.len());
        }
        Ok(())
    }

    fn copy_device_to_host(&self, dst: &mut [u8], src: &DeviceBuffer) -> Result<()> {
        let Some(b) = src.downcast_ref::<metal::Buffer>() else {
            return Err(FluxError::Device("metal buffer downcast failed".into()));
        };
        if dst.len() != src.bytes {
            return Err(FluxError::Device("copy size mismatch".into()));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(b.contents() as *const u8, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}
