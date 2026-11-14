use crate::autograd::tape::{NodeId, Tape};
use crate::{DType, FluxError, Result, Tensor};
use std::sync::Arc;

fn cpu_only(a: &Tensor) -> Result<()> {
    if a.device.kind() != crate::device::DeviceKind::Cpu {
        return Err(FluxError::Device("elemwise only CPU implemented".into()));
    }
    Ok(())
}

pub fn add(tape: &mut Tape, a: &Tensor, b: &Tensor) -> Result<NodeId> {
    cpu_only(a)?;
    cpu_only(b)?;
    if a.shape != b.shape || a.dtype != DType::F32 || b.dtype != DType::F32 {
        return Err(FluxError::Shape("add: shape or dtype mismatch".into()));
    }

    let av = a.to_vec_f32()?;
    let bv = b.to_vec_f32()?;
    let mut outv = vec![0f32; av.len()];
    for i in 0..av.len() {
        outv[i] = av[i] + bv[i];
    }

    let mut out = Tensor::from_f32(a.shape.clone(), a.device.clone(), &outv)?;
    out.requires_grad = a.requires_grad || b.requires_grad;

    let aid = a
        .node
        .ok_or_else(|| FluxError::Autograd("add: a not tracked".into()))?;
    let bid = b
        .node
        .ok_or_else(|| FluxError::Autograd("add: b not tracked".into()))?;

    let backward = Arc::new(move |t: &mut Tape, out_id: NodeId| -> Result<()> {
        let grad_out = t
            .tensor(out_id)
            .grad
            .as_ref()
            .ok_or_else(|| FluxError::Autograd("no grad for out".into()))?
            .clone();
        {
            let pa = t.tensor_mut(aid);
            if pa.requires_grad {
                pa.ensure_grad_buffer()?;
                add_inplace(pa, &grad_out)?;
            }
        }
        {
            let pb = t.tensor_mut(bid);
            if pb.requires_grad {
                pb.ensure_grad_buffer()?;
                add_inplace(pb, &grad_out)?;
            }
        }
        Ok(())
    });

    let out_id = tape.register_op(out.clone(), vec![aid, bid], backward);
    tape.tensor_mut(out_id).node = Some(out_id);
    Ok(out_id)
}

fn add_inplace(dst: &mut Tensor, src_grad: &crate::device::DeviceBuffer) -> Result<()> {
    let mut dst_bytes = vec![0u8; dst.shape.numel() * 4];
    let mut src_bytes = vec![0u8; dst.shape.numel() * 4];
    dst.device
        .copy_device_to_host(&mut dst_bytes, dst.grad.as_ref().unwrap())?;
    dst.device.copy_device_to_host(&mut src_bytes, src_grad)?;
    let mut dv = bytemuck::cast_slice::<u8, f32>(&dst_bytes).to_vec();
    let sv = bytemuck::cast_slice::<u8, f32>(&src_bytes);
    for i in 0..dv.len() {
        dv[i] += sv[i];
    }
    let bytes: &[u8] = bytemuck::cast_slice(&dv);
    dst.device
        .copy_host_to_device(dst.grad.as_mut().unwrap(), bytes)?;
    Ok(())
}

pub fn relu(tape: &mut Tape, a: &Tensor) -> Result<NodeId> {
    cpu_only(a)?;
    if a.dtype != DType::F32 {
        return Err(FluxError::DType("relu: f32 only".into()));
    }

    let av = a.to_vec_f32()?;
    let mut outv = vec![0f32; av.len()];
    let mut mask = vec![0u8; av.len()];
    for i in 0..av.len() {
        if av[i] > 0.0 {
            outv[i] = av[i];
            mask[i] = 1;
        }
    }

    let mut out = Tensor::from_f32(a.shape.clone(), a.device.clone(), &outv)?;
    out.requires_grad = a.requires_grad;

    let aid = a
        .node
        .ok_or_else(|| FluxError::Autograd("relu: input not tracked".into()))?;
    let backward = Arc::new(move |t: &mut Tape, out_id: NodeId| -> Result<()> {
        let grad_out_buf = t
            .tensor(out_id)
            .grad
            .as_ref()
            .ok_or_else(|| FluxError::Autograd("relu: no grad".into()))?
            .clone();
        let mut gout_bytes = vec![0u8; t.tensor(out_id).shape.numel() * 4];
        t.tensor(out_id)
            .device
            .copy_device_to_host(&mut gout_bytes, &grad_out_buf)?;
        let gout = bytemuck::cast_slice::<u8, f32>(&gout_bytes);

        let mut gin = vec![0f32; gout.len()];
        for i in 0..gin.len() {
            gin[i] = if mask[i] == 1 { gout[i] } else { 0.0 };
        }

        let pa = t.tensor_mut(aid);
        if pa.requires_grad {
            pa.ensure_grad_buffer()?;
            let bytes: &[u8] = bytemuck::cast_slice(&gin);
            pa.device
                .copy_host_to_device(pa.grad.as_mut().unwrap(), bytes)?;
        }
        Ok(())
    });

    let out_id = tape.register_op(out.clone(), vec![aid], backward);
    tape.tensor_mut(out_id).node = Some(out_id);
    Ok(out_id)
}
