use crate::autograd::tape::{NodeId, Tape};
use crate::{DType, FluxError, Result, Tensor};
use std::sync::Arc;

pub fn matmul(tape: &mut Tape, a: &Tensor, b: &Tensor) -> Result<NodeId> {
    if a.device.kind() != crate::device::DeviceKind::Cpu
        || b.device.kind() != crate::device::DeviceKind::Cpu
    {
        return Err(FluxError::Device("matmul only CPU implemented".into()));
    }
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        return Err(FluxError::DType("matmul: f32 only".into()));
    }
    let (m, k1) = a.shape.dim2()?;
    let (k2, n) = b.shape.dim2()?;
    if k1 != k2 {
        return Err(FluxError::Shape("matmul: inner dim mismatch".into()));
    }

    let av = a.to_vec_f32()?;
    let bv = b.to_vec_f32()?;
    let mut out = vec![0f32; m * n];

    for i in 0..m {
        for kk in 0..k1 {
            let aik = av[i * k1 + kk];
            for j in 0..n {
                out[i * n + j] += aik * bv[kk * n + j];
            }
        }
    }

    let mut out_t = Tensor::from_f32(crate::Shape::new(vec![m, n]), a.device.clone(), &out)?;
    out_t.requires_grad = a.requires_grad || b.requires_grad;

    let aid = a
        .node
        .ok_or_else(|| FluxError::Autograd("matmul: a not tracked".into()))?;
    let bid = b
        .node
        .ok_or_else(|| FluxError::Autograd("matmul: b not tracked".into()))?;

    let backward = Arc::new(move |t: &mut Tape, out_id: NodeId| -> Result<()> {
        let out_t = t.tensor(out_id).clone();
        let mut gout_bytes = vec![0u8; out_t.shape.numel() * 4];
        out_t.device.copy_device_to_host(
            &mut gout_bytes,
            out_t
                .grad
                .as_ref()
                .ok_or_else(|| FluxError::Autograd("matmul: no grad out".into()))?,
        )?;
        let gout = bytemuck::cast_slice::<u8, f32>(&gout_bytes);

        let a_cur = t.tensor(aid).clone();
        let b_cur = t.tensor(bid).clone();

        let av = a_cur.to_vec_f32()?;
        let bv = b_cur.to_vec_f32()?;

        let (m, k) = a_cur.shape.dim2()?;
        let (_, n) = b_cur.shape.dim2()?;

        if a_cur.requires_grad {
            let mut ga = vec![0f32; m * k];
            for i in 0..m {
                for kk in 0..k {
                    let mut sum = 0f32;
                    for j in 0..n {
                        sum += gout[i * n + j] * bv[kk * n + j];
                    }
                    ga[i * k + kk] = sum;
                }
            }
            let pa = t.tensor_mut(aid);
            pa.ensure_grad_buffer()?;
            let bytes: &[u8] = bytemuck::cast_slice(&ga);
            pa.device
                .copy_host_to_device(pa.grad.as_mut().unwrap(), bytes)?;
        }

        if b_cur.requires_grad {
            let mut gb = vec![0f32; k * n];
            for kk in 0..k {
                for j in 0..n {
                    let mut sum = 0f32;
                    for i in 0..m {
                        sum += av[i * k + kk] * gout[i * n + j];
                    }
                    gb[kk * n + j] = sum;
                }
            }
            let pb = t.tensor_mut(bid);
            pb.ensure_grad_buffer()?;
            let bytes: &[u8] = bytemuck::cast_slice(&gb);
            pb.device
                .copy_host_to_device(pb.grad.as_mut().unwrap(), bytes)?;
        }

        Ok(())
    });

    let out_id = tape.register_op(out_t.clone(), vec![aid, bid], backward);
    tape.tensor_mut(out_id).node = Some(out_id);
    Ok(out_id)
}
