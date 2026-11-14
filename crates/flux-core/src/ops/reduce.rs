use crate::autograd::tape::{NodeId, Tape};
use crate::{DType, FluxError, Result, Tensor};
use std::sync::Arc;

pub fn mse(tape: &mut Tape, pred: &Tensor, target: &Tensor) -> Result<NodeId> {
    if pred.device.kind() != crate::device::DeviceKind::Cpu
        || target.device.kind() != crate::device::DeviceKind::Cpu
    {
        return Err(FluxError::Device("mse only CPU implemented".into()));
    }
    if pred.dtype != DType::F32 || target.dtype != DType::F32 {
        return Err(FluxError::DType("mse: f32 only".into()));
    }
    if pred.shape != target.shape {
        return Err(FluxError::Shape("mse: shape mismatch".into()));
    }

    let pv = pred.to_vec_f32()?;
    let tv = target.to_vec_f32()?;
    let n = pv.len() as f32;
    let tv_saved = tv.clone();

    let mut sum = 0f32;
    for i in 0..pv.len() {
        let d = pv[i] - tv[i];
        sum += d * d;
    }
    let loss = sum / n;

    let mut out = Tensor::from_f32(crate::Shape::new(vec![1]), pred.device.clone(), &[loss])?;
    out.requires_grad = pred.requires_grad;

    let pid = pred
        .node
        .ok_or_else(|| FluxError::Autograd("mse: pred not tracked".into()))?;

    let backward = Arc::new(move |t: &mut Tape, _out_id: NodeId| -> Result<()> {
        let pred_t = t.tensor(pid).clone();
        let pv = pred_t.to_vec_f32()?;
        let n = pv.len() as f32;

        let mut grad = vec![0f32; pv.len()];
        for i in 0..pv.len() {
            grad[i] = 2.0 * (pv[i] - tv_saved[i]) / n;
        }
        let pa = t.tensor_mut(pid);
        if pa.requires_grad {
            pa.ensure_grad_buffer()?;
            let bytes: &[u8] = bytemuck::cast_slice(&grad);
            pa.device
                .copy_host_to_device(pa.grad.as_mut().unwrap(), bytes)?;
        }
        Ok(())
    });

    let out_id = tape.register_op(out.clone(), vec![pid], backward);
    tape.tensor_mut(out_id).node = Some(out_id);
    Ok(out_id)
}
