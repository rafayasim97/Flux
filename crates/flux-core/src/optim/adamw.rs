use crate::{DType, FluxError, Result, Tensor};
use std::collections::HashMap;

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step: usize,
    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &mut [Tensor]) -> Result<()> {
        self.step += 1;
        let t = self.step as f32;

        for p in params.iter_mut() {
            if p.dtype != DType::F32 || p.device.kind() != crate::device::DeviceKind::Cpu {
                return Err(FluxError::DType("AdamW: only CPU f32 for now".into()));
            }
            if !p.requires_grad {
                continue;
            }
            let gbuf = p
                .grad
                .as_ref()
                .ok_or_else(|| FluxError::Autograd("AdamW: param has no grad".into()))?
                .clone();

            let mut p_bytes = vec![0u8; p.shape.numel() * 4];
            let mut g_bytes = vec![0u8; p.shape.numel() * 4];
            p.device.copy_device_to_host(&mut p_bytes, &p.buf)?;
            p.device.copy_device_to_host(&mut g_bytes, &gbuf)?;

            let mut pv = bytemuck::cast_slice::<u8, f32>(&p_bytes).to_vec();
            let gv = bytemuck::cast_slice::<u8, f32>(&g_bytes);

            let key = p.buf.inner.as_ref() as *const _ as *const () as usize;
            let m = self.m.entry(key).or_insert_with(|| vec![0.0; pv.len()]);
            let v = self.v.entry(key).or_insert_with(|| vec![0.0; pv.len()]);

            let b1 = self.beta1;
            let b2 = self.beta2;

            for i in 0..pv.len() {
                let grad = gv[i] + self.weight_decay * pv[i];
                m[i] = b1 * m[i] + (1.0 - b1) * grad;
                v[i] = b2 * v[i] + (1.0 - b2) * grad * grad;

                let mhat = m[i] / (1.0 - b1.powf(t));
                let vhat = v[i] / (1.0 - b2.powf(t));

                pv[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
            }

            let bytes: &[u8] = bytemuck::cast_slice(&pv);
            p.device.copy_host_to_device(&mut p.buf, bytes)?;
        }
        Ok(())
    }
}
