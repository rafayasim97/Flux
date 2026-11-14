use crate::{FluxError, Result, Tensor};
use std::sync::Arc;

pub type NodeId = usize;

pub struct Tape {
    nodes: Vec<Node>,
}

struct Node {
    pub tensor: Tensor,
    pub parents: Vec<NodeId>,
    pub backward: Option<Arc<dyn Fn(&mut Tape, NodeId) -> Result<()> + Send + Sync>>,
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: vec![] }
    }

    pub fn register_leaf(&mut self, t: Tensor) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            tensor: t,
            parents: vec![],
            backward: None,
        });
        id
    }

    pub fn register_op(
        &mut self,
        out: Tensor,
        parents: Vec<NodeId>,
        backward: Arc<dyn Fn(&mut Tape, NodeId) -> Result<()> + Send + Sync>,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            tensor: out,
            parents,
            backward: Some(backward),
        });
        id
    }

    pub fn tensor(&self, id: NodeId) -> &Tensor {
        &self.nodes[id].tensor
    }
    pub fn tensor_mut(&mut self, id: NodeId) -> &mut Tensor {
        &mut self.nodes[id].tensor
    }

    pub fn backward(&mut self, loss_id: NodeId) -> Result<()> {
        self.tensor_mut(loss_id).ensure_grad_buffer()?;
        self.set_grad_scalar_one(loss_id)?;

        for nid in (0..=loss_id).rev() {
            if let Some(bw) = self.nodes[nid].backward.clone() {
                bw(self, nid)?;
            }
        }
        Ok(())
    }

    fn set_grad_scalar_one(&mut self, id: NodeId) -> Result<()> {
        let t = self.tensor(id).clone();
        if t.shape.numel() != 1 || t.dtype != crate::DType::F32 {
            return Err(FluxError::Autograd(
                "loss must be scalar f32 for now".into(),
            ));
        }
        let one: [f32; 1] = [1.0];
        let bytes: &[u8] = bytemuck::cast_slice(&one);
        let grad = self.tensor_mut(id).grad.as_mut().unwrap();
        t.device.copy_host_to_device(grad, bytes)?;
        Ok(())
    }
}
