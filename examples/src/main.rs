use flux_core::autograd::tape::Tape;
use flux_core::ops::nn;
use flux_core::optim::adamw::AdamW;
use flux_core::{device, Result, Shape, Tensor};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

fn main() -> Result<()> {
    let dev = device::default_device();

    let x = Tensor::from_f32(
        Shape::new(vec![4, 3]),
        dev.clone(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )?;

    let y = Tensor::from_f32(
        Shape::new(vec![4, 2]),
        dev.clone(),
        &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    )?;

    let mut w1 = rand_param(dev.clone(), 3, 8)?;
    let mut w2 = rand_param(dev.clone(), 8, 2)?;

    let mut opt = AdamW::new(0.05);

    for step in 0..200 {
        let mut tape = Tape::new();

        let x_t = x.clone().with_grad(&mut tape);
        let y_t = y.clone().with_grad(&mut tape);
        let w1_t = w1.clone().with_grad(&mut tape);
        let w2_t = w2.clone().with_grad(&mut tape);

        let z1_id = nn::matmul(&mut tape, &x_t, &w1_t)?;
        let z1 = tape.tensor(z1_id).clone();
        let a1_id = nn::relu(&mut tape, &z1)?;
        let a1 = tape.tensor(a1_id).clone();
        let z2_id = nn::matmul(&mut tape, &a1, &w2_t)?;
        let z2 = tape.tensor(z2_id).clone();
        let loss_id = nn::mse(&mut tape, &z2, &y_t)?;

        tape.backward(loss_id)?;

        if step % 20 == 0 {
            let lv = tape.tensor(loss_id).to_vec_f32()?;
            println!("step {step} loss {}", lv[0]);
        }

        let mut params = vec![
            tape.tensor(w1_t.node.expect("w1 should be tracked"))
                .clone(),
            tape.tensor(w2_t.node.expect("w2 should be tracked"))
                .clone(),
        ];
        opt.step(&mut params)?;

        w1 = detach(params.remove(0));
        w2 = detach(params.remove(0));
    }

    Ok(())
}

fn detach(mut t: Tensor) -> Tensor {
    t.requires_grad = false;
    t.grad = None;
    t.node = None;
    t
}

fn rand_param(
    dev: std::sync::Arc<dyn flux_core::device::Device>,
    in_dim: usize,
    out_dim: usize,
) -> Result<Tensor> {
    let normal = Normal::new(0.0, 0.02).unwrap();
    let mut rng = thread_rng();
    let mut v = vec![0f32; in_dim * out_dim];
    for i in 0..v.len() {
        v[i] = normal.sample(&mut rng) as f32;
    }
    Tensor::from_f32(Shape::new(vec![in_dim, out_dim]), dev, &v)
}
