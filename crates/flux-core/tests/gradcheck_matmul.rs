use flux_core::autograd::tape::Tape;
use flux_core::ops::matmul::matmul;
use flux_core::ops::reduce::mse;
use flux_core::{device, Result, Shape, Tensor};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn gradcheck_matmul_wrt_a() -> Result<()> {
    let dev = device::default_device();
    let eps = 1e-3;
    let tol = 5e-2;

    let a_data = vec![0.2f32, -0.1, 0.05, 0.3];
    let b_data = vec![0.4f32, -0.2, 0.1, 0.25];

    let target = vec![0.0f32, 0.0, 0.0, 0.0];

    let mut tape = Tape::new();
    let a = Tensor::from_f32(Shape::new(vec![2, 2]), dev.clone(), &a_data)?.with_grad(&mut tape);
    let b = Tensor::from_f32(Shape::new(vec![2, 2]), dev.clone(), &b_data)?.with_grad(&mut tape);
    let t = Tensor::from_f32(Shape::new(vec![2, 2]), dev.clone(), &target)?.with_grad(&mut tape);

    let out_id = matmul(&mut tape, &a, &b)?;
    let out = tape.tensor(out_id).clone();
    let loss_id = mse(&mut tape, &out, &t)?;
    tape.backward(loss_id)?;

    let a_grad = tape.tensor(a.node.unwrap()).grad.as_ref().unwrap().clone();
    let mut a_grad_bytes = vec![0u8; 2 * 2 * 4];
    dev.copy_device_to_host(&mut a_grad_bytes, &a_grad)?;
    let grad_autodiff = bytemuck::cast_slice::<u8, f32>(&a_grad_bytes).to_vec();

    for idx in 0..a_data.len() {
        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let lp = forward_loss(&a_plus, &b_data, &target)?;

        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let lm = forward_loss(&a_minus, &b_data, &target)?;

        let grad_fd = (lp - lm) / (2.0 * eps);
        assert!(
            approx_eq(grad_autodiff[idx], grad_fd, tol),
            "idx {idx} autodiff={} fd={}",
            grad_autodiff[idx],
            grad_fd
        );
    }

    Ok(())
}

fn forward_loss(a: &[f32], b: &[f32], target: &[f32]) -> Result<f32> {
    let dev = device::default_device();
    let mut tape = Tape::new();
    let a = Tensor::from_f32(Shape::new(vec![2, 2]), dev.clone(), a)?.with_grad(&mut tape);
    let b = Tensor::from_f32(Shape::new(vec![2, 2]), dev.clone(), b)?.with_grad(&mut tape);
    let t = Tensor::from_f32(Shape::new(vec![2, 2]), dev.clone(), target)?.with_grad(&mut tape);
    let out_id = matmul(&mut tape, &a, &b)?;
    let out = tape.tensor(out_id).clone();
    let loss_id = mse(&mut tape, &out, &t)?;
    let lv = tape.tensor(loss_id).to_vec_f32()?;
    Ok(lv[0])
}
