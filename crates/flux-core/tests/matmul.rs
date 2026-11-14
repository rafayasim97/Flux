use flux_core::autograd::tape::Tape;
use flux_core::ops::matmul::matmul;
use flux_core::{device, Result, Shape, Tensor};

#[test]
fn matmul_forward_correct() -> Result<()> {
    let dev = device::default_device();
    let mut tape = Tape::new();

    let a = Tensor::from_f32(
        Shape::new(vec![2, 3]),
        dev.clone(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?
    .with_grad(&mut tape);

    let b = Tensor::from_f32(
        Shape::new(vec![3, 2]),
        dev.clone(),
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    )?
    .with_grad(&mut tape);

    let out_id = matmul(&mut tape, &a, &b)?;
    let out = tape.tensor(out_id).to_vec_f32()?;

    let expected = vec![58.0, 64.0, 139.0, 154.0];
    assert_eq!(out, expected);
    Ok(())
}
