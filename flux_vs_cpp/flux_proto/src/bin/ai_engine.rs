// --- FLUX ENGINE (UNIVERSAL LOADER FIX) ---
#[allow(unused_imports, non_snake_case)]
use std::alloc::{alloc, dealloc, Layout};
use std::io::Write;


#[inline(always)] fn from_bf16(val: u16) -> f32 { let bits = (val as u32) << 16; f32::from_bits(bits) }
#[inline(always)] fn to_bf16(val: f32) -> u16 { let bits = val.to_bits(); (bits >> 16) as u16 }

fn pseudo_random(index: usize) -> f32 {
    let mut x = index as u32;
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    x = (x >> 16) ^ x;
    (x as f32) / (u32::MAX as f32)
}

unsafe fn matmul_kernel(a_ptr: *mut u16, b_ptr: *mut u16, c_ptr: *mut u16, m: usize, k: usize, n: usize, trans_a: bool, trans_b: bool) {
    let c_slice = unsafe { std::slice::from_raw_parts_mut(c_ptr, m * n) };
    let a_addr = a_ptr as usize; let b_addr = b_ptr as usize;
    for (row, row_slice) in c_slice.chunks_mut(n).enumerate() {
        let a = a_addr as *mut u16; let b = b_addr as *mut u16;
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                unsafe {
                    let a_val = if trans_a { from_bf16(*a.add(i * m + row)) } else { from_bf16(*a.add(row * k + i)) };
                    let b_val = if trans_b { from_bf16(*b.add(col * k + i)) } else { from_bf16(*b.add(i * n + col)) };
                    sum += a_val * b_val;
                }
            }
            row_slice[col] = to_bf16(sum);
        }
    }
}

unsafe fn adam_update_kernel(w_ptr: *mut u16, g_ptr: *mut u16, m_ptr: *mut u16, v_ptr: *mut u16, size: usize, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: f32) {
    let w_slice = unsafe { std::slice::from_raw_parts_mut(w_ptr, size) };
    let g_slice = unsafe { std::slice::from_raw_parts(g_ptr, size) };
    let m_slice = unsafe { std::slice::from_raw_parts_mut(m_ptr, size) };
    let v_slice = unsafe { std::slice::from_raw_parts_mut(v_ptr, size) };
    let bias_corr1 = 1.0 - beta1.powf(t);
    let bias_corr2 = 1.0 - beta2.powf(t);
    for i in 0..size {
        let grad = from_bf16(g_slice[i]);
        let mut m = from_bf16(m_slice[i]);
        let mut v = from_bf16(v_slice[i]);
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * (grad * grad);
        let m_hat = m / bias_corr1;
        let v_hat = v / bias_corr2;
        let mut w = from_bf16(w_slice[i]);
        w = w - (lr * m_hat / (v_hat.sqrt() + epsilon));
        w_slice[i] = to_bf16(w);
        m_slice[i] = to_bf16(m);
        v_slice[i] = to_bf16(v);
    }
}

unsafe fn mse_gradient_kernel(pred: *mut u16, target: *mut u16, grad_out: *mut u16, size: usize) -> f32 {
    let p_slice = unsafe { std::slice::from_raw_parts(pred, size) };
    let t_slice = unsafe { std::slice::from_raw_parts(target, size) };
    let g_slice = unsafe { std::slice::from_raw_parts_mut(grad_out, size) };
    let mut total_error = 0.0;
    for (i, g) in g_slice.iter_mut().enumerate() {
        let p_val = from_bf16(p_slice[i]);
        let t_val = from_bf16(t_slice[i]);
        let diff = p_val - t_val;
        total_error += diff * diff;
        *g = to_bf16(diff * 2.0 / (size as f32)); 
    }
    total_error / (size as f32)
}

unsafe fn leaky_relu_forward_kernel(input: *mut u16, output: *mut u16, size: usize) {
    let in_slice = unsafe { std::slice::from_raw_parts(input, size) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(output, size) };
    for (i, out) in out_slice.iter_mut().enumerate() {
        let val = from_bf16(in_slice[i]);
        *out = to_bf16(if val > 0.0 { val } else { val * 0.1 });
    }
}

unsafe fn leaky_relu_backward_kernel(input: *mut u16, grad_in: *mut u16, grad_out: *mut u16, size: usize) {
    let in_slice = unsafe { std::slice::from_raw_parts(input, size) };
    let g_in_slice = unsafe { std::slice::from_raw_parts(grad_in, size) };
    let g_out_slice = unsafe { std::slice::from_raw_parts_mut(grad_out, size) };
    for (i, g_out) in g_out_slice.iter_mut().enumerate() {
        let val = from_bf16(in_slice[i]);
        let grad = from_bf16(g_in_slice[i]);
        *g_out = to_bf16(if val > 0.0 { grad } else { grad * 0.1 });
    }
}

unsafe fn save_matrix(ptr: *mut u16, rows: usize, cols: usize, filename: &str) {
    let size = rows * cols;
    let byte_slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size * 2) };
    let mut file = std::fs::File::create(filename).expect("Failed to create file");
    file.write_all(byte_slice).expect("Failed to write weights");
    println!("  [Disk] Saved {}x{} matrix to '{}'", rows, cols, filename);
}

unsafe fn load_matrix_from_disk(ptr: *mut u16, rows: usize, cols: usize, filename: &str) {
    let path = std::path::Path::new(filename);
    let bytes = std::fs::read(path).expect("Failed to load file");
    let expected_bytes = rows * cols * 2;
    if bytes.len() != expected_bytes {
        println!("Warning: Expected {} bytes, got {}", expected_bytes, bytes.len());
    }
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, expected_bytes); }
    println!("  [Disk] Loaded {}x{} matrix from '{}'", rows, cols, filename);
}
fn main() {
    println!("Initializing Flux Engine (Universal Loader)...");

    // --- DYNAMIC BATCH CALCULATION ---
    let in_feat: usize = 90;
    let input_path = "input_alphabet.bin";
    let meta = std::fs::metadata(input_path).expect("❌ Error: Input file not found!");
    let file_bytes = meta.len() as usize;
    let batch = file_bytes / (in_feat * 2);
    println!("  [Auto-Config] Detected File: {} bytes", file_bytes);
    println!("  [Auto-Config] Calculated Batch Size: {}", batch);
    let hid_feat: usize = 512; let out_feat: usize = 80;
    let input_rows = batch; let input_cols = 90;
    let layout = Layout::from_size_align(batch * 90 * 2, 32).unwrap();
    let input = unsafe { alloc(layout) as *mut u16 };
    unsafe { load_matrix_from_disk(input, input_rows, input_cols, "input_alphabet.bin"); }
    let target_rows = batch; let target_cols = 80;
    let layout = Layout::from_size_align(batch * 80 * 2, 32).unwrap();
    let target = unsafe { alloc(layout) as *mut u16 };
    unsafe { load_matrix_from_disk(target, target_rows, target_cols, "target_alphabet.bin"); }
    let w1_rows = 90; let w1_cols = 512;
    let layout = Layout::from_size_align(90 * 512 * 2, 32).unwrap();
    let w1 = unsafe { alloc(layout) as *mut u16 };
    unsafe { 
        let slice = std::slice::from_raw_parts_mut(w1, 90 * 512);
        for (i, x) in slice.iter_mut().enumerate() {
             let r = pseudo_random(i + 90); 
             let noise = (r - 0.5) * 0.1; 
             *x = to_bf16(noise);
        }
    }
    let layout_m_w1 = Layout::from_size_align(90 * 512 * 2, 32).unwrap();
    let m_w1 = unsafe { alloc(layout_m_w1) as *mut u16 };
    unsafe { std::ptr::write_bytes(m_w1, 0, 90 * 512); }
    let layout_v_w1 = Layout::from_size_align(90 * 512 * 2, 32).unwrap();
    let v_w1 = unsafe { alloc(layout_v_w1) as *mut u16 };
    unsafe { std::ptr::write_bytes(v_w1, 0, 90 * 512); }
    let w2_rows = 512; let w2_cols = 80;
    let layout = Layout::from_size_align(512 * 80 * 2, 32).unwrap();
    let w2 = unsafe { alloc(layout) as *mut u16 };
    unsafe { 
        let slice = std::slice::from_raw_parts_mut(w2, 512 * 80);
        for (i, x) in slice.iter_mut().enumerate() {
             let r = pseudo_random(i + 512); 
             let noise = (r - 0.5) * 0.1; 
             *x = to_bf16(noise);
        }
    }
    let layout_m_w2 = Layout::from_size_align(512 * 80 * 2, 32).unwrap();
    let m_w2 = unsafe { alloc(layout_m_w2) as *mut u16 };
    unsafe { std::ptr::write_bytes(m_w2, 0, 512 * 80); }
    let layout_v_w2 = Layout::from_size_align(512 * 80 * 2, 32).unwrap();
    let v_w2 = unsafe { alloc(layout_v_w2) as *mut u16 };
    unsafe { std::ptr::write_bytes(v_w2, 0, 512 * 80); }

    // --- BUFFER LAYOUTS ---
    let layout_hid = Layout::from_size_align(batch * hid_feat * 2, 32).unwrap();
    let hidden = unsafe { alloc(layout_hid) as *mut u16 };
    let layout_out = Layout::from_size_align(batch * out_feat * 2, 32).unwrap();
    let pred = unsafe { alloc(layout_out) as *mut u16 };
    let d_output = unsafe { alloc(layout_out) as *mut u16 };
    let d_hidden = unsafe { alloc(layout_hid) as *mut u16 };
    let layout_w1 = Layout::from_size_align(in_feat * hid_feat * 2, 32).unwrap();
    let d_w1 = unsafe { alloc(layout_w1) as *mut u16 };
    let layout_w2 = Layout::from_size_align(hid_feat * out_feat * 2, 32).unwrap();
    let d_w2 = unsafe { alloc(layout_w2) as *mut u16 };
    let d_hidden_pre_relu = unsafe { alloc(layout_hid) as *mut u16 };

    println!("Starting Training Loop (Adam - 5,000 Epochs)...");
    let lr = 0.01; 
    let beta1 = 0.9; 
    let beta2 = 0.999; 
    let epsilon = 1e-8; 
    for epoch in 1..=5000 {
        unsafe { matmul_kernel(input, w1, hidden, batch, in_feat, hid_feat, false, false); }
        unsafe { leaky_relu_forward_kernel(hidden, hidden, batch * hid_feat); }
        unsafe { matmul_kernel(hidden, w2, pred, batch, hid_feat, out_feat, false, false); }

        // --- BACKWARD PASS ---
        unsafe {
            let mse = mse_gradient_kernel(pred, target, d_output, batch * out_feat);
            matmul_kernel(d_output, w2, d_hidden_pre_relu, batch, out_feat, hid_feat, false, true);
            leaky_relu_backward_kernel(hidden, d_hidden_pre_relu, d_hidden, batch * hid_feat);
            matmul_kernel(input, d_hidden, d_w1, in_feat, batch, hid_feat, true, false);
            matmul_kernel(hidden, d_output, d_w2, hid_feat, batch, out_feat, true, false);
            adam_update_kernel(w2, d_w2, m_w2, v_w2, hid_feat * out_feat, lr, beta1, beta2, epsilon, epoch as f32);
            adam_update_kernel(w1, d_w1, m_w1, v_w1, in_feat * hid_feat, lr, beta1, beta2, epsilon, epoch as f32);
            if epoch % 500 == 0 {
                let p0 = from_bf16(*pred.add(batch/2 * out_feat + 10));
                println!("      Epoch {}: MSE Loss = {:.6} (Sample={:.4})", epoch, mse, p0);
            }
        }
    }

    unsafe {
        dealloc(hidden as *mut u8, layout_hid);
        dealloc(pred as *mut u8, layout_out);
        dealloc(d_output as *mut u8, layout_out);
        dealloc(d_hidden as *mut u8, layout_hid);
        dealloc(d_w1 as *mut u8, layout_w1);
        dealloc(d_w2 as *mut u8, layout_w2);
    }
    unsafe { save_matrix(w1, w1_rows, w1_cols, "hello_w1.bin"); }
    unsafe { save_matrix(w2, w2_rows, w2_cols, "hello_w2.bin"); }
}

