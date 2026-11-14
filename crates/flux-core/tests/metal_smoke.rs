use flux_core::device::metal_device_if_available;

#[cfg(target_os = "macos")]
#[test]
fn metal_device_smoke() {
    let d = metal_device_if_available().unwrap();
    assert!(d.is_some(), "Metal device not found on macOS");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_device_smoke() {
    let d = metal_device_if_available().unwrap();
    assert!(d.is_none(), "Metal should be unavailable off macOS");
}
