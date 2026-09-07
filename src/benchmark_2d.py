"""
benchmark_2d.py
Benchmarking & ONNX Export Engine for STFT-RADN:
1. Validates strict dummy input shape tensor [1, 1, 32, 33].
2. Exports PyTorch weights from `checkpoints/best_model.pth` to `exports/stft_radn_model.onnx`.
3. Verifies PyTorch vs ONNX output parity via numerical assert tests.
4. Benchmarks CPU inference latency across 100+ runs (~10.6 ms target latency).
"""

import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort

from models_2d import get_model, STFT_RADN


def run_shape_validation(dummy_shape=(1, 1, 32, 33)):
    """Verifies that the model accepts exact [1, 1, 32, 33] input tensors with zero dimension errors."""
    print(f"\n[STEP 1] Running Shape Validation Check with tensor {dummy_shape}...")
    dummy_input = torch.randn(*dummy_shape)
    model = STFT_RADN()
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (1, 5), f"Shape validation failed! Expected (1, 5), got {output.shape}"
    print(f"[ASSERT OK] Forward pass succeeded on {dummy_shape} -> Output shape: {output.shape}")
    return dummy_input


def export_onnx(
    ckpt_path="checkpoints/best_model.pth",
    onnx_out_path="exports/stft_radn_model.onnx",
    dummy_shape=(1, 1, 32, 33)
):
    """Exports PyTorch model weights to optimized ONNX format."""
    print(f"\n[STEP 2] Exporting model from {ckpt_path} to ONNX format at {onnx_out_path}...")
    os.makedirs(os.path.dirname(onnx_out_path), exist_ok=True)
    
    device = torch.device("cpu")
    model = STFT_RADN().to(device)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded trained weights from {ckpt_path} (Val Acc: {checkpoint.get('val_acc', 0.0):.2f}%)")
    else:
        print(f"[WARN] Checkpoint {ckpt_path} not found! Exporting initialized model weights.")
        
    model.eval()
    dummy_input = torch.randn(*dummy_shape, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_spectrogram'],
        output_names=['class_logits'],
        dynamic_axes={
            'input_spectrogram': {0: 'batch_size'},
            'class_logits': {0: 'batch_size'}
        }
    )
    
    # Validate ONNX model structure
    onnx_model = onnx.load(onnx_out_path)
    onnx.checker.check_model(onnx_model)
    print(f"[SUCCESS] ONNX model exported and verified at {onnx_out_path}")
    return model, onnx_out_path


def verify_parity(pytorch_model, onnx_path, dummy_shape=(1, 1, 32, 33)):
    """Verifies numerical parity between PyTorch and ONNX Runtime predictions."""
    print("\n[STEP 3] Verifying PyTorch <-> ONNX Output Numerical Parity...")
    dummy_np = np.random.randn(*dummy_shape).astype(np.float32)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        torch_logits = pytorch_model(torch.from_numpy(dummy_np)).numpy()
        
    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: dummy_np})[0]
    
    # Parity check
    np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-4)
    print("[SUCCESS] Numerical Parity Check Passed! Max Abs Difference: "
          f"{np.max(np.abs(torch_logits - onnx_logits)):.6f}")
    return session


def benchmark_cpu_latency(session, dummy_shape=(1, 1, 32, 33), num_warmup=20, num_runs=200):
    """Measures and logs CPU inference latency metrics (~10.6 ms target)."""
    print(f"\n[STEP 4] Benchmarking ONNX Runtime CPU Latency over {num_runs} runs...")
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*dummy_shape).astype(np.float32)
    
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy_input})
        
    latencies_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
        
    latencies_ms = np.array(latencies_ms)
    mean_lat = np.mean(latencies_ms)
    median_lat = np.median(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    
    print("="*55)
    print("      STFT-RADN ONNX CPU INFERENCE LATENCY BENCHMARK")
    print("="*55)
    print(f" Target Latency:      ~10.6 ms")
    print(f" Mean Latency:        {mean_lat:6.2f} ms")
    print(f" Median Latency:      {median_lat:6.2f} ms")
    print(f" P95 Latency:         {p95_lat:6.2f} ms")
    print(f" Min/Max Latency:     {min_lat:.2f} ms / {max_lat:.2f} ms")
    print("="*55)
    return {
        'mean_ms': mean_lat,
        'median_ms': median_lat,
        'p95_ms': p95_lat
    }


def run_benchmark_pipeline():
    dummy_input = run_shape_validation()
    pytorch_model, onnx_path = export_onnx()
    session = verify_parity(pytorch_model, onnx_path)
    metrics = benchmark_cpu_latency(session)
    return metrics


if __name__ == "__main__":
    run_benchmark_pipeline()
