# Intelligent 2.4 GHz RF Spectrum Monitoring & Interference Detection Engine

An edge-optimized deep learning physical-layer spectrum sensing system designed to classify multi-standard wireless protocols (**Wi-Fi, Bluetooth/BLE, Zigbee, SmartBAN, and Unknown/Noise**) under extremely harsh Signal-to-Noise Ratio (SNR) conditions ranging from **-20 dB to 0 dB**.

---

## 🌟 Key Architecture & Performance Highlights
- **Tensor Shape Standard**: Strictly enforces `[Batch, 1, 32, 33]` dimensions across datasets, PyTorch models, ONNX export, and real-time visualization.
- **Deep Learning Model (`STFT-RADN`)**:
  - Incorporates **Residual Dense Blocks (RDB)** for feature reuse & local feature fusion.
  - Features **CBAM (Convolutional Block Attention Module)** for channel & spatial attention.
  - Parameter Count: **~852,653 trainable parameters**.
- **Edge Deployment**:
  - Exported to ONNX runtime format (`exports/stft_radn_model.onnx`).
  - CPU Inference Latency: **~10.6 ms target**.
- **Interactive UI**: Full Streamlit dashboard featuring 2D PSD heatmaps, 1D I/Q waveforms, real-time ONNX probability distributions, and live latency metrics.

---

## 📁 Repository Directory Structure

```
RF_Spectrum_Monitor/
├── data/
│   ├── raw_iq/
│   │   └── raw_iq_dataset.npz      # Raw complex 1D I/Q time-domain streams
│   └── stft_dataset.npz            # Memory-cached 2D STFT spectrogram dataset [N, 1, 32, 33]
├── src/
│   ├── 01_data_generator.py        # Multi-standard 2.4 GHz signal generator (-20 dB to 0 dB AWGN)
│   ├── 02_stft_converter.py        # Kaiser-windowed STFT log-PSD transformation & caching
│   ├── models_2d.py                # STFT-RADN (~852k params) & ResNet18_2D baseline models
│   ├── train_2d.py                 # PyTorch training engine (AdamW, ReduceLROnPlateau, EarlyStopping)
│   ├── benchmark_2d.py             # Shape validation, ONNX export, parity & CPU latency benchmark
│   └── dashboard.py                # Interactive Streamlit demo dashboard
├── checkpoints/
│   └── best_model.pth              # Optimal PyTorch model weights
├── exports/
│   └── stft_radn_model.onnx        # High-performance ONNX model artifact
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Requirements Installation
Ensure Python 3.10+ is installed along with required packages:
```bash
pip install torch numpy scipy onnx onnxruntime streamlit matplotlib tqdm
```

### 2. Generate Synthetic Dataset & Compute STFT Spectrograms
Generate 5,000 multi-standard I/Q signals (-20 dB to 0 dB SNR) and compute Kaiser-windowed STFT spectrograms:
```bash
PYTHONPATH=src python3 src/02_stft_converter.py
```
This generates `data/stft_dataset.npz` with strict shape `[5000, 1, 32, 33]`.

### 3. Train the STFT-RADN Model
Train the deep learning model with memory-optimized data loading (`pin_memory=True`), AdamW optimizer, and early stopping:
```bash
PYTHONPATH=src python3 src/train_2d.py
```
Optimal model weights will be saved to `checkpoints/best_model.pth`.

### 4. Benchmark & Export to ONNX
Verify `[1, 1, 32, 33]` shape compliance, export model to ONNX, verify numerical parity, and benchmark CPU inference latency:
```bash
PYTHONPATH=src python3 src/benchmark_2d.py
```

### 5. Launch Interactive Streamlit Dashboard
Run the dark-mode interactive monitoring dashboard:
```bash
streamlit run src/dashboard.py
```

---

## 🔬 Signal Generation & Spectrogram Specifications

### 2.4 GHz Wireless Standards Simulated
1. **Wi-Fi (IEEE 802.11 b/g/n)**: OFDM 64-subcarrier wideband bursts.
2. **Bluetooth / BLE**: GFSK narrowband bursts with frequency hopping offsets.
3. **Zigbee (IEEE 802.15.4)**: O-QPSK DSSS spreading sequence.
4. **SmartBAN (IEEE 802.15.6)**: Duty-cycled medical body area network pulses.
5. **Unknown / Noise**: AWGN and non-stationary linear chirp interference.

### Kaiser STFT Spectrogram Framing
- **Sampling Rate**: 20 MHz
- **Signal Length**: 1,056 samples
- **FFT Size (`nperseg`)**: 64 (yields 33 positive frequency bins)
- **Hop Size (`noverlap`)**: 32 (yields 32 time frames)
- **Window Function**: Kaiser window ($\beta = 14.0$)
- **Output Tensor Shape**: `[Batch, 1, 32, 33]`
