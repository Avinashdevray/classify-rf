# Initial Repository Audit Report

**Date**: 2026-10-07  
**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Resubmission Remediation)  
**Target Submission Platform**: MathWorks MATLAB/Simulink Challenge Project Hub  
**Repository**: `https://github.com/Avinashdevray/classify-rf.git`

---

## 1. Executive Summary

This audit assesses the state of the repository prior to remediation. The project was submitted to the MathWorks MATLAB/Simulink Challenge Project Hub ("Classify RF Signals Using AI"). 

The reviewer provided specific critical feedback:
1. **Accessibility**: Original repository access was blocked (resolved: repository is now public).
2. **Project Scope**: Scope requires deep-learning classification of **overlapping and adjacent wireless signals** with **over-the-air (OTA) SDR validation**.
3. **Canonical Implementation**: Verification requires **MATLAB and Simulink** as the primary implementation, along with reviewer-loadable pretrained model weights, toolbox specifications, clean anonymous cloning, and OTA SDR validation.
4. **Preservation of Python Work**: The existing Python/PyTorch implementation should be preserved as reference research, but not represented as the canonical challenge submission.

---

## 2. Inventory of Current Repository Assets

| Path | Type | Size | Description |
| :--- | :--- | :--- | :--- |
| `README.md` | Markdown | 4.2 KB | Documents Python PyTorch / ONNX / Streamlit pipeline. |
| `docs/ML_STUDENT_GUIDE.md` | Markdown | 18.7 KB | Explains baseband I/Q, STFT math, STFT-RADN, ONNX benchmarks. |
| `src/01_data_generator.py` | Python | 8.1 KB | Synthetic I/Q generator for 5 classes (Wi-Fi, BT, Zigbee, SmartBAN, Noise). |
| `src/02_stft_converter.py` | Python | 3.7 KB | Kaiser-windowed STFT log-PSD converter generating `[N, 1, 32, 33]`. |
| `src/models_2d.py` | Python | 9.2 KB | STFT-RADN (852k params) and ResNet18-2D PyTorch architectures. |
| `src/train_2d.py` | Python | 8.4 KB | PyTorch training engine (AdamW, ReduceLROnPlateau). |
| `src/benchmark_2d.py` | Python | 6.5 KB | Tensor shape assertion, ONNX export, CPU latency benchmark. |
| `src/evaluate_benchmarks.py` | Python | 8.4 KB | Classification metrics (Accuracy, F1, Confusion Matrix, SNR sweep). |
| `src/dashboard.py` | Python | 15.6 KB | Interactive Streamlit demo dashboard. |
| `data/raw_iq/raw_iq_dataset.npz` | NPZ binary | ~78 MB | 5,000 raw I/Q samples (1,056 complex samples each). |
| `data/stft_dataset.npz` | NPZ binary | ~19 MB | 5,000 STFT tensors `[5000, 1, 32, 33]`. |
| `checkpoints/best_model.pth` | PyTorch PTH | ~9.8 MB | Model weights (epoch 4, val_loss 1.031, val_acc 56.0%). |
| `exports/stft_radn_model.onnx` | ONNX binary | ~155 KB | ONNX graph definition. |
| `exports/stft_radn_model.onnx.data` | ONNX binary | ~3.2 MB | External tensor weights for ONNX. |

---

## 3. Technical Gap Analysis

### 3.1 Primary Implementation Language & Framework
- **Current State**: 100% Python/PyTorch/Streamlit. No `.m`, `.mlx`, or `.slx` files existed.
- **Challenge Requirement**: MATLAB/Simulink must be the canonical implementation.
- **Remediation**: Build a complete, modular MATLAB codebase covering setup, signal generation, overlapping RF signal synthesis, STFT feature preprocessing, deep learning model construction, training, inference, and evaluation. Add an authentic Simulink model (`simulink/rf_signal_classifier.slx`) with offline replay support.

### 3.2 Signal Overlap & Multi-Signal Scope
- **Current State**: `src/01_data_generator.py` only synthesizes **single-signal isolated bursts** with AWGN and CFO:
  - Class 0: Wi-Fi only
  - Class 1: Bluetooth only
  - Class 2: Zigbee only
  - Class 3: SmartBAN only
  - Class 4: Noise/Chirp only
- **Challenge Requirement**: The project specifically targets **overlapping and adjacent wireless signals** (e.g., Wi-Fi + Bluetooth co-channel or adjacent-channel interference, Bluetooth + Zigbee collisions, multi-signal scenarios).
- **Remediation**: Create `matlab/data_generation/createOverlappingRFExample.m` supporting:
  - Isolated signals
  - Adjacent signals (frequency offsets within band)
  - Partial spectral overlap
  - Heavy spectral overlap
  - Simultaneous multi-signal scenarios (2 to 4 active emitters)
  - Full ground-truth metadata per sample (active protocol indicators, carrier offsets, power ratios, SNR).

### 3.3 Machine Learning Formulation
- **Current State**: Single-label multiclass classification (`CrossEntropyLoss`, `argmax` across 5 classes).
- **Limitation**: When Wi-Fi and Bluetooth overlap simultaneously, a single-label model can only output one class, inherently failing to represent spectrum coexistence.
- **Challenge Requirement**: Support multi-signal presence.
- **Remediation**: Formulate the primary model as **multi-label classification** (sigmoid activations per class / binary cross-entropy) where an RF observation can have simultaneous labels `[WiFi=1, Bluetooth=1, Zigbee=0, SmartBAN=0]`.

### 3.4 Model Artifacts & Weights
- **Current State**: Checkpoints are in PyTorch `.pth` and ONNX formats.
- **Challenge Requirement**: Reviewer-accessible MATLAB `trainedNetwork.mat` / `dlnetwork` weights loadable directly in MATLAB without training or Python dependencies.
- **Remediation**: Implement model definition, export/training pipeline, and produce `models/trainedNetwork.mat` containing the MATLAB network architecture and trained weights alongside normalization constants and STFT parameters.

### 3.5 Over-the-Air (OTA) SDR Testing
- **Current State**: Pure synthetic generation; no hardware capture scripts or SDR configuration exists.
- **Challenge Requirement**: Real-world or recorded SDR over-the-air validation workflow (ADALM-PLUTO / USRP / RTL-SDR).
- **Remediation**: 
  - Create dedicated `ota/` suite: `ota/configureSDR.m`, `ota/runOTATest.m`, `ota/processCapturedIQ.m`.
  - Provide hardware-independent **I/Q replay mode** with sample captured RF data so reviewers without physical SDR hardware can execute and verify the exact OTA inference pipeline.
  - Truthfully state that physical SDR execution requires connected hardware, distinguishing simulated/offline tests from physical OTA measurements.

### 3.6 Repository Structure & Dependencies
- **Current State**: Flat structure mixing Python scripts in `src/`, heavy binary files in git without Git LFS.
- **Remediation**: Reorganize into structured directories (`matlab/`, `simulink/`, `models/`, `data/`, `results/`, `ota/`, `python/reference_implementation/`, `docs/`, `scripts/`).

---

## 4. Unsupported Claims in Previous Documentation

1. **"Overlapping signal classification"**: The previous codebase and dataset contained only isolated signals; overlapping signal classification was not implemented.
2. **"OTA SDR Validation"**: No SDR scripts or capture files were present.
3. **"MATLAB/Simulink Challenge Submission"**: The repository was purely Python-based.

All future documentation will eliminate unverified claims and adhere to absolute transparency regarding implemented features, offline verification, and hardware-dependent procedures.
