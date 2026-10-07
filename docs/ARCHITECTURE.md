# System Architecture & Technical Specifications

**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Submission)  
**Primary Framework**: MATLAB & Simulink Deep Learning PHY Pipeline

---

## 1. High-Level System Architecture

The project implements a complete physical-layer RF spectrum sensing and interference classification pipeline:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           1. RF INGESTION LAYER                                │
│                                                                                │
│  [Physical ADALM-PLUTO / USRP SDR]   OR   [Recorded I/Q Replay .mat Stream]    │
│  - Center Freq: 2.437 GHz (ISM Band)      - 1,056 Complex Baseband Samples     │
│  - Sampling Rate: 20 MSPS                 - Multi-Signal Co-existence          │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                        2. PHY PREPROCESSING & CALIBRATION                      │
│                                                                                │
│  - Hardware Calibration: DC offset removal & power standardization             │
│  - Kaiser-Windowed STFT (nperseg=64, noverlap=32, beta=14.0)                   │
│  - Log-Scale Power Spectral Density (dB) Transformation                        │
│  - Standardized Spectrogram Tensor: [32 Time Frames x 33 Frequency Bins x 1]   │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                     3. DEEP LEARNING INFERENCE LAYER                           │
│                                                                                │
│  - Residual Dense Network (STFT-RADN) implemented in MATLAB dlnetwork         │
│  - Dense Feature Aggregation + Global Average Pooling 2D                       │
│  - 5-Head Sigmoid Activation for Simultaneous Multi-Label Protocol Detection   │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                   4. DECISION & TELEMETRY DISPLAY LAYER                        │
│                                                                                │
│  - Calibrated Probability Thresholding (tau = 0.5)                             │
│  - Simulink Scope & Display Sink Blocks                                        │
│  - Simultaneous Detection Vector: [WiFi, Bluetooth, Zigbee, SmartBAN, Noise]   │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

```
classify-rf/
├── matlab/
│   ├── setup/               # Path initialization & toolbox verification (checkToolboxes.m, setup.m)
│   ├── data_generation/     # Waveform synthesis (Wi-Fi, BT, Zigbee, SmartBAN, Noise, Overlapping RF)
│   ├── preprocessing/       # Kaiser STFT [32x33], normalization, RF tensor framing
│   ├── models/              # MATLAB deep learning architecture builder (createRFClassifier.m)
│   ├── training/            # Custom BCE training loop with Adam optimizer (trainRFClassifier.m)
│   ├── inference/           # Standalone forward evaluation & thresholding (runInference.m)
│   ├── evaluation/          # Multi-label metrics, overlap & SNR breakdown (evaluateRFClassifier.m)
│   └── utils/               # Signal metrics, plotting, and formatting helpers
├── simulink/
│   ├── rf_signal_classifier.slx  # Native Simulink model-based inference pipeline
│   ├── build_rf_classifier_model.m # Programmatic Simulink model generator
│   └── run_simulink_replay.m       # Offline workspace simulation runner
├── models/
│   └── trainedNetwork.mat   # Pretrained MATLAB dlnetwork weights and metadata
├── data/
│   ├── examples/            # Reference recorded I/Q captures for offline replay
│   └── metadata/            # Dataset manifests and configuration records
├── results/
│   ├── offline/             # Evaluation summary CSV and metric records
│   ├── overlap/             # Regime performance metrics
│   ├── snr/                 # SNR curve telemetry
│   └── ota/                 # Over-the-air capture logs and replay telemetry
├── ota/
│   ├── configureSDR.m       # SDR hardware configuration (ADALM-PLUTO / USRP)
│   ├── processCapturedIQ.m  # Calibration, STFT, and inference on SDR frames
│   ├── runOTATest.m         # Master OTA capture test with replay fallback
│   └── README.md            # Hardware testbed setup and operation guide
├── python/
│   ├── reference_implementation/ # Preserved exploratory PyTorch research code
│   └── README.md            # Contextual guide distinguishing reference from canonical MATLAB
├── docs/                    # Architectural, model, hardware, and audit documentation
└── scripts/
    ├── run_all_offline_experiments.m # Complete automated experiment runner
    └── validate_submission.m         # 9-gate automated verification test suite
```

---

## 3. Preprocessing Specifications

- **Signal Duration**: 1,056 complex baseband samples ($52.8\text{ µs}$ at $20\text{ MSPS}$).
- **Window**: Kaiser window ($N = 64$, shape parameter $\beta = 14.0$).
- **Overlap**: 32 samples (50% overlap, hop size $R = 32$).
- **Output Spectrogram Geometry**:
  $$\text{Frames} = \frac{1056 - 64}{32} + 1 = 32$$
  $$\text{Positive Frequency Bins} = \frac{64}{2} + 1 = 33$$
  Resulting in a strictly uniform tensor size of **`[32 x 33 x 1]`**.
- **Log-PSD Normalization**:
  $$P_{\text{dB}}[m, k] = 10 \log_{10}(|Z_{xx}[m, k]|^2 + 10^{-12})$$
  $$Z_{\text{norm}}[m, k] = \frac{P_{\text{dB}}[m, k] - \mu}{\sigma + 10^{-6}}$$
