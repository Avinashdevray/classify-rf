# 📚 Comprehensive Technical Guide: 2.4 GHz RF Spectrum Monitoring & Deep Learning Interference Engine

Welcome! This guide is specially written for **Machine Learning (ML), Signal Processing, and Edge AI students**. It breaks down every single mathematical, architectural, and software engineering concept used in this project from first principles.

---

## 📋 Table of Contents
1. [Introduction to Physical-Layer RF Spectrum Sensing](#1-introduction-to-physical-layer-rf-spectrum-sensing)
2. [Baseband I/Q Signals & Data Generation (`01_data_generator.py`)](#2-baseband-iq-signals--data-generation-01_data_generatorpy)
3. [Short-Time Fourier Transform (STFT) Feature Extraction (`02_stft_converter.py`)](#3-short-time-fourier-transform-stft-feature-extraction-02_stft_converterpy)
4. [Deep Learning Model Architecture (`models_2d.py`)](#4-deep-learning-model-architecture-models_2dpy)
5. [PyTorch Training Engine (`train_2d.py`)](#5-pytorch-training-engine-train_2dpy)
6. [ONNX Export & CPU Latency Benchmarking (`benchmark_2d.py`)](#6-onnx-export--cpu-latency-benchmarking-benchmark_2dpy)
7. [Comprehensive Evaluation Metrics (`evaluate_benchmarks.py`)](#7-comprehensive-evaluation-metrics-evaluate_benchmarkspy)
8. [Interactive Dashboard UI (`dashboard.py`)](#8-interactive-dashboard-ui-dashboardpy)
9. [Student Summary & Quick Reference Cheat Sheet](#9-student-summary--quick-reference-cheat-sheet)

---

## 1. Introduction to Physical-Layer RF Spectrum Sensing

### What is 2.4 GHz Spectrum Monitoring?
The **2.4 GHz ISM (Industrial, Scientific, and Medical) radio frequency band** (2.400 GHz to 2.4835 GHz) is an unlicenced, extremely crowded wireless medium. Multiple wireless protocols operate simultaneously in this band:
- **Wi-Fi (IEEE 802.11 b/g/n)**: High data-rate, wideband communications (20 MHz channels).
- **Bluetooth / BLE (IEEE 802.15.1)**: Low-power, narrowband bursts with fast Frequency Hopping.
- **Zigbee (IEEE 802.15.4)**: Low-data-rate smart home / sensor mesh networks.
- **SmartBAN (IEEE 802.15.6)**: Ultra-low-power medical Body Area Networks.
- **Unclassified Noise / Chirp**: Microwave ovens, radar pulses, and thermal noise.

Because these standards share the exact same frequency space, **co-existence interference** occurs, degrading network performance.

### Why Deep Learning on Physical Layer (PHY)?
Traditional spectrum sensing relies on energy detectors or handcrafted feature extractors (like cyclostationary feature detection). However:
- Energy detectors **cannot distinguish between Wi-Fi and Bluetooth** at low SNR.
- Handcrafted features break down in harsh noise environments (**-20 dB to 0 dB SNR**).

By transforming 1D raw radio waves into **2D Time-Frequency Spectrograms**, we can reframe wireless signal classification as an **Image Classification task** solved by modern Convolutional Neural Networks (CNNs).

---

## 2. Baseband I/Q Signals & Data Generation (`01_data_generator.py`)

### What are In-Phase and Quadrature (I/Q) Signals?
Radio receivers convert high-frequency 2.4 GHz radio waves into complex baseband signals represented as two channels:
- **$I(t)$ (In-Phase)**: Real component of the complex envelope.
- **$Q(t)$ (Quadrature)**: Imaginary component of the complex envelope ($90^\circ$ phase-shifted).

A single complex baseband sample at time step $t$ is represented mathematically as:
$$s(t) = I(t) + j Q(t) = A(t) e^{j \phi(t)}$$
where $A(t) = \sqrt{I(t)^2 + Q(t)^2}$ is the instantaneous amplitude, and $\phi(t) = \arctan\left(\frac{Q(t)}{I(t)}\right)$ is the instantaneous phase.

### Mathematics of the 5 Wireless Protocols

```
                          1D Baseband Complex Signals (N = 1056)
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │ Wi-Fi: Multi-carrier OFDM (64 subcarriers, Cyclic Prefix)                         │
  │ Bluetooth: Single-carrier GFSK with Frequency Hopping Offsets                     │
  │ Zigbee: O-QPSK DSSS spreading (2 Mchips/s half-sine pulse shaping)               │
  │ SmartBAN: Duty-cycled narrowband pulse bursts                                     │
  │ Unknown/Noise: Additive White Gaussian Noise (AWGN) + Linear Chirp                │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

#### 1. Wi-Fi (OFDM Simulation)
- Uses **Orthogonal Frequency Division Multiplexing (OFDM)** with 64 subcarriers.
- Random QPSK symbols are assigned to active subcarrier indices $[-26 \dots -1, 1 \dots 26]$.
- Inverse Fast Fourier Transform (IFFT) converts subcarriers to time domain:
  $$x[n] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X[k] e^{j \frac{2\pi k n}{N}}$$
- A **Cyclic Prefix (CP)** of 16 samples is prepended to eliminate Inter-Symbol Interference (ISI).

#### 2. Bluetooth / BLE (GFSK & Frequency Hopping)
- Uses **Gaussian Frequency Shift Keying (GFSK)** where binary bits $\{-1, +1\}$ pass through a Gaussian pulse shaping filter:
  $$h(t) = \frac{1}{\sqrt{2\pi} \sigma T} e^{-\frac{t^2}{2\sigma^2 T^2}}$$
- The signal is frequency modulated with peak deviation $f_\Delta = 250\text{ kHz}$:
  $$\phi(t) = 2\pi f_\Delta \int_{-\infty}^{t} g(\tau) d\tau$$
- **Frequency Hopping**: A random carrier offset $f_{hop} \in [-4\text{ MHz}, +4\text{ MHz}]$ is added to simulate Bluetooth channel hopping across 2.4 GHz channels.

#### 3. Zigbee (O-QPSK DSSS)
- Uses **Direct Sequence Spread Spectrum (DSSS)** with a 2 Mchip/s spreading rate.
- **Offset QPSK (O-QPSK)** delays the $Q(t)$ channel by half a chip period ($T_c / 2$) to reduce envelope fluctuations.
- Half-sine pulse shaping filter: $p(t) = \sin\left(\frac{\pi t}{T_c}\right)$ for $0 \le t \le T_c$.

#### 4. SmartBAN (IEEE 802.15.6)
- Duty-cycled narrowband pulses representing low-power medical telemetry bursts.
- Uses pulse envelope masking: $s_{smartban}(t) = s_{GFSK}(t) \cdot m(t)$, where $m(t) \in \{0, 1\}$ is an active transmission window.

#### 5. Unknown / Noise
- Models unclassified interference: pure **Additive White Gaussian Noise (AWGN)** or **Linear Chirps** where frequency sweeps linearly across time:
  $$s_{chirp}(t) = \exp\left(j 2\pi \left(f_0 t + \frac{f_1 - f_0}{2 T} t^2\right)\right)$$

### Impairment Injections (Noise & Carrier Offsets)
Real-world radio receivers suffer from noise and oscillator misalignment.
1. **AWGN Noise Injection**: Given a target $SNR_{dB} \in [-20\text{ dB}, 0\text{ dB}]$, noise power is computed as:
   $$P_{signal} = \frac{1}{N} \sum_{n=1}^{N} |s[n]|^2, \quad P_{noise} = \frac{P_{signal}}{10^{SNR_{dB}/10}}$$
   $$n[n] = \frac{1}{\sqrt{2}} (\mathcal{N}(0, P_{noise}) + j \mathcal{N}(0, P_{noise}))$$
2. **Carrier Frequency Offset (CFO)**: Oscillator inaccuracy creates phase drift $\Delta f \in [-50\text{ kHz}, +50\text{ kHz}]$:
   $$s_{noisy}[n] = \left(s[n] \cdot e^{j (2\pi \Delta f \frac{n}{F_s} + \theta)}\right) + n[n]$$

---

## 3. Short-Time Fourier Transform (STFT) Feature Extraction (`02_stft_converter.py`)

### DFT vs. STFT: The Time-Frequency Trade-off
A standard Fast Fourier Transform (FFT) tells us **which frequencies** are present, but loses **when** they occurred. 
The **Short-Time Fourier Transform (STFT)** solves this by sliding a small window function $w[n]$ across time:

$$STFT\{x[n]\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m R] e^{-j \omega n}$$

where $m$ is the time frame index and $R$ is the hop size.

```
       1D I/Q Signal (1056 samples)
      [============================]
                   │
                   ▼  Sliding Kaiser Window (nperseg=64, hop=32)
      Frame 1: [====] ───► 64-pt FFT ───► [33 Freq Bins]
      Frame 2:    [====] ───► 64-pt FFT ───► [33 Freq Bins]
        ...
      Frame 32:      ... [====] ───► 64-pt FFT ───► [33 Freq Bins]
                   │
                   ▼  Transpose & Normalize
      Spectrogram Tensor Shape: [Batch, 1, 32, 33]
```

### Derivation of Strict Tensor Dimensions `[Batch, 1, 32, 33]`
Why exactly **32 time frames** and **33 frequency channels**? Let's check the math:
1. **Frequency Bins ($33$)**:
   - $N_{perseg} = 64$ points per window segment.
   - A 64-point FFT yields $N_{perseg} / 2 + 1 = 64/2 + 1 = \mathbf{33}$ positive frequency bins ($0\text{ Hz}$ to $F_s/2 = 10\text{ MHz}$).
2. **Time Frames ($32$)**:
   - Total I/Q signal length $N_{samples} = 1056$.
   - Window size $N_{perseg} = 64$, overlap $N_{overlap} = 32$, hop size $R = N_{perseg} - N_{overlap} = 32$.
   - Without zero-padding boundaries (`boundary=None`, `padded=False`), the total number of time frames is:
     $$N_{frames} = \frac{N_{samples} - N_{perseg}}{R} + 1 = \frac{1056 - 64}{32} + 1 = \frac{992}{32} + 1 = 31 + 1 = \mathbf{32}$$
3. **Matrix Alignment**:
   - Raw STFT output matrix $Z_{xx}$ has shape $(33\text{ frequency bins}, 32\text{ time frames})$.
   - Transposing $Z_{xx}^T$ yields $(32, 33)$.
   - Adding single channel dimension $\rightarrow (1, 32, 33)$.
   - Batch dimension $B \rightarrow \mathbf{[Batch, 1, 32, 33]}$.

### Log-PSD Normalization
Magnitude Power Spectral Density (PSD) spans dynamic ranges across multiple orders of magnitude. We transform to decibels (dB) and apply Z-score standardization:

$$P_{dB}[m, k] = 10 \log_{10} \left( |Z_{xx}[m, k]|^2 + \epsilon \right)$$
$$Z_{norm}[m, k] = \frac{P_{dB}[m, k] - \mu_{dB}}{\sigma_{dB} + \epsilon}$$

This zero-mean, unit-variance tensor speeds up neural network convergence.

---

## 4. Deep Learning Model Architecture (`models_2d.py`)

The custom **`STFT-RADN` (Residual Dense + CBAM Attention Network)** model is built specifically for physical-layer signal classification.

```
       Input Spectrogram [B, 1, 32, 33]
                      │
                      ▼
             Conv Stem (1 -> 104)
                      │
                      ▼
         ┌─────────────────────────┐
         │ RDB 1 (Dense Feature)   │
         │ CBAM 1 (Attn: Ch & Sp) │  ◄── Block 1
         │ Conv Residual (104->104)│
         └─────────────────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ RDB 2 (Dense Feature)   │
         │ CBAM 2 (Attn: Ch & Sp) │  ◄── Block 2
         │ Conv Residual (104->104)│
         └─────────────────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ RDB 3 (Dense Feature)   │
         │ CBAM 3 (Attn: Ch & Sp) │  ◄── Block 3
         │ Conv Residual (104->104)│
         └─────────────────────────┘
                      │
                      ▼
         AdaptiveAvgPool2d((1, 1)) [B, 104, 1, 1]
                      │
                      ▼
         Linear Classifier (104 -> 115 -> 5)
```

### Module 1: Residual Dense Block (RDB)
Traditional CNNs stack convolutional layers sequentially, losing low-level time-frequency details. An **RDB** re-uses features from **all preceding layers** within the block:

1. **Dense Layer 1**: $x_1 = \text{ReLU}(\text{BN}(\text{Conv}_{3\times 3}(x_0)))$ $\rightarrow$ Concatenate $[x_0, x_1]$.
2. **Dense Layer 2**: $x_2 = \text{ReLU}(\text{BN}(\text{Conv}_{3\times 3}([x_0, x_1])))$ $\rightarrow$ Concatenate $[x_0, x_1, x_2]$.
3. **Dense Layer 3**: $x_3 = \text{ReLU}(\text{BN}(\text{Conv}_{3\times 3}([x_0, x_1, x_2])))$ $\rightarrow$ Concatenate $[x_0, x_1, x_2, x_3]$.
4. **Local Feature Fusion (LFF)**: A $1 \times 1$ conv compresses concatenated features back to input channel count $C$:
   $$y_{LFF} = \text{BN}(\text{Conv}_{1\times 1}([x_0, x_1, x_2, x_3]))$$
5. **Local Residual Connection**:
   $$y_{RDB} = x_0 + y_{LFF}$$

### Module 2: CBAM (Convolutional Block Attention Module)
CBAM applies sequential **Channel Attention** ("WHAT features matter?") and **Spatial Attention** ("WHERE in time-frequency is the signal?"):

1. **Channel Attention**:
   - Performs Average Pooling and Max Pooling across spatial dimensions $(H, W)$:
     $$F_{avg} = \text{Mean}_{H,W}(x), \quad F_{max} = \text{Max}_{H,W}(x)$$
   - Passes both vectors through a shared Multi-Layer Perceptron (MLP) with bottleneck ratio $r=8$:
     $$M_c(x) = \sigma \left( \text{MLP}(F_{avg}) + \text{MLP}(F_{max}) \right)$$
   - Output: $x' = M_c(x) \otimes x$.

2. **Spatial Attention**:
   - Pools feature maps along the channel axis:
     $$F_{avg}^s = \text{Mean}_{C}(x'), \quad F_{max}^s = \text{Max}_{C}(x')$$
   - Concatenates maps and applies a $7 \times 7$ convolution:
     $$M_s(x') = \sigma \left( \text{Conv}_{7\times 7}([F_{avg}^s, F_{max}^s]) \right)$$
   - Final Output: $y_{CBAM} = M_s(x') \otimes x'$.

### Exact Parameter Count Derivation (`852,653` Parameters)
Let's see how we engineered ~852.6k parameters:
- **Stem Conv**: $\text{Conv}(1 \to 104, 3\times 3) + \text{BN} = (1 \cdot 104 \cdot 9) + (104 + 104) = 1,144$ params.
- **Each RDB Block**:
  - Dense Conv 1: $(104 \to 40, 3\times 3) + \text{BN} = 37,440 + 80 = 37,520$ params.
  - Dense Conv 2: $(144 \to 40, 3\times 3) + \text{BN} = 51,840 + 80 = 51,920$ params.
  - Dense Conv 3: $(184 \to 40, 3\times 3) + \text{BN} = 66,240 + 80 = 66,320$ params.
  - Fusion Conv: $(224 \to 104, 1\times 1) + \text{BN} = 23,296 + 208 = 23,504$ params.
- **Total STFT-RADN Parameters**: **852,653 parameters** (~852.6k).
- **Comparison**: ResNet18-2D uses **11,170,245 parameters**. STFT-RADN delivers a **92.6% parameter reduction** while maintaining edge suitability!

---

## 5. PyTorch Training Engine (`train_2d.py`)

### DataLoader Optimization (`pin_memory=True`)
When using GPUs/MPS, `pin_memory=True` allocates tensors in page-locked (pinned) system memory. This allows direct Direct Memory Access (DMA) transfer from Host CPU RAM to GPU VRAM, bypassing CPU copy overheads.

### Optimization Mechanics
1. **AdamW Optimizer**: Extends Adam by decoupling weight decay regularization from gradient updates:
   $$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
   where $\lambda = 10^{-4}$ is the weight decay coefficient.
2. **ReduceLROnPlateau Scheduler**: Automatically decays learning rate by $50\%$ ($\text{factor}=0.5$) if validation loss fails to decrease for 2 consecutive epochs ($\text{patience}=2$).
3. **Early Stopping**: Saves optimal weights to `checkpoints/best_model.pth` and terminates training if no loss reduction occurs after 5 epochs, preventing overfitting on low-SNR noise.

---

## 6. ONNX Export & CPU Latency Benchmarking (`benchmark_2d.py`)

### What is ONNX (Open Neural Network Exchange)?
PyTorch models contain Python overhead during execution. **ONNX** converts the PyTorch dynamic computational graph into a static, highly optimized intermediate representation (`exports/stft_radn_model.onnx`).

### Numerical Parity Verification
Before deployment, we verify that predictions produced by PyTorch and ONNX Runtime are numerically identical:

$$\max \left| y_{\text{PyTorch}} - y_{\text{ONNX}} \right| \le 1.0 \times 10^{-6}$$

### Latency Benchmarking Metrics
We measure ONNX Runtime CPU latency across 500 execution runs after 30 warm-up iterations:
- **Target Budget**: $\sim 10.6\text{ ms}$
- **Measured Mean Latency**: **3.22 ms** ($\sim 3.3\times$ faster than target!)
- **Measured Median Latency**: **3.12 ms**
- **Measured P95 Latency**: **3.67 ms**

---

## 7. Comprehensive Evaluation Metrics (`evaluate_benchmarks.py`)

### Classification Performance Metrics
Given True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$):
- **Precision**: Proportion of positive predictions that are correct: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall**: Proportion of actual positive signals correctly identified: $\text{Recall} = \frac{TP}{TP + FN}$
- **F1-Score**: Harmonic mean balancing precision and recall: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

### Test Results Breakdown (750 Test Samples)
- **Zigbee**: F1-Score **68.32%** (Precision 56.33%, Recall 86.79%)
- **Wi-Fi**: F1-Score **58.87%** (Precision 62.93%, Recall 55.30%)
- **SmartBAN**: F1-Score **55.91%** (Precision 63.39%, Recall 50.00%)
- **Bluetooth**: F1-Score **49.60%** (Precision 43.40%, Recall 57.86%)
- **High-SNR Accuracy ($\ge -2\text{ dB}$)**: **91.78%** at $-2\text{ dB}$ SNR.

---

## 8. Interactive Dashboard UI (`dashboard.py`)

### Streamlit Architecture & State Persistence
Streamlit reruns the script on every user interaction. To prevent regenerating heavy signal bursts or reloading models on every frame, we use `st.session_state`:

```python
if "iq_signal" not in st.session_state or resample_button_clicked:
    st.session_state["iq_signal"] = generate_synthetic_iq(...)
```

### UI Design System
Built using a modern Dribbble-inspired design palette:
- **Primary Lime Accent (`#dcf763`)**: Used for top predicted metrics, active status badges, and sliders.
- **Dark Charcoal (`#141518`)**: Used for sidebar, telemetry cards, and active tab pills.
- **Soft Off-White (`#e9edec`)**: Main dashboard canvas background.
- **Typography**: Imported Google Font **Plus Jakarta Sans** for crisp legibility.

---

## 9. Student Summary & Quick Reference Cheat Sheet

| Component | File Path | Key Formula / Tech | Tensor / Value |
| :--- | :--- | :--- | :--- |
| **I/Q Generation** | `src/01_data_generator.py` | OFDM, GFSK, DSSS, AWGN | $N = 1056$ complex samples |
| **STFT Feature** | `src/02_stft_converter.py` | Kaiser Window $\beta=14.0$, $N_{perseg}=64$ | `[Batch, 1, 32, 33]` |
| **Model Arch** | `src/models_2d.py` | RDB + CBAM Attention | **852,653 parameters** |
| **Training Engine** | `src/train_2d.py` | AdamW, ReduceLROnPlateau, EarlyStopping | `checkpoints/best_model.pth` |
| **ONNX Bench** | `src/benchmark_2d.py` | ONNX Runtime CPU Execution | **3.12 ms Median Latency** |
| **Evaluation** | `src/evaluate_benchmarks.py` | Precision, Recall, F1, Confusion Matrix | **91.78% Acc @ -2 dB** |
| **Web Dashboard** | `src/dashboard.py` | Streamlit + Matplotlib + Custom CSS | `http://localhost:8501` |
