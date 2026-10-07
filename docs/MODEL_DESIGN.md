# Model Design & Formulation Document

**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Submission)  
**Task**: Deep Learning-Based Classification of Overlapping & Adjacent Wireless Signals  
**Canonical Implementation**: MATLAB Deep Learning Toolbox (`dlnetwork` / `layerGraph`)

---

## 1. Problem Formulation: The Critical Failure of Single-Label Classification

In traditional modulation classification (AMC), models are trained on isolated signals where each observation contains exactly one emitter:
$$y \in \{0, 1, 2, \dots, C-1\}$$
Using softmax cross-entropy:
$$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}$$
Under this formulation, $\sum_c P(y = c) = 1.0$. The classes are mutually exclusive by construction.

### Why This Fails for the Challenge Scope
In realistic 2.4 GHz ISM spectrum monitoring:
- A Wi-Fi packet (20 MHz OFDM) and a Bluetooth frequency-hopping burst (1 MHz GFSK) frequently **collide or overlap in time and frequency**.
- Zigbee (802.15.4) packets share channels 11–26 inside Wi-Fi channels 1, 6, and 11.
- If a radio sensor observes simultaneous Wi-Fi and Bluetooth transmission, softmax forces an artificial zero-sum trade-off. Predicting Wi-Fi with 60% probability forces Bluetooth down to at most 40%, preventing the system from identifying co-existence.

---

## 2. Multi-Label Deep Learning Formulation

To genuinely satisfy the challenge requirements for overlapping and adjacent RF signals, we formulate the task as **Multi-Label Spectrum Classification**:

$$\mathbf{y} = [y_1, y_2, y_3, y_4, y_5]^T \in \{0, 1\}^5$$
where:
- $y_1$: Wi-Fi (802.11 b/g/n) active
- $y_2$: Bluetooth / BLE active
- $y_3$: Zigbee (802.15.4) active
- $y_4$: SmartBAN (802.15.6) active
- $y_5$: Unknown / Chirp / High-Noise active

### Output Layer Activation
The neural network outputs unnormalized logits $\mathbf{z} = [z_1, \dots, z_5]^T$. Each class probability is computed independently using the sigmoid function:
$$\hat{y}_c = \sigma(z_c) = \frac{1}{1 + e^{-z_c}} \in [0, 1]$$

### Loss Function
The training loss is Multi-Label Binary Cross-Entropy (BCE):
$$\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{C} \sum_{c=1}^C \left[ y_c \log(\hat{y}_c) + (1 - y_c) \log(1 - \hat{y}_c) \right]$$

### Decision Rule
A protocol is detected as active if its calibrated sigmoid score exceeds a threshold $\tau \in [0, 1]$ (nominal $\tau = 0.5$):
$$\text{Decision}_c = \begin{cases} 1 & \text{if } \hat{y}_c \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Neural Network Architecture: MATLAB STFT-RADN

The network takes a 2D Log-PSD STFT Spectrogram $\mathbf{X} \in \mathbb{R}^{32 \times 33 \times 1}$ as input.

```
       Input Spectrogram [32 x 33 x 1]
                     │
                     ▼
             Image Input Layer
                     │
                     ▼
         Conv2D (1 -> 64, 3x3) + BatchNorm + ReLU
                     │
                     ▼
      Residual Dense Block 1 (RDB-1)
      - Conv (64 -> 32) + ReLU
      - Conv (96 -> 32) + ReLU
      - Conv (128 -> 64) + Local Residual Add
                     │
                     ▼
      Attention Enhancement (Channel/Spatial)
                     │
                     ▼
      Residual Dense Block 2 (RDB-2)
      - Conv (64 -> 32) + ReLU
      - Conv (96 -> 32) + ReLU
      - Conv (128 -> 64) + Local Residual Add
                     │
                     ▼
      Global Average Pooling 2D [1 x 1 x 64]
                     │
                     ▼
      Fully Connected Layer (64 -> 32) + ReLU + Dropout(0.2)
                     │
                     ▼
      Fully Connected Layer (32 -> 5)
                     │
                     ▼
      Sigmoid Output Layer [5 Multi-Hot Probabilities]
```

### Key Architectural Strengths
1. **Residual Dense Connectivity**: Dense inter-layer feature reuse preserves fine spectral details (narrowband GFSK peaks) while capturing wideband OFDM envelope contours.
2. **Global Average Pooling**: Reduces spatial dimensions to a compact channel vector, drastically cutting parameter count and preventing overfitting.
3. **Multi-Label Output**: Each of the 5 protocol heads operates independently, enabling reliable detection of single, dual, triple, or quadruple simultaneous signals.

---

## 4. Evaluation Metrics for Overlapping Scenarios

| Metric | Formulation | Purpose |
| :--- | :--- | :--- |
| **Exact Match (Subset Acc)** | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{\mathbf{y}}_i == \mathbf{y}_i)$ | Strictest metric: all 5 active/inactive states must match exactly. |
| **Hamming Loss** | $\frac{1}{N \cdot C} \sum_{i=1}^N \sum_{c=1}^C |y_{i,c} - \hat{y}_{i,c}|$ | Fraction of wrong protocol predictions across the spectrum. |
| **Macro F1-Score** | $\frac{1}{C} \sum_{c=1}^C \text{F1}_c$ | Balances precision and recall evenly across rare and common protocols. |
| **Micro F1-Score** | Global TP, FP, FN aggregation | Overall multi-label detection efficacy. |
| **Overlap-Specific F1** | F1 calculated strictly on multi-signal subsets | Directly measures performance on simultaneous collisions. |
