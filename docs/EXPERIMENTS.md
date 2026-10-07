# Offline Experimental Suite & Evaluation Protocol

**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Submission)  
**Evaluation Script**: [`scripts/run_all_offline_experiments.m`](file:///Users/harshita/Desktop/rf-classify/scripts/run_all_offline_experiments.m)

---

## 1. Experimental Methodology

The challenge submission evaluates the deep learning model across seven distinct experimental setups designed to test modulation fidelity, spectral co-existence resolution, and noise robustness:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           7 EVALUATION EXPERIMENTS                             │
│                                                                                │
│  [E1: Isolated Signals]    Single protocol baseline (Wi-Fi, BT, Zigbee, etc.)  │
│  [E2: Adjacent Signals]    Emitters separated by frequency guard-bands         │
│  [E3: Partial Overlap]     Signals partially overlapping in frequency spectrum │
│  [E4: Heavy Overlap]       Severe frequency collision between narrowband bursts│
│  [E5: Multi-Signal]        3 or 4 simultaneous active protocols                │
│  [E6: SNR Sweep]           Sensitivity sweep across -20 dB to 0 dB             │
│  [E7: RF Impairments]      Robustness under CFO (+/- 50 kHz) and fading        │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Evaluation Metrics

For multi-label classification of simultaneous RF protocols:
1. **Subset Accuracy (Exact Match)**: Percentage of samples where the predicted multi-hot vector matches ground truth for all 5 protocols simultaneously:
   $$\text{ExactMatch} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{\mathbf{y}}_i == \mathbf{y}_i)$$
2. **Hamming Loss**: Average fraction of misclassified protocol labels across all samples:
   $$\text{HammingLoss} = \frac{1}{N \cdot C} \sum_{i=1}^N \sum_{c=1}^C |y_{ic} - \hat{y}_{ic}|$$
3. **Per-Class Precision, Recall, and F1-Score**: Harmonic balance between true positive detections and false alarms for each individual protocol.
4. **Macro F1-Score**: Unweighted average of F1-scores across all 5 classes.

---

## 3. Detailed Experiment Breakdown

### Experiment 1: Isolated Signal Classification (Baseline)
- **Objective**: Verify that individual protocols are correctly classified in the absence of co-channel interference.
- **Scenarios**: S1 (`Wi-Fi Only`), S2 (`Bluetooth Only`), Clean Zigbee, Clean SmartBAN, and AWGN Noise.

### Experiment 2: Adjacent Signal Classification
- **Objective**: Verify that adjacent spectral channels (e.g., Bluetooth operating at $\pm 7.5\text{ MHz}$ relative to Wi-Fi channel center) are independently resolved without causing false positive alarms on unrepresented channels.
- **Scenario**: S3 (`WiFi_BT_Adjacent`).

### Experiment 3: Partial Spectral Overlap
- **Objective**: Assess multi-label detection when Bluetooth frequency hops directly into the outer subcarriers of an active Wi-Fi transmission.
- **Scenario**: S4 (`WiFi_BT_PartialOverlap`).

### Experiment 4: Heavy Spectral Overlap
- **Objective**: Benchmark performance when two narrowband transmissions (e.g., Bluetooth and Zigbee) directly collide within the exact same frequency channel.
- **Scenario**: S6 (`BT_Zigbee_Overlap`).

### Experiment 5: Multi-Signal Crowded Spectrum
- **Objective**: Test detection capability when three or four distinct protocols transmit concurrently within the observation window.
- **Scenarios**: S7 (`WiFi_BT_Zigbee_Simultaneous`) and S8 (`Crowded_Spectrum_4Signals`).

### Experiment 6: SNR Performance Sweep
- **Objective**: Measure detection degradation across extreme noise regimes from **-20 dB to 0 dB** in 2 dB steps.

### Experiment 7: Carrier Frequency Offset (CFO) Robustness
- **Objective**: Measure classifier resilience to uncalibrated local oscillator drift ($\pm 50\text{ kHz}$) and phase noise.

---

## 4. Execution Command

To run the complete automated evaluation suite in MATLAB:
```matlab
setup;
results = run_all_offline_experiments('smoke');  % Fast validation
% or
results = run_all_offline_experiments('full');   % Comprehensive evaluation
```
Numerical results are written directly to `results/offline/summary.csv` and `results/offline/metrics.mat`.
