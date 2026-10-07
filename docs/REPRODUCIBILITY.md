# Reviewer Reproducibility Guide

**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Submission)  
**Target Platform**: MathWorks MATLAB / Simulink (R2021a or newer; R2022b+ recommended)

---

## 1. Quick-Start Verification (Under 2 Minutes)

Reviewers can verify the entire canonical submission in four commands inside the MATLAB Command Window:

```matlab
% Step 1: Initialize paths and verify toolboxes
setup;

% Step 2: Validate submission integrity across 9 automated quality gates
validate_submission;

% Step 3: Run standalone multi-label inference on a Wi-Fi + Bluetooth overlap burst
runInference;

% Step 4: Run hardware-independent SDR replay validation
runOTATest('pluto', 5, true);
```

---

## 2. End-to-End Workflow Reproduction

### Step 1: Project Setup & Environment Verification
Execute [`matlab/setup/setup.m`](file:///Users/harshita/Desktop/rf-classify/matlab/setup/setup.m):
```matlab
setup;
```
This script adds all necessary directories to your MATLAB path, checks required and optional toolboxes, verifies that output directories exist, and confirms that `models/trainedNetwork.mat` is loadable.

### Step 2: Dataset Generation
To generate a reproducible multi-signal RF dataset with deterministic random seeds:
```matlab
% Mode can be 'smoke' (120 samples), 'development' (800 samples), or 'full' (4000 samples)
[dataset, manifest] = generateDataset('smoke', 'data', 42);

% Validate dataset integrity and shape compliance
validateDataset('data/rf_dataset_smoke.mat');
```

### Step 3: Deep Learning Model Training (Optional)
To retrain the MATLAB multi-label classifier from scratch using `dlnetwork` and the custom BCE training loop:
```matlab
trainRFClassifier('data/rf_dataset_smoke.mat');
```
*Note: A pretrained model is already included in `models/trainedNetwork.mat`, so retraining is optional.*

### Step 4: Standalone Inference
Test inference without retraining:
```matlab
% Runs inference on a synthetic test burst
[results, activeProtocols, probs] = runInference();

% Or pass a specific sample file
runInference('data/examples/sample_wifi_bt_overlap.mat');
```

### Step 5: Automated Offline Experiments
Run all 7 experiments across overlap regimes and SNR sweeps:
```matlab
results = run_all_offline_experiments('smoke');
```
Outputs are saved to `results/offline/summary.csv` and `results/offline/metrics.mat`.

### Step 6: Simulink Offline Replay Simulation
To inspect and run the Simulink model without SDR hardware:
```matlab
open_system('simulink/rf_signal_classifier.slx');
run_simulink_replay('data/examples/sample_wifi_bt_overlap.mat');
```

### Step 7: SDR / Over-the-Air Validation
- **With ADALM-PLUTO hardware attached**:
  ```matlab
  otaResults = runOTATest('pluto', 20, false);
  ```
- **Without hardware (Replay Mode)**:
  ```matlab
  otaResults = runOTATest('pluto', 10, true);
  ```

---

## 3. Determinism & Random Seeds

All synthetic data generation functions use explicit, configurable seeds:
- Dataset master seed: `42`
- Sample offset calculation: `baseSeed + sampleIdx * 101`
- Train/Validation/Test split seed: `rng(42)`
This guarantees exact numerical reproducibility across different MATLAB sessions.
