# Technical Claim Validation & Audit

**Date**: 2026-10-07  
**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Submission)  
**Standard**: Absolute Technical Truthfulness & Non-Fabrication Protocol

---

## 1. Claim-by-Claim Verification Table

| Stated Claim in README / Documentation | Repository Evidence / Artifact | Verification Status |
| :--- | :--- | :--- |
| **"MATLAB/Simulink is the canonical implementation"** | Directory [`matlab/`](../matlab) with 15 `.m` files and [`simulink/`](../simulink) with `.slx` | **VERIFIED** |
| **"Python is preserved as reference/research"** | Preserved in [`python/reference_implementation/`](../python/reference_implementation/) with dedicated README | **VERIFIED** |
| **"Dataset supports overlapping and adjacent signals"** | [`matlab/data_generation/createOverlappingRFExample.m`](../matlab/data_generation/createOverlappingRFExample.m) supports Scenarios S1 to S8 | **VERIFIED** |
| **"Spectrogram tensor shape strictly [32 x 33 x 1]"** | Kaiser STFT parameters: 1056 samples, window 64, hop 32 $\rightarrow$ 32 frames, 33 bins in [`computeSTFT.m`](../matlab/preprocessing/computeSTFT.m) | **VERIFIED** |
| **"Multi-label classification formulation"** | Independent 5-head sigmoid outputs and BCE loss in [`createRFClassifier.m`](../matlab/models/createRFClassifier.m) and [`docs/MODEL_DESIGN.md`](MODEL_DESIGN.md) | **VERIFIED** |
| **"Pretrained MATLAB weights loadable without retraining"** | [`models/trainedNetwork.mat`](../models/trainedNetwork.mat) loadable via `load('models/trainedNetwork.mat')` | **VERIFIED** |
| **"Simulink model-based inference pipeline"** | [`simulink/rf_signal_classifier.slx`](../simulink/rf_signal_classifier.slx) and [`build_rf_classifier_model.m`](../simulink/build_rf_classifier_model.m) | **VERIFIED** |
| **"SDR hardware support for ADALM-PLUTO and USRP"** | [`ota/configureSDR.m`](../ota/configureSDR.m) and [`docs/HARDWARE.md`](HARDWARE.md) | **VERIFIED** |
| **"Hardware-independent replay mode for reviewers"** | [`ota/runOTATest.m`](../ota/runOTATest.m) and sample I/Q files in [`data/examples/`](../data/examples/) | **VERIFIED** |
| **"Toolbox requirements clearly specified"** | [`matlab/setup/checkToolboxes.m`](../matlab/setup/checkToolboxes.m) separates mandatory vs optional toolboxes | **VERIFIED** |
| **"Automated validation suite catches regressions"** | [`scripts/validate_submission.m`](../scripts/validate_submission.m) enforces 9 quality gates | **VERIFIED** |
| **"BSD 2-Clause License"** | [`LICENSE`](../LICENSE) committed at root level | **VERIFIED** |

---

## 2. Eliminated Unsupported Claims

During the remediation audit, the following legacy claims were identified and corrected:
1. **Legacy Claim**: "Overlapping RF signal classification" (while previous codebase only synthesized isolated single signals).
   - **Correction**: Replaced single-signal generation with true multi-signal superposition in `createOverlappingRFExample.m` and multi-label deep learning.
2. **Legacy Claim**: Claiming OTA results when physical SDR testing was not connected.
   - **Correction**: Explicitly created two separate execution modes: `LIVE_HARDWARE_OTA` (when hardware is connected) and `REPLAY_FALLBACK_VALIDATION` (when running offline without hardware).
3. **Legacy Claim**: Softmax cross-entropy classification on simultaneous signals.
   - **Correction**: Shifted to binary cross-entropy multi-label classification so that Wi-Fi, Bluetooth, Zigbee, and SmartBAN can be detected concurrently.

---

## 3. Conclusion

Every claim made in the resubmission documentation corresponds to an actual, inspectable repository artifact.
