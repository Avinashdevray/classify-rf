# Over-the-Air (OTA) SDR Testing & Replay Guide

**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Resubmission)  
**Hardware Platforms**: ADALM-PLUTO, USRP (B200 / B210), or Recorded I/Q Replay

---

## 1. Scope & Strict Engineering Disclosure

This directory provides the complete workflow for validating the deep learning spectrum classifier against real-world radio frequency emissions captured over-the-air using Software Defined Radios (SDR).

> **Important Disclosure on Reviewer Reproducibility**:  
> In accordance with the challenge principles, **no hardware results are fabricated**. If physical SDR hardware is not attached to the host running MATLAB, the test harness automatically executes in **Hardware-Independent Replay Mode** using sample recorded RF bursts stored in `data/examples/`. This guarantees full end-to-end execution of the receiver calibration, framing, STFT preprocessing, and deep learning classification pipelines without requiring external laboratory equipment.

---

## 2. Hardware Testbed Specifications

| Parameter | Recommended Specification |
| :--- | :--- |
| **SDR Platform** | Analog Devices ADALM-PLUTO or NI/Ettus USRP B210 |
| **Center Frequency ($f_c$)** | `2.437 GHz` (Wi-Fi Channel 6 / BLE Advertising Ch 38) |
| **Baseband Sample Rate ($F_s$)** | `20.0 MSPS` (captures full 20 MHz channel) |
| **RF Bandwidth** | `20.0 MHz` analog filter bandwidth |
| **Receiver Gain** | `38 dB – 42 dB` (Manual Gain Mode) |
| **Antenna** | 2.4 GHz 3 dBi omnidirectional dipole whip antenna |
| **Test Environment** | Laboratory / Office multi-emitter environment |
| **Distance to Emitters** | 1.0 m – 3.0 m |
| **Signal Sources** | Commercial 802.11n Wi-Fi Access Point + BLE smartphone peripheral |
| **Frame Size** | `1,056` complex baseband samples (~52.8 µs per frame) |

---

## 3. Script Inventory & Roles

| Script | Purpose |
| :--- | :--- |
| [`ota/configureSDR.m`](file:///Users/harshita/Desktop/rf-classify/ota/configureSDR.m) | Connects to and configures ADALM-PLUTO or USRP receivers. |
| [`ota/processCapturedIQ.m`](file:///Users/harshita/Desktop/rf-classify/ota/processCapturedIQ.m) | Removes DC offset, measures RF power & SNR, computes STFT, and runs inference. |
| [`ota/runOTATest.m`](file:///Users/harshita/Desktop/rf-classify/ota/runOTATest.m) | Orchestrates frame captures, live/replay execution, and telemetry logging. |

---

## 4. Execution Instructions

### A. Live Over-The-Air Hardware Capture (Requires Physical SDR)
1. Connect ADALM-PLUTO via USB. Ensure the **Communications Toolbox Support Package for ADALM-PLUTO Radio** is installed in MATLAB.
2. In MATLAB Command Window:
   ```matlab
   setup;
   otaResults = runOTATest('pluto', 20, false);
   ```
3. The script will log in-band RF power, estimated SNR, and simultaneous protocol predictions to `results/ota/ota_test_results.mat`.

### B. Hardware-Independent Replay Mode (Zero Hardware Required)
If you do not have an SDR connected:
```matlab
setup;
otaResults = runOTATest('pluto', 10, true);
```
The script will load pre-recorded bursts from `data/examples/sample_wifi_bt_overlap.mat` and `data/examples/sample_wifi_zigbee_overlap.mat`, execute the exact calibration and inference chain, and verify that the multi-label detection system functions correctly.
