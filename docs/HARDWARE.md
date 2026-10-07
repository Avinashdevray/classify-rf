# SDR Hardware Testbed & Interfacing Guide

**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Submission)  
**Supported Radios**: Analog Devices ADALM-PLUTO, NI/Ettus Research USRP B210

---

## 1. Supported Software Defined Radio (SDR) Hardware

| Hardware Platform | RF Coverage | Max Bandwidth | Connection | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Analog Devices ADALM-PLUTO** | 325 MHz – 3.8 GHz | 20 MHz (hackable to 56 MHz) | USB 2.0 | Portable physical-layer 2.4 GHz monitoring |
| **NI / Ettus Research USRP B210** | 70 MHz – 6.0 GHz | 56 MHz real-time | USB 3.0 | High-fidelity multi-channel laboratory sensing |
| **Recorded I/Q Replay (Software)** | Virtual 2.4 GHz | 20 MHz equivalent | N/A | Hardware-independent reviewer evaluation |

---

## 2. MATLAB Hardware Support Packages

To interface with physical hardware, install the appropriate MathWorks support package:
- **ADALM-PLUTO**:
  - In MATLAB Add-On Explorer, install: `Communications Toolbox Support Package for ADALM-PLUTO Radio`.
  - Verify via MATLAB command: `checkToolboxes;` or `plutorx = sdrrx('Pluto');`.
- **USRP Radios**:
  - In MATLAB Add-On Explorer, install: `Communications Toolbox Support Package for USRP Radio`.

---

## 3. Physical Testbed Setup

```
   ┌───────────────────────┐             ┌─────────────────────────┐
   │ 2.4 GHz Wi-Fi Router  │             │   Commercial Smartphone │
   │ (e.g. 802.11n Ch 6)   │             │   (Active BLE Beacon)   │
   └───────────┬───────────┘             └────────────┬────────────┘
               │                                      │
               │ Over-the-Air Wireless Propagation    │
               │ (Distance: 1.0 - 3.0 meters)         │
               ▼                                      ▼
      ┌────────────────────────────────────────────────────────┐
      │  2.4 GHz Omnidirectional Whip Antenna (SMA Connector)  │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │               ADALM-PLUTO Radio Receiver               │
      │  - Center Frequency: 2.437 GHz                         │
      │  - Baseband Sample Rate: 20 MSPS                       │
      │  - Receiver Gain: 40 dB (Manual)                       │
      └───────────────────────────┬────────────────────────────┘
                                  │ USB 2.0 / 3.0
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │                 Host Computer running                  │
      │          MATLAB / Simulink Challenge Pipeline          │
      └────────────────────────────────────────────────────────┘
```

---

## 4. Hardware Calibration & Impairment Compensation

1. **Direct-Conversion DC Offset**:
   - Zero-IF SDR receivers introduce local oscillator (LO) leakage, causing a sharp spike at $0\text{ Hz}$ DC.
   - The preprocessing module (`ota/processCapturedIQ.m`) automatically subtracts the frame mean:
     $$\tilde{s}[n] = s[n] - \frac{1}{N}\sum_{k=0}^{N-1} s[k]$$
2. **Gain Staging**:
   - Avoid receiver saturation: Manual gain set to **38 dB – 42 dB** ensures good dynamic range for weak Bluetooth bursts without clipping strong Wi-Fi frames.
3. **Hardware-Independent Replay**:
   - When no physical radio is plugged into the system, `runOTATest` automatically uses stored reference I/Q captures in `data/examples/`, allowing full testing without hardware.
