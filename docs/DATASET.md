# RF Dataset Design & Overlapping Signal Scenarios

**Project**: Classify RF Signals Using AI (MATLAB/Simulink Challenge Submission)  
**Pipeline**: MATLAB Standards-Oriented RF Waveform Synthesis

---

## 1. Overview & Dataset Philosophy

Traditional modulation datasets (such as RadioML 2016/2018) only evaluate isolated transmissions where each sample contains a single emitter. In the unlicenced 2.4 GHz ISM band, multiple protocols share the same physical medium simultaneously.

The canonical MATLAB dataset generator (`matlab/data_generation/generateDataset.m`) synthesizes multi-signal mixtures where signals can be **isolated, spectrally adjacent, partially overlapping, or heavily collided**.

---

## 2. Supported Wireless Standards & Physical Layer Models

| Protocol | Physical Layer Standard | Modulation & PHY Details | Bandwidth / Spacing |
| :--- | :--- | :--- | :--- |
| **Wi-Fi** | IEEE 802.11a/g/n | 64-subcarrier OFDM, 52 active data/pilot tones, 16-sample cyclic prefix | ~16.6 MHz nominal BW |
| **Bluetooth** | Bluetooth Core v5.3 / BLE | GFSK ($BT=0.5$), symbol rate 1 Msps, frequency hopping carrier shifts | 1.0 MHz nominal channel |
| **Zigbee** | IEEE 802.15.4-2020 | O-QPSK with half-sine pulse shaping, 2.0 Mchips/s DSSS chip rate | 2.0 MHz nominal channel |
| **SmartBAN** | IEEE 802.15.6 | Duty-cycled pulse bursts (30% to 70% active duration), symbol rate 500 ksps | 0.5–1.0 MHz channel |
| **Noise / Chirp** | Unclassified Interference | Complex AWGN thermal noise + linear frequency-swept chirps | Full band sweeps |

---

## 3. Co-existence & Overlap Scenarios

Every sample in the canonical dataset is generated from one of 8 structured co-existence scenarios:

| Scenario ID | Name | Composition | Description |
| :--- | :--- | :--- | :--- |
| **S1** | `S1_WiFi_Only` | Wi-Fi | Clean single-protocol baseline at baseband center. |
| **S2** | `S2_Bluetooth_Only` | Bluetooth | Narrowband GFSK burst with random channel hopping offset. |
| **S3** | `S3_WiFi_BT_Adjacent` | Wi-Fi + BT Adjacent | Bluetooth placed outside Wi-Fi main lobe ($\pm 7.5\text{ MHz}$). |
| **S4** | `S4_WiFi_BT_PartialOverlap` | Wi-Fi + BT Overlap | Bluetooth hopping directly into Wi-Fi shoulder ($\pm 4\text{ MHz}$). |
| **S5** | `S5_WiFi_Zigbee_Overlap` | Wi-Fi + Zigbee | Zigbee DSSS transmission colliding inside Wi-Fi band. |
| **S6** | `S6_BT_Zigbee_Overlap` | BT + Zigbee Heavy Overlap | Narrowband collision where BT and Zigbee share carrier frequencies. |
| **S7** | `S7_WiFi_BT_Zigbee_Simultaneous` | 3 Simultaneous Protocols | Crowded channel with simultaneous Wi-Fi, BT, and Zigbee activity. |
| **S8** | `S8_Crowded_Spectrum_4Signals` | 4 Simultaneous Protocols | Crowded spectrum containing Wi-Fi, BT, Zigbee, and SmartBAN. |

---

## 4. Multi-Hot Ground Truth Representation

Because a single RF sample can contain multiple concurrent emitters, ground truth is formulated as a 5-element multi-hot binary vector:

$$\mathbf{y} = [y_{\text{WiFi}}, y_{\text{BT}}, y_{\text{Zigbee}}, y_{\text{SmartBAN}}, y_{\text{Noise}}] \in \{0, 1\}^5$$

For example:
- **Scenario S1 (Wi-Fi Only)**: `[1, 0, 0, 0, 0]`
- **Scenario S4 (Wi-Fi + BT Overlap)**: `[1, 1, 0, 0, 0]`
- **Scenario S7 (Wi-Fi + BT + Zigbee)**: `[1, 1, 1, 0, 0]`
- **Scenario S8 (Crowded 4 Signals)**: `[1, 1, 1, 1, 0]`

---

## 5. Channel Impairments & Realistic Effects

1. **Signal-to-Noise Ratio (SNR)**:
   - Evaluated across **-20 dB to 0 dB** in 2 dB increments to benchmark performance in harsh RF conditions.
2. **Carrier Frequency Offset (CFO)**:
   - Random Doppler / oscillator misalignment: $\Delta f \in [-50\text{ kHz}, +50\text{ kHz}]$ with uniform initial phase $\phi_0 \in [0, 2\pi)$.
3. **Multipath Fading**:
   - Optional 3-tap frequency-selective channel model with discrete propagation delays and Rayleigh amplitudes.

---

## 6. Dataset Generation Modes

The dataset generation engine (`generateDataset.m`) supports three reproducible operational modes:

| Mode | Total Samples | Purpose | Typical Execution Time |
| :--- | :--- | :--- | :--- |
| **`smoke`** | 120 samples | Fast CI / local environment verification | ~3 seconds |
| **`development`** | 800 samples | Rapid training and iteration | ~15 seconds |
| **`full`** | 4,000 samples | Canonical benchmark and rigorous evaluation | ~70 seconds |
