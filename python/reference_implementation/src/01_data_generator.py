"""
01_data_generator.py
Synthesizes multi-standard 2.4 GHz baseband I/Q signal streams:
1. Wi-Fi (OFDM 802.11 b/g/n-like wideband signal)
2. Bluetooth / BLE (Narrowband GFSK with frequency hopping)
3. Zigbee (802.15.4 O-QPSK DSSS spreading)
4. SmartBAN (802.15.6 Low-power medical BAN bursts)
5. Unknown / Noise (AWGN + unclassified interference)

Injects AWGN across low SNR regimes (-20 dB to 0 dB), Carrier Frequency Offsets (CFO),
and phase jitter.
"""

import os
import numpy as np

# Configuration Constants
SIGNAL_LENGTH = 1056  # Yields exactly 32 time frames with STFT nperseg=64, hop=32
FS = 20e6  # 20 MHz sampling rate
CLASSES = ["Wi-Fi", "Bluetooth", "Zigbee", "SmartBAN", "Unknown/Noise"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}


def generate_wifi_signal(length=SIGNAL_LENGTH, fs=FS):
    """Simulates a wideband OFDM Wi-Fi signal (64 subcarriers)."""
    n_subcarriers = 64
    cp_len = 16
    symbol_len = n_subcarriers + cp_len
    num_symbols = int(np.ceil(length / symbol_len)) + 1
    
    # Random QPSK symbols on active subcarriers (-26 to 26 excluding DC)
    active_idx = list(range(1, 27)) + list(range(38, 64))
    all_symbols = []
    
    for _ in range(num_symbols):
        subcarriers = np.zeros(n_subcarriers, dtype=complex)
        bits = np.random.choice([1, -1], size=(len(active_idx), 2))
        qpsk = (bits[:, 0] + 1j * bits[:, 1]) / np.sqrt(2)
        subcarriers[active_idx] = qpsk
        
        # IFFT
        time_sym = np.fft.ifft(np.fft.ifftshift(subcarriers)) * np.sqrt(n_subcarriers)
        # Add Cyclic Prefix
        cp = time_sym[-cp_len:]
        sym_with_cp = np.concatenate([cp, time_sym])
        all_symbols.append(sym_with_cp)
        
    signal = np.concatenate(all_symbols)[:length]
    return signal / np.std(signal)


def generate_bluetooth_signal(length=SIGNAL_LENGTH, fs=FS):
    """Simulates Bluetooth / BLE GFSK burst with frequency offset (hopping)."""
    # GFSK parameters: Symbol rate 1 Mbps, BT product ~0.5
    sps = int(fs / 1e6)  # 20 samples per symbol
    num_bits = int(np.ceil(length / sps)) + 5
    bits = np.random.choice([-1, 1], size=num_bits)
    
    # Upsample
    upsampled = np.repeat(bits, sps)
    # Simple Gaussian pulse filter approximation
    t = np.linspace(-2, 2, sps * 2)
    gauss_filter = np.exp(-t**2 / 0.5)
    gauss_filter /= np.sum(gauss_filter)
    filtered = np.convolve(upsampled, gauss_filter, mode='same')
    
    # Continuous phase integration (GFSK h=0.5)
    freq_dev = 250e3  # 250 kHz dev
    phase = 2 * np.pi * freq_dev * np.cumsum(filtered) / fs
    
    # Frequency hopping offset (simulate channel shift within baseband)
    hop_freq = np.random.uniform(-4e6, 4e6)
    hop_phase = 2 * np.pi * hop_freq * np.arange(len(phase)) / fs
    
    signal = np.exp(1j * (phase + hop_phase))[:length]
    return signal / np.std(signal)


def generate_zigbee_signal(length=SIGNAL_LENGTH, fs=FS):
    """Simulates IEEE 802.15.4 Zigbee O-QPSK DSSS spreading signal."""
    # 2 Mchips/s chip rate
    cps = int(fs / 2e6)  # 10 samples per chip
    num_chips = int(np.ceil(length / cps)) + 10
    
    # Pseudo-random DSSS chip sequence
    chips_i = np.random.choice([-1, 1], size=num_chips)
    chips_q = np.random.choice([-1, 1], size=num_chips)
    
    # O-QPSK offset: half chip delay on Q branch
    upsampled_i = np.repeat(chips_i, cps)
    upsampled_q = np.repeat(chips_q, cps)
    
    half_chip = cps // 2
    upsampled_q = np.roll(upsampled_q, half_chip)
    
    # Half-sine pulse shaping
    t = np.linspace(0, 1, cps)
    pulse = np.sin(np.pi * t)
    
    i_shaped = np.convolve(upsampled_i, pulse, mode='same')
    q_shaped = np.convolve(upsampled_q, pulse, mode='same')
    
    signal = (i_shaped + 1j * q_shaped)[:length]
    return signal / np.std(signal)


def generate_smartban_signal(length=SIGNAL_LENGTH, fs=FS):
    """Simulates SmartBAN (IEEE 802.15.6) duty-cycled narrowband pulse burst."""
    sps = int(fs / 500e3)  # 500 kbps, 40 samples per symbol
    num_bits = int(np.ceil(length / sps)) + 2
    bits = np.random.choice([-1, 1], size=num_bits)
    
    # Duty cycle envelope (pulse burst pattern)
    burst_mask = np.zeros(length)
    burst_len = int(length * np.random.uniform(0.3, 0.7))
    start_idx = np.random.randint(0, max(1, length - burst_len))
    burst_mask[start_idx:start_idx + burst_len] = 1.0
    
    upsampled = np.repeat(bits, sps)[:length]
    freq_dev = 150e3
    phase = 2 * np.pi * freq_dev * np.cumsum(upsampled) / fs
    
    carrier_shift = np.random.uniform(-2e6, 2e6)
    carr_phase = 2 * np.pi * carrier_shift * np.arange(length) / fs
    
    signal = np.exp(1j * (phase + carr_phase)) * burst_mask
    std = np.std(signal)
    if std > 1e-6:
        signal /= std
    return signal


def generate_noise_signal(length=SIGNAL_LENGTH, fs=FS):
    """Simulates pure AWGN or unclassified non-stationary noise/chirp interference."""
    if np.random.rand() > 0.5:
        # Pure AWGN
        noise = (np.random.randn(length) + 1j * np.random.randn(length)) / np.sqrt(2)
    else:
        # Linear Chirp interference + AWGN
        f0 = np.random.uniform(-8e6, -2e6)
        f1 = np.random.uniform(2e6, 8e6)
        t = np.linspace(0, length / fs, length)
        chirp = np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t**2))
        awgn = (np.random.randn(length) + 1j * np.random.randn(length)) / np.sqrt(2)
        noise = 0.7 * chirp + 0.3 * awgn
    return noise / np.std(noise)


def inject_awgn_and_cfo(signal, snr_db, fs=FS):
    """Injects AWGN noise corresponding to specified SNR (dB) and random CFO."""
    # Signal power
    sig_power = np.mean(np.abs(signal)**2)
    if sig_power == 0:
        sig_power = 1e-6
    
    # Calculate noise power for target SNR
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    # Complex AWGN
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * np.sqrt(noise_power / 2.0)
    
    # Carrier Frequency Offset (CFO) & Phase Noise
    cfo = np.random.uniform(-50e3, 50e3)  # +/- 50 kHz CFO
    t = np.arange(len(signal)) / fs
    cfo_phase = np.exp(1j * (2 * np.pi * cfo * t + np.random.uniform(0, 2*np.pi)))
    
    noisy_signal = (signal * cfo_phase) + noise
    return noisy_signal


def generate_dataset(num_samples_per_class=1000, snr_range=(-20, 0), save_dir="data/raw_iq"):
    """
    Generates dataset of raw I/Q samples across classes and SNRs.
    
    Returns:
        raw_iq: ndarray [N, SIGNAL_LENGTH] complex128
        labels: ndarray [N] int64
        snrs: ndarray [N] float32
    """
    os.makedirs(save_dir, exist_ok=True)
    
    generators = [
        generate_wifi_signal,
        generate_bluetooth_signal,
        generate_zigbee_signal,
        generate_smartban_signal,
        generate_noise_signal
    ]
    
    iq_list = []
    label_list = []
    snr_list = []
    
    snr_values = np.linspace(snr_range[0], snr_range[1], 11)  # -20, -18, ..., 0 dB
    
    print(f"[INFO] Generating raw I/Q signal samples ({num_samples_per_class} per class)...")
    for cls_idx, gen_func in enumerate(generators):
        cls_name = CLASSES[cls_idx]
        print(f"  -> Generating Class {cls_idx}: {cls_name}")
        for i in range(num_samples_per_class):
            clean_sig = gen_func()
            snr = float(np.random.choice(snr_values))
            noisy_sig = inject_awgn_and_cfo(clean_sig, snr)
            
            iq_list.append(noisy_sig)
            label_list.append(cls_idx)
            snr_list.append(snr)
            
    raw_iq = np.array(iq_list, dtype=np.complex128)
    labels = np.array(label_list, dtype=np.int64)
    snrs = np.array(snr_list, dtype=np.float32)
    
    out_file = os.path.join(save_dir, "raw_iq_dataset.npz")
    np.savez_compressed(out_file, raw_iq=raw_iq, labels=labels, snrs=snrs, class_names=CLASSES)
    print(f"[SUCCESS] Saved raw I/Q dataset to {out_file} (Total: {len(labels)} samples)")
    return raw_iq, labels, snrs


if __name__ == "__main__":
    generate_dataset(num_samples_per_class=1000)
