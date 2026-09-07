"""
02_stft_converter.py
Transforms 1D I/Q complex time-series signals into 2D Power Spectral Density (PSD)
spectrograms using Kaiser-windowed STFT.

Strictly enforces output tensor shape of [Batch, 1, 32, 33].
Caches the dataset to `data/stft_dataset.npz` to maximize PyTorch DataLoader efficiency.
"""

import os
import numpy as np
import scipy.signal as signal
from tqdm import tqdm

# Import generator if raw dataset needs to be generated on the fly
import importlib
data_gen = importlib.import_module("01_data_generator")
generate_dataset = data_gen.generate_dataset
SIGNAL_LENGTH = data_gen.SIGNAL_LENGTH
FS = data_gen.FS
CLASSES = data_gen.CLASSES


def convert_iq_to_stft(raw_iq, nperseg=64, noverlap=32, kaiser_beta=14.0):
    """
    Converts 1D I/Q signals to 2D log-PSD STFT Spectrograms.
    
    Args:
        raw_iq: ndarray [N, 1056] complex128/64
    
    Returns:
        stft_tensors: ndarray [N, 1, 32, 33] float32
    """
    N = raw_iq.shape[0]
    stft_tensors = np.zeros((N, 1, 32, 33), dtype=np.float32)
    
    window = signal.windows.kaiser(nperseg, beta=kaiser_beta)
    
    print(f"[INFO] Computing Kaiser-windowed STFT spectrograms for {N} samples...")
    for i in tqdm(range(N), desc="STFT Conversion"):
        iq_sample = raw_iq[i]
        
        # Compute STFT without zero-padding boundaries to ensure exactly 32 time frames
        _, _, Zxx = signal.stft(
            iq_sample,
            fs=FS,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
            padded=False,
            return_onesided=False
        )
        
        # Zxx has shape (64, 32). Select first 33 frequency channels and transpose to (32, 33)
        psd_mag = np.abs(Zxx[:33, :]).T  # shape: (32, 33)
        
        # Convert to Log-scale Power Spectral Density (dB)
        log_psd = 10.0 * np.log10(psd_mag**2 + 1e-12)
        
        # Standardize / Normalize sample (zero-mean, unit variance)
        mean_val = np.mean(log_psd)
        std_val = np.std(log_psd)
        if std_val < 1e-6:
            std_val = 1e-6
        norm_psd = (log_psd - mean_val) / std_val
        
        stft_tensors[i, 0, :, :] = norm_psd.astype(np.float32)
        
    return stft_tensors


def create_and_cache_stft_dataset(raw_iq_path="data/raw_iq/raw_iq_dataset.npz", out_path="data/stft_dataset.npz", num_samples_per_class=1000):
    """Loads or generates raw I/Q data, converts to STFT spectrograms [N, 1, 32, 33], and caches npz."""
    if not os.path.exists(raw_iq_path):
        print(f"[WARN] Raw I/Q dataset not found at {raw_iq_path}. Generating now...")
        raw_iq, labels, snrs = generate_dataset(num_samples_per_class=num_samples_per_class)
    else:
        print(f"[INFO] Loading raw I/Q dataset from {raw_iq_path}...")
        data = np.load(raw_iq_path)
        raw_iq = data['raw_iq']
        labels = data['labels']
        snrs = data['snrs']
        
    stft_tensors = convert_iq_to_stft(raw_iq)
    
    # Verify strict shape requirement
    assert stft_tensors.shape[1:] == (1, 32, 33), f"STFT shape mismatch! Expected (1, 32, 33), got {stft_tensors.shape[1:]}"
    print(f"[ASSERT OK] All output tensors strictly adhere to shape [Batch, 1, 32, 33]")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        X=stft_tensors,
        y=labels,
        snrs=snrs,
        class_names=CLASSES
    )
    print(f"[SUCCESS] Saved cached STFT dataset to {out_path} (Shape: {stft_tensors.shape})")
    return stft_tensors, labels, snrs


if __name__ == "__main__":
    create_and_cache_stft_dataset()
