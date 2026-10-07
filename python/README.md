# Python Reference & Research Prototype

> **NOTE FOR REVIEWERS**:  
> The canonical submission for the **MATLAB/Simulink Challenge ("Classify RF Signals Using AI")** is located in the [`matlab/`](../matlab) and [`simulink/`](../simulink) directories.  
> 
> This directory (`python/`) contains the earlier exploratory research prototype implemented in Python/PyTorch. It is retained strictly as a reference implementation and is **not required** for running the canonical MATLAB/Simulink challenge submission.

---

## Architecture Overview (Python Prototype)

The Python prototype explored edge-optimized deep learning physical-layer spectrum sensing:
- **Model**: `STFT-RADN` (~852.6k parameters) featuring Residual Dense Blocks (RDB) and CBAM (Convolutional Block Attention Module).
- **Spectrogram Dimension**: Fixed `[Batch, 1, 32, 33]` STFT log-PSD tensors.
- **Export**: ONNX runtime model (`exports/stft_radn_model.onnx`).
- **Dashboard**: Interactive visualization in Streamlit (`src/dashboard.py`).

## Key Files

- `src/01_data_generator.py`: Synthetic single-signal I/Q generator.
- `src/02_stft_converter.py`: Kaiser-windowed STFT log-PSD transformation.
- `src/models_2d.py`: PyTorch STFT-RADN and ResNet18-2D definitions.
- `src/train_2d.py`: Training engine with AdamW and ReduceLROnPlateau.
- `src/benchmark_2d.py`: ONNX export and CPU latency benchmarking.
- `src/evaluate_benchmarks.py`: Evaluation metrics across SNR levels.
- `src/dashboard.py`: Interactive Streamlit dashboard.

For full educational notes on the PyTorch prototype, see [ML_STUDENT_GUIDE.md](../docs/ML_STUDENT_GUIDE.md).
