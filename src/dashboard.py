"""
dashboard.py
Ultra-Modern Management Dashboard UI for 2.4 GHz RF Spectrum Monitoring:
- Clean, professional typography (Google Font 'Plus Jakarta Sans').
- Zero emojis - uses clean SVG line art and modern Dribbble-inspired UI components.
- Lime Green (#dcf763), Dark Charcoal (#141518), and Soft Cream (#e9edec) color scheme.
- Fixed empty card wrappers and sidebar selectbox / button text contrast.
"""

import os
import time
import importlib
import numpy as np
import scipy.signal as signal
import streamlit as st
import matplotlib.pyplot as plt

# Import data generator and ONNX session exporter
data_gen = importlib.import_module("01_data_generator")
stft_conv = importlib.import_module("02_stft_converter")
import onnxruntime as ort

CLASSES = data_gen.CLASSES
FS = data_gen.FS
SIGNAL_LENGTH = data_gen.SIGNAL_LENGTH

# Page Configuration
st.set_page_config(
    page_title="2.4 GHz RF Spectrum Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Light & Lime Theme matching reference design
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Main Background & Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #e9edec !important;
        color: #141518 !important;
    }
    
    /* Hide Default Streamlit Chrome Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1500px !important;
    }

    /* Top Title Header Styling */
    .main-header-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #141518;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }

    /* Pill Navigation / Filter Tabs */
    .tab-pill-container {
        display: flex;
        gap: 8px;
        margin-top: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    
    .tab-pill {
        padding: 10px 20px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.88rem;
        cursor: pointer;
        background-color: #ffffff;
        color: #525866;
        border: 1px solid #e1e6e4;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .tab-pill.active {
        background-color: #141518;
        color: #ffffff;
        border-color: #141518;
        box-shadow: 0 4px 12px rgba(20,21,24,0.15);
    }

    /* Cards Layout Base */
    .custom-card {
        background-color: #ffffff;
        border-radius: 24px;
        padding: 24px;
        border: 1px solid #e1e6e4;
        box-shadow: 0 4px 16px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }

    .custom-card-lime {
        background-color: #dcf763;
        border: 1px solid #cbe64d;
        color: #141518;
    }

    .custom-card-dark {
        background-color: #141518;
        color: #ffffff;
        border: 1px solid #141518;
    }

    /* Card Metrics Styling */
    .card-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #687082;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .card-label-dark {
        color: #9da4b4;
    }

    .metric-value-large {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.0;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }

    .metric-subtitle {
        font-size: 0.85rem;
        font-weight: 500;
        color: #525866;
    }

    /* Status Pill Badges */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        background-color: rgba(20,21,24,0.08);
        color: #141518;
    }

    .status-pill-lime {
        background-color: #141518;
        color: #dcf763;
    }

    /* =========================================================
       CUSTOM SIDEBAR STYLING FIXES (TEXT & BUTTON CONTRAST)
       ========================================================= */
    section[data-testid="stSidebar"] {
        background-color: #141518 !important;
        border-right: none !important;
        padding: 1.2rem 1rem !important;
    }
    
    /* Sidebar Labels */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #9da4b4 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }

    /* Selectbox Input Container */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #202227 !important;
        border: 1px solid #2e3138 !important;
        border-radius: 14px !important;
        color: #ffffff !important;
    }

    /* Selectbox Selected Value Text */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Dropdown Arrow Icon */
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #dcf763 !important;
        color: #dcf763 !important;
    }

    /* Sidebar Slider Text */
    section[data-testid="stSidebar"] div[data-testid="stSliderTickBarMinMax"] span,
    section[data-testid="stSidebar"] div[data-testid="stSliderThumbValue"] {
        color: #dcf763 !important;
        font-weight: 700 !important;
    }

    /* Sidebar Button Styling */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #dcf763 !important;
        color: #141518 !important;
        border: 1px solid #cbe64d !important;
        border-radius: 30px !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 12px rgba(220, 247, 99, 0.25) !important;
        width: 100% !important;
        transition: transform 0.15s ease !important;
    }
    
    section[data-testid="stSidebar"] div.stButton > button * {
        color: #141518 !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        transform: translateY(-2px) !important;
        background-color: #e5f979 !important;
    }

    .sidebar-logo-box {
        background-color: #202227;
        border-radius: 20px;
        padding: 18px;
        text-align: center;
        margin-bottom: 24px;
        border: 1px solid #2e3138;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .sidebar-logo-icon {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background-color: #2e3138;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }

    /* Custom Progress Bar Meters */
    .meter-container {
        margin-bottom: 14px;
    }

    .meter-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .meter-track {
        height: 10px;
        background-color: #f0f3f2;
        border-radius: 10px;
        overflow: hidden;
    }

    .meter-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.4s ease;
    }

    /* Resource Action Cards */
    .resource-card {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 18px;
        border: 1px solid #e1e6e4;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.15s ease;
    }

    .resource-icon-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #f4f6f5;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Main Content Area Buttons */
    div.stButton > button {
        background-color: #ffffff !important;
        color: #141518 !important;
        border: 1px solid #e1e6e4 !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_onnx_session(onnx_path="exports/stft_radn_model.onnx"):
    """Loads ONNX Runtime session."""
    if not os.path.exists(onnx_path):
        benchmark_mod = importlib.import_module("benchmark_2d")
        benchmark_mod.export_onnx(onnx_out_path=onnx_path)
        
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def generate_synthetic_iq(class_idx, snr_db):
    """Generates synthetic 1D complex I/Q signal for selected class and SNR."""
    generators = [
        data_gen.generate_wifi_signal,
        data_gen.generate_bluetooth_signal,
        data_gen.generate_zigbee_signal,
        data_gen.generate_smartban_signal,
        data_gen.generate_noise_signal
    ]
    clean_sig = generators[class_idx]()
    noisy_sig = data_gen.inject_awgn_and_cfo(clean_sig, snr_db)
    return noisy_sig


def compute_stft_tensor(iq_signal):
    """Computes normalized log-PSD STFT tensor [1, 1, 32, 33]."""
    kw = signal.windows.kaiser(64, beta=14.0)
    _, _, Zxx = signal.stft(
        iq_signal,
        fs=FS,
        window=kw,
        nperseg=64,
        noverlap=32,
        boundary=None,
        padded=False,
        return_onesided=False
    )
    psd_mag = np.abs(Zxx[:33, :]).T  # shape: (32, 33)
    log_psd = 10.0 * np.log10(psd_mag**2 + 1e-12)
    
    mean_val = np.mean(log_psd)
    std_val = np.std(log_psd) if np.std(log_psd) > 1e-6 else 1e-6
    norm_psd = (log_psd - mean_val) / std_val
    
    tensor = norm_psd.reshape(1, 1, 32, 33).astype(np.float32)
    return tensor, norm_psd


def main():
    # Load ONNX Session
    session = load_onnx_session()
    input_name = session.get_inputs()[0].name

    # -------------------------------------------------------------
    # Sidebar Controls (Dark Navigation Bar)
    # -------------------------------------------------------------
    st.sidebar.markdown("""
    <div class="sidebar-logo-box">
        <div class="sidebar-logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#dcf763" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2v20M2 12h20M4.93 4.93l14.14 14.14M4.93 19.07l14.14-14.14"/>
            </svg>
        </div>
        <div style="font-weight: 800; font-size: 1.05rem; letter-spacing: -0.02em; color: #ffffff;">RF MONITOR</div>
        <div style="font-size: 0.75rem; color: #9da4b4; margin-top: 2px;">STFT-RADN Engine v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #9da4b4; margin-bottom: 12px;'>SIMULATION PARAMETERS</div>", unsafe_allow_html=True)
    
    selected_class_name = st.sidebar.selectbox(
        "Target Signal Class",
        options=CLASSES,
        index=0
    )
    selected_class_idx = CLASSES.index(selected_class_name)

    snr_db = st.sidebar.slider(
        "SNR Noise Level (dB)",
        min_value=-20.0,
        max_value=0.0,
        value=-10.0,
        step=1.0
    )

    st.sidebar.markdown("<div style='margin-top: 16px; margin-bottom: 16px;'></div>", unsafe_allow_html=True)
    resample_btn = st.sidebar.button("Re-Synthesize Burst", use_container_width=True)

    # State Persistence
    if "iq_signal" not in st.session_state or resample_btn or "last_class" not in st.session_state or st.session_state["last_class"] != selected_class_idx or st.session_state.get("last_snr") != snr_db:
        iq_sig = generate_synthetic_iq(selected_class_idx, snr_db)
        st.session_state["iq_signal"] = iq_sig
        st.session_state["last_class"] = selected_class_idx
        st.session_state["last_snr"] = snr_db
    else:
        iq_sig = st.session_state["iq_signal"]

    # Compute STFT & ONNX Inference
    tensor_input, spectrogram_2d = compute_stft_tensor(iq_sig)
    
    start_t = time.perf_counter()
    logits = session.run(None, {input_name: tensor_input})[0][0]
    latency_ms = (time.perf_counter() - start_t) * 1000.0
    
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    pred_idx = int(np.argmax(probs))
    pred_class_name = CLASSES[pred_idx]
    confidence_pct = probs[pred_idx] * 100.0

    # -------------------------------------------------------------
    # Main Header & Title Banner
    # -------------------------------------------------------------
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 4px;">
        <div>
            <div class="main-header-title">
                Monitoring Spectrum and Workflows
            </div>
            <div style="color: #687082; font-size: 0.95rem; font-weight: 500;">
                Edge-Optimized Physical-Layer Deep Learning Signal Sensing & Interference Engine
            </div>
        </div>
        <div style="display: flex; gap: 12px; margin-top: 10px;">
            <div style="background: #ffffff; color: #141518; padding: 10px 18px; border-radius: 30px; font-weight: 700; font-size: 0.85rem; border: 1px solid #e1e6e4; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                ONNX CPU Engine
            </div>
            <div style="background: #141518; color: #dcf763; padding: 10px 20px; border-radius: 30px; font-weight: 700; font-size: 0.85rem;">
                Telemetry Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Sub-header Protocol Tab Pills
    # -------------------------------------------------------------
    st.markdown('<div class="tab-pill-container">', unsafe_allow_html=True)
    cols = st.columns(len(CLASSES))
    for idx, c_name in enumerate(CLASSES):
        is_active = (c_name == selected_class_name)
        if cols[idx].button(f"{c_name}", key=f"tab_{idx}"):
            st.session_state["last_class"] = idx
            st.rerun()

    # -------------------------------------------------------------
    # Top 3 Hero Cards (Lime Green Accent, Off-White, Dark Charcoal)
    # -------------------------------------------------------------
    c1, c2, c3 = st.columns([1, 1, 1.1])

    with c1:
        st.markdown(f"""
        <div class="custom-card custom-card-lime">
            <div class="card-label">
                <span>PREDICTED STANDARD</span>
                <span class="status-pill status-pill-lime">{confidence_pct:.1f}% CONFIDENCE</span>
            </div>
            <div class="metric-value-large">{pred_class_name}</div>
            <div class="metric-subtitle">
                Target Ground Truth: <strong>{selected_class_name}</strong>
            </div>
            <div style="margin-top: 18px; height: 8px; background: rgba(20,21,24,0.12); border-radius: 10px;">
                <div style="height: 100%; width: {confidence_pct:.1f}%; background: #141518; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-label">
                <span>SIGNAL-TO-NOISE RATIO</span>
                <span class="status-pill">{snr_db:.0f} dB SNR</span>
            </div>
            <div class="metric-value-large">{snr_db:.1f} <span style="font-size: 1.2rem; font-weight: 600; color: #687082;">dB</span></div>
            <div class="metric-subtitle">
                AWGN Low-Noise Regime: <strong>{-20.0:.0f} to 0.0 dB</strong>
            </div>
            <div style="margin-top: 18px; height: 8px; background: #f0f3f2; border-radius: 10px;">
                <div style="height: 100%; width: {((snr_db + 20) / 20) * 100:.1f}%; background: #dcf763; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="custom-card custom-card-dark">
            <div class="card-label card-label-dark">
                <span>STFT-RADN LATENCY</span>
                <span style="color: #dcf763; font-weight: 700; font-size: 0.8rem;">~10.6 ms TARGET</span>
            </div>
            <div class="metric-value-large" style="color: #ffffff;">{latency_ms:.2f} <span style="font-size: 1.2rem; font-weight: 600; color: #dcf763;">ms</span></div>
            <div class="metric-subtitle" style="color: #9da4b4;">
                Model Parameters: <strong>852,653</strong> (~852.6k)
            </div>
            <div style="margin-top: 18px; height: 8px; background: #2a2c32; border-radius: 10px;">
                <div style="height: 100%; width: {(latency_ms / 10.6) * 100:.1f}%; background: #dcf763; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Main Middle Section: Spectrogram/Waveform + Class Meters & Cards
    # -------------------------------------------------------------
    col_left, col_right = st.columns([1.75, 1.0], gap="medium")

    with col_left:
        # Header Box
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; margin-top: 10px;">
            <div style="font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; color: #141518;">
                Spectrum Statistics and 2D STFT Spectrogram
            </div>
            <div style="font-size: 0.82rem; font-weight: 600; color: #687082; background: #ffffff; padding: 4px 14px; border-radius: 20px; border: 1px solid #e1e6e4;">
                Shape: [32, 33]
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render Clean Modern Matplotlib Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), gridspec_kw={'height_ratios': [2.4, 1.0]})
        fig.patch.set_facecolor('#ffffff')
        
        # Spectrogram Heatmap
        ax1.set_facecolor('#ffffff')
        im = ax1.imshow(
            spectrogram_2d.T,
            aspect='auto',
            origin='lower',
            cmap='inferno',
            extent=[0, 32, -10, 10]
        )
        ax1.set_title(f"Log-PSD Spectrogram (Kaiser Window β=14.0) | {selected_class_name} ({snr_db:.0f} dB)", color='#141518', fontsize=11, fontweight='bold', pad=10)
        ax1.set_xlabel("Time Frames (32)", color='#687082', fontsize=9, fontweight='medium')
        ax1.set_ylabel("Frequency Bins (33)", color='#687082', fontsize=9, fontweight='medium')
        ax1.tick_params(colors='#141518', labelsize=8)
        
        for spine in ax1.spines.values():
            spine.set_color('#e1e6e4')

        cbar = fig.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
        cbar.ax.tick_params(colors='#687082', labelsize=8)
        cbar.set_label('Normalized PSD (dB)', color='#687082', fontsize=8)

        # 1D I/Q Time Domain Waveform Plot
        ax2.set_facecolor('#ffffff')
        t_us = np.arange(len(iq_sig)) / (FS / 1e6)
        ax2.plot(t_us, iq_sig.real, label="I Channel", color="#141518", linewidth=1.2, alpha=0.9)
        ax2.plot(t_us, iq_sig.imag, label="Q Channel", color="#b0d418", linewidth=1.2, alpha=0.85)
        ax2.set_title("1D Complex Baseband Waveform (N = 1056 samples)", color='#141518', fontsize=10, fontweight='bold', pad=8)
        ax2.set_xlabel("Time (µs)", color='#687082', fontsize=9)
        ax2.set_ylabel("Amplitude", color='#687082', fontsize=9)
        ax2.tick_params(colors='#141518', labelsize=8)
        ax2.legend(loc="upper right", facecolor="#ffffff", edgecolor="#e1e6e4", labelcolor="#141518", fontsize=8)
        
        for spine in ax2.spines.values():
            spine.set_color('#e1e6e4')

        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.markdown("""
        <div style="font-size: 1.2rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 12px; margin-top: 10px; color: #141518;">
            Classification Probabilities
        </div>
        """, unsafe_allow_html=True)

        for idx, c_name in enumerate(CLASSES):
            prob = float(probs[idx])
            pct = prob * 100.0
            is_top = (idx == pred_idx)
            
            bg_color = "#dcf763" if is_top else "#141518"
            text_style = "font-weight: 800; color: #141518;" if is_top else "font-weight: 600; color: #525866;"
            
            st.markdown(f"""
            <div class="meter-container" style="background: #ffffff; padding: 12px 18px; border-radius: 16px; border: 1px solid #e1e6e4;">
                <div class="meter-header" style="{text_style}">
                    <span>{c_name}</span>
                    <span>{pct:.1f}%</span>
                </div>
                <div class="meter-track">
                    <div class="meter-fill" style="width: {pct:.1f}%; background-color: {bg_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Resource Action Cards with Clean SVG Arrows
        st.markdown("""
        <div class="resource-card" style="margin-top: 18px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div class="resource-icon-circle">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#141518" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/><path d="M12 6v6l4 2"/>
                    </svg>
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #141518;">STFT-RADN Architecture</div>
                    <div style="font-size: 0.78rem; color: #687082;">Residual Dense + CBAM Attention</div>
                </div>
            </div>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#141518" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <line x1="7" y1="17" x2="17" y2="7"></line>
                <polyline points="7 7 17 7 17 17"></polyline>
            </svg>
        </div>
        
        <div class="resource-card">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div class="resource-icon-circle">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#141518" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                    </svg>
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #141518;">ONNX Runtime Performance</div>
                    <div style="font-size: 0.78rem; color: #687082;">3.12 ms Mean CPU Latency</div>
                </div>
            </div>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#141518" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <line x1="7" y1="17" x2="17" y2="7"></line>
                <polyline points="7 7 17 7 17 17"></polyline>
            </svg>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
