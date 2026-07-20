"""
Parity and Integration test suite for `ccs_sdk.py` (`CCS_Algorithm`).
Tests multi-scale coupled windowing, flexible execution modes, and PPG/ECG/EEG feature parity.
"""

import math
import sys
from typing import List
from ccs_sdk import CCSAlgorithmSDK


def generate_synthetic_pulsatile_signal(srate: float, duration_sec: float) -> List[float]:
    """Generate a synthetic 1 Hz (~60 BPM) pulse train."""
    n_samples = int(duration_sec * srate)
    signal = []
    for i in range(n_samples):
        t = i / srate
        beat_phase = math.sin(t * 2.0 * math.pi)
        if beat_phase > 0.8:
            val = (beat_phase - 0.8) * 5.0
        else:
            val = 0.1 * beat_phase
        signal.append(val)
    return signal


def generate_synthetic_eeg_signal(srate: float, duration_sec: float, freq_hz: float) -> List[float]:
    """Generate a synthetic sine wave EEG channel."""
    n_samples = int(duration_sec * srate)
    return [math.sin(2.0 * math.pi * freq_hz * (i / srate)) for i in range(n_samples)]


def main():
    print("Initializing CCSAlgorithmSDK...")
    sdk = CCSAlgorithmSDK()
    print("SDK initialized successfully.")

    srate = 100.0
    duration = 30.0
    print(f"Generating synthetic {duration}s signals at {srate} Hz...")
    ppg_signal = generate_synthetic_pulsatile_signal(srate, duration)
    eeg_ch1 = generate_synthetic_eeg_signal(srate, duration, 10.0)  # 10 Hz alpha
    eeg_ch2 = generate_synthetic_eeg_signal(srate, duration, 15.0)  # 15 Hz beta

    # ---------------------------------------------------------
    # 1. Test Cardiac Isolated (PPG vs ECG distinctions)
    # ---------------------------------------------------------
    print("\n--- 1. Testing Isolated Cardiac Feature Extraction (PPG vs ECG) ---")
    ppg_feats = sdk.compute_cardiac_isolated(ppg_signal, srate=srate, modality="PPG")
    ecg_feats = sdk.compute_cardiac_isolated(ppg_signal, srate=srate, modality="ECG")

    print(f"PPG HR: {ppg_feats['avg_hr_bpm']:.1f} BPM, Peaks: {ppg_feats['num_peaks']}")
    print(f"PPG APG ratios - b/a: {ppg_feats['apg_b_a_ratio']:.4f}, d/a: {ppg_feats['apg_d_a_ratio']:.4f}, e/a: {ppg_feats['apg_e_a_ratio']:.4f}")
    print(f"ECG HR: {ecg_feats['avg_hr_bpm']:.1f} BPM, Peaks: {ecg_feats['num_peaks']}")
    print(f"ECG APG ratios (expect 0.0) - b/a: {ecg_feats['apg_b_a_ratio']:.4f}, d/a: {ecg_feats['apg_d_a_ratio']:.4f}, e/a: {ecg_feats['apg_e_a_ratio']:.4f}")

    assert 30.0 <= ppg_feats["avg_hr_bpm"] <= 180.0, "PPG heart rate out of bounds!"
    assert 30.0 <= ecg_feats["avg_hr_bpm"] <= 180.0, "ECG heart rate out of bounds!"
    assert ppg_feats["modality"] == "PPG"
    assert ecg_feats["modality"] == "ECG"
    assert ecg_feats["apg_b_a_ratio"] == 0.0
    assert ecg_feats["apg_d_a_ratio"] == 0.0
    assert ecg_feats["apg_e_a_ratio"] == 0.0

    # ---------------------------------------------------------
    # 2. Test Signal Quality Index (SQI)
    # ---------------------------------------------------------
    print("\n--- 2. Testing Signal Quality Index (SQI) ---")
    ppg_sqi = sdk.compute_sqi(ppg_signal, srate=srate, modality="PPG")
    ecg_sqi = sdk.compute_sqi(ppg_signal, srate=srate, modality="ECG")
    eeg_sqi = sdk.compute_sqi(eeg_ch1, srate=srate, modality="EEG")
    print(f"PPG SQI: {ppg_sqi:.3f}, ECG SQI: {ecg_sqi:.3f}, EEG SQI: {eeg_sqi:.3f}")
    assert 0.0 <= ppg_sqi <= 1.0 and 0.0 <= ecg_sqi <= 1.0 and 0.0 <= eeg_sqi <= 1.0

    # ---------------------------------------------------------
    # 3. Test Coupled Multi-Scale Session Modes
    # ---------------------------------------------------------
    print("\n--- 3. Testing Coupled Multi-Scale Session Windows & Flexible Execution Modes ---")
    base_config = {
        "enable_eeg": True,
        "enable_cardiac": True,
        "eeg_srate": srate,
        "cardiac_srate": srate,
        "eeg_subepoch_sec": 4.0,
        "cardiac_epoch_sec": 16.0,
        "macro_epoch_sec": 16.0,
        "window_step_sec": 8.0,
        "cardiac_modality": "PPG",
        "eeg_options": {
            "mode": "preprocessed",
            "start_seconds": 0.0,
            "end_seconds": 16.0,
            "bin_seconds": 4.0,
            "psd": True,
            "fooof": False,
            "acw": True,
            "irasa": False,
            "connectivity": False,
            "nonlinear": False,
            "mic": False,
            "mim": False,
            "gc": False,
            "gc_tr": False,
            "wpli": False,
            "coh": False,
            "plv": False,
            "ciplv": False,
            "pli": False,
            "remove_non_eeg": False,
        },
    }

    # Mode A: Both EEG + Cardiac
    print("Mode A: Both EEG + Cardiac...")
    res_both = sdk.compute_coupled_session(
        config=base_config,
        eeg_channels=[eeg_ch1, eeg_ch2],
        eeg_labels=["CH1", "CH2"],
        eeg_srate=srate,
        cardiac_signal=ppg_signal,
    )
    assert res_both["num_windows"] >= 1
    assert res_both["records"][0]["cardiac_features"] is not None
    assert len(res_both["records"][0]["eeg_subepochs"]) > 0
    print(f"Both Mode verified: {res_both['num_windows']} macro-windows computed with synced EEG subepochs and Cardiac features.")

    # Mode B: EEG Only (enable_cardiac = False)
    print("Mode B: EEG Only (zero cardiac overhead)...")
    config_eeg_only = dict(base_config)
    config_eeg_only["enable_cardiac"] = False
    res_eeg = sdk.compute_coupled_session(
        config=config_eeg_only,
        eeg_channels=[eeg_ch1, eeg_ch2],
        eeg_labels=["CH1", "CH2"],
        eeg_srate=srate,
        cardiac_signal=None,
    )
    assert res_eeg["num_windows"] >= 1
    assert res_eeg["records"][0]["cardiac_features"] is None
    assert len(res_eeg["records"][0]["eeg_subepochs"]) > 0
    print(f"EEG-Only Mode verified: {res_eeg['num_windows']} macro-windows computed.")

    # Mode C: Cardiac Only (enable_eeg = False)
    print("Mode C: Cardiac Only (zero EEG overhead)...")
    config_cardiac_only = dict(base_config)
    config_cardiac_only["enable_eeg"] = False
    res_cardiac = sdk.compute_coupled_session(
        config=config_cardiac_only,
        eeg_channels=None,
        eeg_labels=None,
        cardiac_signal=ppg_signal,
    )
    assert res_cardiac["num_windows"] >= 1
    assert res_cardiac["records"][0]["cardiac_features"] is not None
    assert res_cardiac["records"][0]["eeg_subepochs"] is None
    print(f"Cardiac-Only Mode verified: {res_cardiac['num_windows']} macro-windows computed.")

    print("\nSUCCESS: All Python SDK tests passed exactly as required!")


if __name__ == "__main__":
    main()
