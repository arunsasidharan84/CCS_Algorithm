use ccs_algorithm::cardiac::{extract_cardiac_features, Modality};
use ccs_algorithm::coupled::{compute_coupled_session, CoupledEpochConfig};
use ccs_algorithm::eeg::Options as EegOptions;
use ccs_algorithm::ffi::{ccs_compute_coupled_session_json, ccs_free_string};
use serde::Serialize;
use std::ffi::{CStr, CString};

#[test]
fn test_cardiac_ppg_vs_ecg_distinctions() {
    let srate = 100.0;
    // Generate 30 seconds of synthetic pulsatile signal (~60 BPM = 1 beat/sec)
    let n_samples = (30.0 * srate) as usize;
    let mut signal = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f64 / srate;
        // Synthetic pulse train with sharp peaks every 1 second
        let beat_phase = (t * 2.0 * std::f64::consts::PI).sin();
        let val = if beat_phase > 0.8 {
            (beat_phase - 0.8) * 5.0
        } else {
            0.1 * beat_phase
        };
        signal.push(val);
    }

    // Test PPG extraction
    // Test PPG extraction
    let ppg_feats = extract_cardiac_features(&signal, srate, Modality::Ppg);
    assert!(ppg_feats.avg_hr_bpm > 30.0 && ppg_feats.avg_hr_bpm < 180.0, "PPG heart rate out of bounds: {}", ppg_feats.avg_hr_bpm);
    assert_eq!(ppg_feats.modality, "PPG");
    assert!(ppg_feats.num_peaks >= 20, "Expected at least 20 peaks, got {}", ppg_feats.num_peaks);

    // Test ECG extraction
    let ecg_feats = extract_cardiac_features(&signal, srate, Modality::Ecg);
    assert!(ecg_feats.avg_hr_bpm > 30.0 && ecg_feats.avg_hr_bpm < 180.0, "ECG heart rate out of bounds: {}", ecg_feats.avg_hr_bpm);
    assert_eq!(ecg_feats.modality, "ECG");
    assert!(ecg_feats.num_peaks >= 20, "Expected at least 20 peaks, got {}", ecg_feats.num_peaks);
    // Verify ECG zeroes out PPG-specific second derivative APG ratios
    assert_eq!(ecg_feats.apg_b_a_ratio, 0.0);
    assert_eq!(ecg_feats.apg_d_a_ratio, 0.0);
    assert_eq!(ecg_feats.apg_e_a_ratio, 0.0);
}

#[test]
fn test_coupled_flexible_execution_modes() {
    let srate = 100.0;
    let duration = 30.0; // 30 seconds total
    let n_samples = (duration * srate) as usize;

    // Create 2 synthetic EEG channels and 1 cardiac channel
    let mut eeg_ch1 = vec![0.0; n_samples];
    let mut eeg_ch2 = vec![0.0; n_samples];
    let mut cardiac = vec![0.0; n_samples];
    for i in 0..n_samples {
        let t = i as f64 / srate;
        eeg_ch1[i] = (2.0 * std::f64::consts::PI * 10.0 * t).sin(); // 10 Hz alpha
        eeg_ch2[i] = (2.0 * std::f64::consts::PI * 15.0 * t).sin(); // 15 Hz beta
        cardiac[i] = if (t % 1.0) < 0.1 { 1.0 } else { 0.0 }; // 1 Hz cardiac pulse
    }

    let eeg_channels = vec![eeg_ch1.clone(), eeg_ch2.clone()];
    let eeg_labels = vec!["CH1".to_string(), "CH2".to_string()];

    let eeg_opts = EegOptions {
        mode: "preprocessed".to_string(),
        start_seconds: 0.0,
        end_seconds: 16.0,
        bin_seconds: 2.0,
        psd: true,
        fooof: false,
        acw: true,
        irasa: false,
        connectivity: false,
        nonlinear: false,
        mic: false,
        mim: false,
        gc: false,
        gc_tr: false,
        wpli: false,
        coh: false,
        plv: false,
        ciplv: false,
        pli: false,
        remove_non_eeg: false,
        non_eeg_channels: Vec::new(),
    };

    // Mode 1: Both EEG + Cardiac
    let config_both = CoupledEpochConfig {
        enable_eeg: true,
        enable_cardiac: true,
        eeg_srate: srate,
        cardiac_srate: srate,
        eeg_subepoch_sec: 2.0,
        cardiac_epoch_sec: 16.0,
        macro_epoch_sec: 16.0,
        window_step_sec: 8.0,
        cardiac_modality: "PPG".to_string(),
        eeg_options: eeg_opts.clone(),
    };
    let res_both = compute_coupled_session(
        Some(&eeg_channels),
        Some(&eeg_labels),
        Some(&cardiac),
        &config_both,
    );
    assert!(res_both.num_windows >= 1);
    assert!(res_both.records[0].cardiac_features.is_some());
    assert!(!res_both.records[0].eeg_subepochs.as_ref().unwrap().is_empty());

    // Mode 2: EegOnly
    let config_eeg = CoupledEpochConfig {
        enable_eeg: true,
        enable_cardiac: false,
        ..config_both.clone()
    };
    let res_eeg = compute_coupled_session(
        Some(&eeg_channels),
        Some(&eeg_labels),
        None,
        &config_eeg,
    );
    assert!(res_eeg.num_windows >= 1);
    assert!(res_eeg.records[0].cardiac_features.is_none());
    assert!(!res_eeg.records[0].eeg_subepochs.as_ref().unwrap().is_empty());

    // Mode 3: CardiacOnly
    let config_cardiac = CoupledEpochConfig {
        enable_eeg: false,
        enable_cardiac: true,
        ..config_both.clone()
    };
    let res_cardiac = compute_coupled_session(
        None,
        None,
        Some(&cardiac),
        &config_cardiac,
    );
    assert!(res_cardiac.num_windows >= 1);
    assert!(res_cardiac.records[0].cardiac_features.is_some());
    assert!(res_cardiac.records[0].eeg_subepochs.is_none());
}

#[derive(Serialize)]
struct EegInputPayload {
    channels: Vec<Vec<f64>>,
    labels: Option<Vec<String>>,
    srate: Option<f64>,
}

#[derive(Serialize)]
struct CardiacInputPayload {
    signal: Vec<f64>,
}

#[test]
fn test_ffi_json_interface_and_memory_cleanup() {
    let srate = 100.0;
    let n_samples = 1800; // 18 seconds
    let mut eeg_ch1 = vec![0.0; n_samples];
    let mut cardiac = vec![0.0; n_samples];
    for i in 0..n_samples {
        let t = i as f64 / srate;
        eeg_ch1[i] = (2.0 * std::f64::consts::PI * 10.0 * t).sin();
        if (t % 1.0) < 0.1 {
            cardiac[i] = 1.0;
        }
    }

    let eeg_opts = EegOptions {
        mode: "raw".to_string(),
        start_seconds: 0.0,
        end_seconds: 16.0,
        bin_seconds: 4.0,
        psd: true,
        fooof: false,
        acw: true,
        irasa: false,
        connectivity: false,
        nonlinear: false,
        mic: false,
        mim: false,
        gc: false,
        gc_tr: false,
        wpli: false,
        coh: false,
        plv: false,
        ciplv: false,
        pli: false,
        remove_non_eeg: false,
        non_eeg_channels: Vec::new(),
    };

    let config = CoupledEpochConfig {
        enable_eeg: true,
        enable_cardiac: true,
        eeg_srate: srate,
        cardiac_srate: srate,
        eeg_subepoch_sec: 4.0,
        cardiac_epoch_sec: 16.0,
        macro_epoch_sec: 16.0,
        window_step_sec: 8.0,
        cardiac_modality: "PPG".to_string(),
        eeg_options: eeg_opts,
    };

    let eeg_payload = EegInputPayload {
        channels: vec![eeg_ch1],
        labels: Some(vec!["O1".to_string()]),
        srate: Some(srate),
    };

    let cardiac_payload = CardiacInputPayload {
        signal: cardiac,
    };

    let config_json = serde_json::to_string(&config).unwrap();
    let eeg_json = serde_json::to_string(&eeg_payload).unwrap();
    let cardiac_json = serde_json::to_string(&cardiac_payload).unwrap();

    let c_config = CString::new(config_json).unwrap();
    let c_eeg = CString::new(eeg_json).unwrap();
    let c_cardiac = CString::new(cardiac_json).unwrap();

    let raw_ptr = ccs_compute_coupled_session_json(
        c_config.as_ptr(),
        c_eeg.as_ptr(),
        c_cardiac.as_ptr(),
    );
    assert!(!raw_ptr.is_null());

    let output_str = unsafe { CStr::from_ptr(raw_ptr).to_str().unwrap() };
    assert!(output_str.contains("num_windows"));

    ccs_free_string(raw_ptr);
}
