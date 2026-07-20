"""
CCS_Algorithm Python SDK (`ccs_sdk.py`).

Provides Pythonic bindings to the high-performance Rust `CCS_Algorithm` library via ctypes and JSON exchange.
Supports modular and flexible execution:
- Coupled multi-scale windowing (EEG + Cardiac simultaneously)
- Isolated EEG feature & connectivity extraction
- Isolated Cardiac (PPG / ECG) peak detection, HRV, and APG morphology
- Signal Quality Index (SQI) assessment for EEG, PPG, and ECG
"""

import ctypes
import json
import os
import platform
import sys
from typing import Dict, List, Optional, Union, Any


class CCSAlgorithmError(Exception):
    """Exception raised when C ABI execution fails or returns invalid JSON."""
    pass


class CCSAlgorithmSDK:
    """Main wrapper class for the `CCS_Algorithm` dynamic library."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self._find_dynamic_library()
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find CCS_Algorithm library at: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        self._setup_ffi_signatures()

    @staticmethod
    def _find_dynamic_library() -> str:
        system = platform.system().lower()
        if system == "darwin":
            lib_name = "libccs_algorithm.dylib"
        elif system == "windows":
            lib_name = "ccs_algorithm.dll"
        else:
            lib_name = "libccs_algorithm.so"

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(curr_dir, "..", "target", "release", lib_name),
            os.path.join(curr_dir, "..", "target", "debug", lib_name),
            os.path.join(curr_dir, lib_name),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return candidates[0]  # Fallback return for explicit error message

    def _setup_ffi_signatures(self):
        # ccs_compute_coupled_session_json
        self.lib.ccs_compute_coupled_session_json.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.lib.ccs_compute_coupled_session_json.restype = ctypes.c_void_p

        # ccs_compute_eeg_isolated_json
        self.lib.ccs_compute_eeg_isolated_json.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.lib.ccs_compute_eeg_isolated_json.restype = ctypes.c_void_p

        # ccs_compute_cardiac_isolated_json
        self.lib.ccs_compute_cardiac_isolated_json.argtypes = [
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_char_p,
        ]
        self.lib.ccs_compute_cardiac_isolated_json.restype = ctypes.c_void_p

        # ccs_compute_sqi_json
        self.lib.ccs_compute_sqi_json.argtypes = [
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_char_p,
        ]
        self.lib.ccs_compute_sqi_json.restype = ctypes.c_void_p

        # ccs_free_string
        self.lib.ccs_free_string.argtypes = [ctypes.c_void_p]
        self.lib.ccs_free_string.restype = None

    def _call_and_parse(self, func_name: str, *c_args) -> Dict[str, Any]:
        func = getattr(self.lib, func_name)
        raw_ptr = func(*c_args)
        if not raw_ptr:
            raise CCSAlgorithmError(f"Function {func_name} returned null pointer")

        try:
            c_str = ctypes.cast(raw_ptr, ctypes.c_char_p).value
            if c_str is None:
                raise CCSAlgorithmError(f"Function {func_name} returned null string value")
            decoded = c_str.decode("utf-8")
            return json.loads(decoded)
        finally:
            self.lib.ccs_free_string(raw_ptr)

    def compute_coupled_session(
        self,
        config: Dict[str, Any],
        eeg_channels: Optional[List[List[float]]] = None,
        eeg_labels: Optional[List[str]] = None,
        eeg_srate: Optional[float] = None,
        cardiac_signal: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute coupled multi-scale windowing features across an entire session.
        Highly flexible: can extract both EEG + Cardiac simultaneously (`enable_eeg=True, enable_cardiac=True`),
        or EEG-only (`enable_cardiac=False`), or Cardiac-only (`enable_eeg=False`).
        """
        c_config = json.dumps(config).encode("utf-8")

        if eeg_channels is not None:
            eeg_payload = {
                "channels": eeg_channels,
                "labels": eeg_labels,
                "srate": eeg_srate,
            }
            c_eeg = json.dumps(eeg_payload).encode("utf-8")
        else:
            c_eeg = None

        if cardiac_signal is not None:
            cardiac_payload = {"signal": cardiac_signal}
            c_cardiac = json.dumps(cardiac_payload).encode("utf-8")
        else:
            c_cardiac = None

        return self._call_and_parse(
            "ccs_compute_coupled_session_json",
            c_config,
            c_eeg,
            c_cardiac,
        )

    def compute_eeg_isolated(
        self,
        channels: List[List[float]],
        options: Dict[str, Any],
        labels: Optional[List[str]] = None,
        srate: float = 250.0,
    ) -> Dict[str, Any]:
        """Compute isolated EEG features (spectral, non-linear, connectivity, ACW, IRASA)."""
        eeg_payload = {
            "channels": channels,
            "labels": labels,
            "srate": srate,
        }
        c_eeg = json.dumps(eeg_payload).encode("utf-8")
        c_options = json.dumps(options).encode("utf-8")
        return self._call_and_parse("ccs_compute_eeg_isolated_json", c_eeg, c_options)

    def compute_cardiac_isolated(
        self,
        signal: List[float],
        srate: float = 100.0,
        modality: str = "PPG",
    ) -> Dict[str, Any]:
        """
        Compute isolated Cardiac features (`peaks`, `HRV`, and `APG morphology`).
        `modality` can be `'PPG'` (computes APG ratios b/a, d/a, e/a and pulse slope)
        or `'ECG'` (computes R-peak amplitude and SD, zeroing out APG ratios).
        """
        cardiac_payload = {"signal": signal}
        c_signal = json.dumps(cardiac_payload).encode("utf-8")
        c_modality = modality.encode("utf-8")
        return self._call_and_parse(
            "ccs_compute_cardiac_isolated_json",
            c_signal,
            ctypes.c_double(srate),
            c_modality,
        )

    def compute_sqi(
        self,
        signal: List[float],
        srate: float = 100.0,
        modality: str = "EEG",
    ) -> float:
        """
        Compute Signal Quality Index (`SQI`) score (0.0 to 1.0).
        `modality` can be `'EEG'`, `'PPG'`, or `'ECG'`.
        """
        payload = {"signal": signal}
        c_payload = json.dumps(payload).encode("utf-8")
        c_modality = modality.encode("utf-8")
        res = self._call_and_parse(
            "ccs_compute_sqi_json",
            c_payload,
            ctypes.c_double(srate),
            c_modality,
        )
        return float(res.get("sqi", 0.0))
