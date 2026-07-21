import sys
import os
import time
import csv
import json
import collections
import threading
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2
import numpy as np
from scipy.optimize import curve_fit
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, QSlider,
                             QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea,
                             QCheckBox, QMessageBox, QInputDialog, QMdiSubWindow,
                             QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
                             QGroupBox, QFormLayout, QAbstractItemView)
from PyQt6.QtCore import QTimer, Qt, QSettings, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon
from PyQt6 import uic

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    elif getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

if sys.stdout is None or sys.stderr is None:
    _log_path = resource_path("beam_profiler_run.log")
    _log_file = open(_log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _log_file
    sys.stderr = _log_file

try:
    from pylablib.devices import Thorlabs
    _HAS_THORLABS = True
except Exception:
    _HAS_THORLABS = False

try:
    from pylablib.devices import uc480
    _HAS_UC480 = True
except Exception:
    _HAS_UC480 = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False

APP_NAME = "VOILALab Beam"
APP_VERSION = "v4.6"
ORG_NAME = "VOILA Lab"

SENSOR_ASSUMED_MAX_RAW = 1023
SENSOR_DISPLAY_DIVISOR = (SENSOR_ASSUMED_MAX_RAW + 1) / 256.0

GAUSSIAN_FIT_INTERVAL = 3
BACKGROUND_PERCENTILE = 5.0
PROFILE_BAND_HALFWIDTH = 5
GAUSSIAN_R2_MIN = 0.995
LOG_EVERY_N_FRAMES = 10
MOMENT_TRUNCATION_LEVEL = 0.135
ADVANCED_METRICS_INTERVAL_S = 0.20
DEFAULT_CLIP_LEVEL_PERCENT = 20
DEFAULT_RAW_DATA_OPACITY_PERCENT = 15
DEFAULT_MAX_DISPLAY_FPS = 30

DRAW_LIVE_OVERLAY = True
OVERLAY_CURVE_HEIGHT = 100
OVERLAY_ALPHA = 0.85

OVERLAY_GRID_LINE_BASE_PX = 5
OVERLAY_TICK_LEN_BASE_PX = 12
OVERLAY_FONT_SCALE_BASE = 1.0
OVERLAY_FONT_THICK_MULTIPLIER = 1.8
OVERLAY_CURVE_LINE_BASE_PX = 3
OVERLAY_CROSSHAIR_BIG_BASE_PX = 34
OVERLAY_CROSSHAIR_SMALL_BASE_PX = 16
OVERLAY_CROSSHAIR_CENTROID_THICK_BASE_PX = 3
OVERLAY_CROSSHAIR_THICK_BASE_PX = 1

STALE_FRAME_LIMIT = 15

DARK_QSS = """
QMainWindow, QWidget { background-color: #2b2b2b; color: #e0e0e0; }
QMdiArea { background-color: #1e1e1e; }
QLabel { color: #e0e0e0; }
QPushButton {
    background-color: #3c3f41; color: #e0e0e0; border: 1px solid #555555;
    border-radius: 4px; padding: 4px 8px;
}
QPushButton:hover { background-color: #4b4f52; }
QPushButton:checked { background-color: #3a7bd5; border: 1px solid #5a9bf5; }
QSlider::groove:horizontal { background: #555555; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #3a7bd5; width: 12px; margin: -6px 0; border-radius: 6px; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #3c3f41; color: #e0e0e0; border: 1px solid #555555; border-radius: 3px; padding: 2px;
}
QCheckBox { color: #e0e0e0; }
QMenuBar { background-color: #2b2b2b; color: #e0e0e0; }
QMenuBar::item:selected { background-color: #3a7bd5; }
QMenu { background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #555555; }
QMenu::item:selected { background-color: #3a7bd5; }
QToolBar { background-color: #2b2b2b; border: none; spacing: 2px; }
QToolButton {
    background-color: #3c3f41; color: #e0e0e0; border: 1px solid #555555;
    border-radius: 4px; padding: 4px 6px; margin: 2px;
}
QToolButton:hover { background-color: #4b4f52; }
QToolButton:checked {
    background-color: #3a7bd5; border: 1px solid #7fb0ff; color: #ffffff;
}
QMdiSubWindow { background-color: #333333; color: #e0e0e0; }
QScrollArea { background-color: #1e1e1e; }
"""

LIGHT_QSS = """
QToolBar { border: none; spacing: 2px; }
QToolButton {
    background-color: #f2f2f2; color: #202020; border: 1px solid #b5b5b5;
    border-radius: 4px; padding: 4px 6px; margin: 2px;
}
QToolButton:hover { background-color: #e2e2e2; }
QToolButton:checked {
    background-color: #3a7bd5; border: 1px solid #7fb0ff; color: #ffffff;
}
"""


@dataclass(frozen=True)
class CameraMetadata:
    camera_backend: str | None
    camera_model: str
    resolution: tuple[int, int]
    dtype: str
    sensor_bit_depth: int
    sensor_max_raw: int
    exposure_ms: float
    gain: object = None
    pixel_format: object = None


@dataclass(frozen=True)
class RawFrame:
    data: np.ndarray
    camera: CameraMetadata
    timestamp: float
    frame_index: int


@dataclass(frozen=True)
class ProcessedFrame:
    raw: RawFrame
    corrected: np.ndarray
    analysis: np.ndarray
    display_u8: np.ndarray
    roi: np.ndarray
    roi_bounds: tuple[int, int, int, int] | None
    offset_x: int
    offset_y: int
    dark_applied: bool
    dark_valid: bool
    dark_mismatch_reason: str = ""


@dataclass(frozen=True)
class MeasurementWarnings:
    low_snr: bool = False
    central_saturation: bool = False
    isolated_hot_pixel_saturation: bool = False
    sensor_edge_truncated: bool = False
    roi_edge_truncated: bool = False

    @property
    def invalidates_widths(self):
        return self.low_snr or self.central_saturation or self.sensor_edge_truncated or self.roi_edge_truncated


@dataclass(frozen=True)
class GaussianFitResult:
    parameters: tuple[float, float, float, float] | None
    r_squared: float | None
    diameter_px: float | None = None
    diameter_um: float | None = None
    valid: bool = False


@dataclass(frozen=True)
class BeamMeasurement:
    peak_x: int
    peak_y: int
    centroid_x: float
    centroid_y: float
    x_values: np.ndarray
    y_values: np.ndarray
    numbers_x: np.ndarray
    numbers_y: np.ndarray
    x_profile: dict | None
    y_profile: dict | None
    analysis_roi: np.ndarray
    second_moment: dict | None
    quality: "FrameQuality"
    warnings: MeasurementWarnings
    raw_saturation_percent: float
    processed_peak_percent: float
    background_subtracted_counts: float


@dataclass(frozen=True)
class CaptureMetadata:
    timestamp: str
    camera_backend: str | None
    capture_source: str
    exposure_ms: float
    pixel_pitch_um: float
    sensor_bit_depth: int
    sensor_max_raw: int
    centroid_px: dict
    dark_correction: dict
    signal_quality: dict | None
    x_profile: dict | None
    y_profile: dict | None
    second_moment: dict | None


@dataclass
class FrameQuality:
    background_level: float
    background_noise_mad: float
    snr_peak: float
    raw_peak: float
    corrected_peak: float
    averaged_peak: float
    displayed_peak: float
    saturated_pixel_count: int
    saturated_pixel_fraction: float
    hot_pixel_count: int
    sensor_edge_truncated: bool
    roi_edge_truncated: bool
    low_snr: bool
    central_saturation: bool
    isolated_hot_pixel_saturation: bool
    dark_applied: bool
    dark_valid: bool
    dark_mismatch_reason: str = ""


def robust_background_stats(image, border_fraction=0.10):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0, 0.0
    h, w = arr.shape
    by = max(1, int(round(h * border_fraction)))
    bx = max(1, int(round(w * border_fraction)))
    if 2 * by >= h or 2 * bx >= w:
        vals = arr.ravel()
    else:
        top = arr[:by, :]; bottom = arr[-by:, :]
        mid_left = arr[by:h - by, :bx]; mid_right = arr[by:h - by, -bx:]
        vals = np.concatenate([top.ravel(), bottom.ravel(), mid_left.ravel(), mid_right.ravel()])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, 1.4826 * mad


def dark_metadata_match(saved, current, exposure_tol_ms=1e-6):
    required = ("camera_backend", "camera_model", "resolution", "dtype",
                "sensor_bit_depth", "gain", "pixel_format")
    for key in required:
        if saved.get(key) != current.get(key):
            return False, f"{key} mismatch: dark={saved.get(key)!r}, current={current.get(key)!r}"
    if abs(float(saved.get("exposure_ms", -1)) - float(current.get("exposure_ms", -2))) > exposure_tol_ms:
        return False, "exposure_ms mismatch"
    return True, ""


def subtract_dark_frame(raw, dark):
    raw_arr = np.asarray(raw)
    dark_arr = np.asarray(dark)
    if raw_arr.shape != dark_arr.shape:
        raise ValueError("dark frame shape does not match raw frame")
    corrected = raw_arr.astype(np.float32) - dark_arr.astype(np.float32)
    np.maximum(corrected, 0.0, out=corrected)
    return corrected


def detect_hot_pixels(image, background_level, background_noise, sigma_threshold=8.0):
    arr = np.asarray(image, dtype=np.float32)
    threshold = float(background_level) + sigma_threshold * max(float(background_noise), 1.0)
    local = cv2.medianBlur(arr, 3)
    return (arr - local) > max(sigma_threshold * max(float(background_noise), 1.0), threshold - float(background_level))


def edge_truncation_flags(corrected, roi_bounds=None, level_fraction=MOMENT_TRUNCATION_LEVEL):
    arr = np.asarray(corrected, dtype=np.float32)
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 0:
        return False, False
    threshold = peak * float(level_fraction)
    sensor_edge = bool(np.any(arr[0, :] >= threshold) or np.any(arr[-1, :] >= threshold) or
                       np.any(arr[:, 0] >= threshold) or np.any(arr[:, -1] >= threshold))
    roi_edge = False
    if roi_bounds is not None:
        x0, y0, x1, y1 = roi_bounds
        roi = arr[y0:y1, x0:x1]
        if roi.size:
            roi_edge = bool(np.any(roi[0, :] >= threshold) or np.any(roi[-1, :] >= threshold) or
                            np.any(roi[:, 0] >= threshold) or np.any(roi[:, -1] >= threshold))
    return sensor_edge, roi_edge


@dataclass
class PropagationSample:
    timestamp: str
    z_mm: float | None = None
    d4sigma_x_um: float | None = None
    d4sigma_y_um: float | None = None
    d4sigma_major_um: float | None = None
    d4sigma_minor_um: float | None = None
    orientation_deg: float | None = None
    ellipticity: float | None = None
    d4sigma_truncated: bool = False
    gaussian_x_um: float | None = None
    gaussian_y_um: float | None = None
    r2_x: float | None = None
    r2_y: float | None = None
    saturation_fraction: float = 0.0
    clipped: bool = False
    valid: bool = False
    exposure_ms: float = 0.0
    pixel_pitch_um: float = 0.0
    wavelength_nm: float = 0.0
    source_frame_count: int = 1
    width_cv_percent: float | None = None


def second_moment_2d(image, background=None, clip_fraction=0.0):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if background is None:
        background = float(np.percentile(arr, BACKGROUND_PERCENTILE))
    weights = np.clip(arr - np.float32(background), 0.0, None)
    peak = float(weights.max()) if weights.size else 0.0
    if peak <= 0:
        return None
    if clip_fraction > 0:
        weights = np.where(weights >= clip_fraction * peak, weights, np.float32(0.0))

    m = cv2.moments(weights, binaryImage=False)
    total = float(m["m00"])
    if total <= 0:
        return None
    cx = m["m10"] / total
    cy = m["m01"] / total
    var_x = max(m["m20"] / total - cx * cx, 0.0)
    var_y = max(m["m02"] / total - cy * cy, 0.0)
    cov_xy = m["m11"] / total - cx * cy
    cx, cy = float(cx), float(cy)
    var_x, var_y, cov_xy = float(var_x), float(var_y), float(cov_xy)
    cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues < -1e-9):
        return None
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    order = np.argsort(eigenvalues)[::-1]
    major_var, minor_var = eigenvalues[order]
    major_vec = eigenvectors[:, order[0]]
    orientation_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    while orientation_deg >= 90.0:
        orientation_deg -= 180.0
    while orientation_deg < -90.0:
        orientation_deg += 180.0
    d4sigma_major = 4.0 * np.sqrt(major_var)
    d4sigma_minor = 4.0 * np.sqrt(minor_var)
    ellipticity = float(d4sigma_minor / d4sigma_major) if d4sigma_major > 0 else None

    border = np.concatenate([weights[0, :], weights[-1, :], weights[:, 0], weights[:, -1]])
    border_peak = float(border.max()) if border.size else 0.0
    truncated = bool(border_peak >= MOMENT_TRUNCATION_LEVEL * peak)
    valid = not truncated

    return {
        "centroid_x_px": cx, "centroid_y_px": cy,
        "var_x_px2": max(var_x, 0.0), "var_y_px2": max(var_y, 0.0),
        "cov_xy_px2": cov_xy,
        "major_var_px2": float(major_var), "minor_var_px2": float(minor_var),
        "d4sigma_x_px": 4.0 * np.sqrt(max(var_x, 0.0)),
        "d4sigma_y_px": 4.0 * np.sqrt(max(var_y, 0.0)),
        "d4sigma_major_px": d4sigma_major,
        "d4sigma_minor_px": d4sigma_minor,
        "orientation_deg": orientation_deg,
        "ellipticity": ellipticity,
        "truncated": truncated,
        "valid": valid,
        "background": float(background),
        "total_corrected_counts": total,
    }


def propagation_radius_model(z_mm, w0_um, z0_mm, m2, wavelength_nm):
    z_um = (np.asarray(z_mm, dtype=np.float64) - z0_mm) * 1000.0
    term = (m2 * wavelength_nm * 1e-3 * z_um) / (np.pi * w0_um * w0_um)
    return w0_um * np.sqrt(1.0 + term * term)


def fit_propagation_axis(z_mm, diameter_um, wavelength_nm):
    z = np.asarray(z_mm, dtype=np.float64)
    d = np.asarray(diameter_um, dtype=np.float64)
    mask = np.isfinite(z) & np.isfinite(d) & (d > 0)
    z, d = z[mask], d[mask]
    if z.size < 4 or np.unique(z).size < 4:
        return None
    w = d / 2.0
    i0 = int(np.argmin(w))
    w0_guess = max(float(w[i0]), 1e-6)
    z0_guess = float(z[i0])
    span = max(float(np.ptp(z)), 1.0)
    lower = [max(0.05 * w0_guess, 1e-6), float(z.min() - 2 * span), 0.2]
    upper = [max(float(w.max()) * 2.0, w0_guess * 1.1), float(z.max() + 2 * span), 100.0]
    def model(zvals, w0, z0, m2):
        return propagation_radius_model(zvals, w0, z0, m2, wavelength_nm)
    try:
        popt, pcov = curve_fit(model, z, w, p0=[w0_guess, z0_guess, 1.2],
                               bounds=(lower, upper), maxfev=30000)
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    predicted = model(z, *popt)
    residuals = w - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    nrmse = rmse / max(float(np.ptp(w)), float(np.mean(w)), 1e-12)
    errors = np.sqrt(np.diag(pcov)) if pcov.shape == (3, 3) and np.all(np.isfinite(pcov)) else np.full(3, np.nan)
    w0, z0, m2 = map(float, popt)
    wavelength_um = wavelength_nm * 1e-3
    z_r_um = np.pi * w0 * w0 / (m2 * wavelength_um)
    z_r_mm = z_r_um / 1000.0
    theta_half_rad = m2 * wavelength_um / (np.pi * w0)
    near_count = int(np.count_nonzero(np.abs(z - z0) <= z_r_mm)) if z_r_mm > 0 else 0
    far_count = int(np.count_nonzero(np.abs(z - z0) >= 2.0 * z_r_mm)) if z_r_mm > 0 else 0
    left = bool(np.any(z < z0))
    right = bool(np.any(z > z0))
    physical = bool(m2 >= 1.0 and w0 > 0 and z_r_mm > 0 and np.all(np.isfinite(popt)))
    return {
        "w0_um": w0, "z0_mm": z0, "m2": m2,
        "w0_error_um": float(errors[0]), "z0_error_mm": float(errors[1]),
        "m2_error": float(errors[2]), "z_r_mm": z_r_mm,
        "theta_half_mrad": theta_half_rad * 1000.0,
        "theta_full_mrad": theta_half_rad * 2000.0,
        "bpp_mm_mrad": w0 / 1000.0 * theta_half_rad * 1000.0,
        "rmse_um_radius": rmse, "nrmse": nrmse,
        "near_count": near_count, "far_count": far_count,
        "left_sampled": left, "right_sampled": right,
        "physical": physical, "covariance_ok": bool(np.all(np.isfinite(pcov))),
        "z": z, "observed_radius_um": w, "predicted_radius_um": predicted,
    }


def gaussian_model(x, a, b, c, d):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d


def r_squared(observed, predicted):
    residual_ss = np.sum((observed - predicted) ** 2)
    total_ss = np.sum((observed - np.mean(observed)) ** 2)
    if total_ss <= 0:
        return 0.0
    return 1.0 - residual_ss / total_ss


def gaussian_1e2_diameter(popt):
    return 4.0 * abs(popt[2])


def divergence_half_angle_mrad(w0_um, wavelength_nm):
    if w0_um is None or w0_um <= 0 or wavelength_nm is None or wavelength_nm <= 0:
        return None
    return wavelength_nm / (np.pi * w0_um)


def correct_for_instrument_psf(measured_um, psf_um):
    if psf_um is None or psf_um <= 0:
        return measured_um
    diff = measured_um ** 2 - psf_um ** 2
    return float(np.sqrt(diff)) if diff > 0 else 0.0


def power_to_dbm(power_mw):
    if power_mw is None or power_mw <= 0:
        return None
    return 10.0 * np.log10(power_mw)


def intensity_weighted_centroid(image, background, clip_level):
    weights = np.clip(image.astype(np.float32) - np.float32(background), 0, None)
    peak_w = weights.max()
    if peak_w <= 0:
        return None
    mask_level = clip_level * peak_w
    weights = np.where(weights >= mask_level, weights, np.float32(0.0))
    m = cv2.moments(weights, binaryImage=False)
    total = m["m00"]
    if total <= 0:
        return None
    cx = m["m10"] / total
    cy = m["m01"] / total
    return cx, cy


def _interpolated_crossing(values, inside_idx, outside_idx, level):
    y_in = values[inside_idx]
    y_out = values[outside_idx]
    if y_in == y_out:
        return float(inside_idx)
    frac = (level - y_out) / (y_in - y_out)
    return outside_idx + frac * (inside_idx - outside_idx)


def analyze_slice(values, min_signal, background_percentile=BACKGROUND_PERCENTILE):
    background = float(np.percentile(values, background_percentile))
    peak = float(np.max(values))
    signal = peak - background
    if signal <= min_signal:
        return None

    info = {"peak": peak, "background": background, "signal": signal}
    n = len(values)

    half_max_level = background + 0.50 * signal
    idx = np.where(values > half_max_level)[0]
    if len(idx) > 0:
        left_i, right_i = int(idx[0]), int(idx[-1])
        touches_edge = (left_i == 0) or (right_i == n - 1)
        left_sub = (_interpolated_crossing(values, left_i, left_i - 1, half_max_level)
                    if left_i > 0 else float(left_i))
        right_sub = (_interpolated_crossing(values, right_i, right_i + 1, half_max_level)
                     if right_i < n - 1 else float(right_i))
        info["fwhm_left"], info["fwhm_right"] = left_sub, right_sub
        info["fwhm_width"] = right_sub - left_sub
        info["fwhm_touches_edge"] = touches_edge
        info["half_max_level"] = half_max_level

    e2_level = background + 0.135 * signal
    idx2 = np.where(values > e2_level)[0]
    if len(idx2) > 0:
        left_i2, right_i2 = int(idx2[0]), int(idx2[-1])
        touches_edge2 = (left_i2 == 0) or (right_i2 == n - 1)
        left_sub2 = (_interpolated_crossing(values, left_i2, left_i2 - 1, e2_level)
                     if left_i2 > 0 else float(left_i2))
        right_sub2 = (_interpolated_crossing(values, right_i2, right_i2 + 1, e2_level)
                      if right_i2 < n - 1 else float(right_i2))
        info["e2_left"], info["e2_right"] = left_sub2, right_sub2
        info["e2_width"] = info["e2_right"] - info["e2_left"]
        info["e2_touches_edge"] = touches_edge2
        info["e2_level"] = e2_level

    return info


class CameraBackend:
    name = "Base Camera"

    def open(self):
        raise NotImplementedError

    def close(self):
        pass

    def get_frame(self):
        raise NotImplementedError

    def set_exposure_ms(self, exposure_ms):
        pass

    def get_bit_depth(self):
        return 8

    def get_sensor_max_raw(self):
        return (2 ** self.get_bit_depth()) - 1

    def get_pixel_pitch_um(self):
        return None

    def get_detector_size(self):
        return None

    def get_info_text(self):
        return f"Backend: {self.name}\nBit depth: {self.get_bit_depth()}-bit (assumed)"


class OpenCVCameraBackend(CameraBackend):

    name = "Generic Webcam / OpenCV"

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera index {self.camera_index}.")

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def set_exposure_ms(self, exposure_ms):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_ms)

    def get_bit_depth(self):
        return 8

    def get_pixel_pitch_um(self):
        return None

    def get_detector_size(self):
        if self.cap is None:
            return None
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w > 0 and h > 0 else None

    def get_info_text(self):
        return (f"Backend: {self.name}\n"
                f"Camera index: {self.camera_index}\n"
                f"Bit depth: 8-bit (standard UVC output; true sensor ADC unknown)")

    @staticmethod
    def list_available_devices(max_probe=5):
        devices = []
        for i in range(max_probe):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append((f"Webcam index {i}", i))
            cap.release()
        return devices


class ThorlabsScientificCameraBackend(CameraBackend):

    name = "Thorlabs Scientific / Zelux"

    def __init__(self, serial=None):
        self.cap = None
        self.bit_depth = 10
        self.sensor_type = None
        self.serial = serial

    def open(self):
        if not _HAS_THORLABS:
            raise RuntimeError("Thorlabs Scientific Camera SDK not available "
                               "(pylablib.devices.Thorlabs failed to import).")
        self.cap = Thorlabs.ThorlabsTLCamera(serial=self.serial)
        self.cap.setup_acquisition()
        self.cap.start_acquisition()
        self.set_exposure_ms(5.0)
        try:
            sensor_type, bit_depth = self.cap.get_sensor_info()
            self.sensor_type = sensor_type
            self.bit_depth = bit_depth if bit_depth else 10
        except Exception:
            self.sensor_type = None
            self.bit_depth = 10

    def close(self):
        if self.cap is not None:
            try:
                self.cap.stop_acquisition()
            except Exception:
                pass
            try:
                self.cap.close()
            except Exception:
                pass
            self.cap = None

    def get_frame(self):
        if self.cap is None:
            return None
        frame = self.cap.read_newest_image()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def set_exposure_ms(self, exposure_ms):
        if self.cap is not None:
            self.cap.set_exposure(exposure_ms / 1000.0)

    def get_bit_depth(self):
        return self.bit_depth

    def get_pixel_pitch_um(self):
        return 3.45

    def get_detector_size(self):
        if self.cap is None:
            return None
        try:
            return self.cap.get_detector_size()
        except Exception:
            return None

    def get_info_text(self):
        try:
            model, name, serial, firmware = self.cap.get_device_info()
        except Exception:
            model = name = serial = firmware = "unavailable"
        return (f"Backend: {self.name}\n"
                f"Camera model: {model}\n"
                f"Camera name: {name}\n"
                f"Serial number: {serial}\n"
                f"Firmware: {firmware}\n"
                f"Sensor type: {self.sensor_type or 'unknown'}\n"
                f"Sensor bit depth: {self.bit_depth}-bit (full scale = {self.get_sensor_max_raw()} counts)")

    @staticmethod
    def list_available_devices():
        if not _HAS_THORLABS:
            return []
        try:
            serials = Thorlabs.list_cameras_tlcam()
        except Exception:
            return []
        return [(f"Zelux/TLCamera S/N {s}", s) for s in serials]


class UC480CameraBackend(CameraBackend):

    name = "IDS uEye / UC480"

    def __init__(self, cam_id=0):
        self.cap = None
        self.bit_depth = 8
        self.cam_id = cam_id

    def open(self):
        if not _HAS_UC480:
            raise RuntimeError("UC480/uEye support not available "
                               "(pylablib.devices.uc480 failed to import).")
        self.cap = uc480.UC480Camera(cam_id=self.cam_id)
        self.cap.setup_acquisition()
        self.cap.start_acquisition()
        self.set_exposure_ms(5.0)

    def close(self):
        if self.cap is not None:
            try:
                self.cap.stop_acquisition()
            except Exception:
                pass
            try:
                self.cap.close()
            except Exception:
                pass
            self.cap = None

    def get_frame(self):
        if self.cap is None:
            return None
        frame = self.cap.read_newest_image()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def set_exposure_ms(self, exposure_ms):
        if self.cap is not None:
            self.cap.set_exposure(exposure_ms / 1000.0)

    def get_bit_depth(self):
        return self.bit_depth

    def get_pixel_pitch_um(self):
        return 5.30

    def get_detector_size(self):
        if self.cap is None:
            return None
        try:
            return self.cap.get_detector_size()
        except Exception:
            return None

    def get_info_text(self):
        try:
            info = self.cap.get_device_info()
            model, manufacturer, serial_number = info[1], info[2], info[3]
        except Exception:
            model = manufacturer = serial_number = "unavailable"
        return (f"Backend: {self.name}\n"
                f"Camera model: {model}\n"
                f"Manufacturer: {manufacturer}\n"
                f"Serial number: {serial_number}\n"
                f"Bit depth: {self.bit_depth}-bit (fixed; not queryable via this SDK)")

    @staticmethod
    def list_available_devices():
        if not _HAS_UC480:
            return []
        try:
            infos = uc480.list_cameras(backend="uc480")
        except Exception:
            return []
        return [(f"{info.model} S/N {info.serial_number} (id {info.cam_id})", info.cam_id) for info in infos]


CAMERA_BACKEND_NAMES = [
    "Generic Webcam / OpenCV",
    "Thorlabs Scientific / Zelux",
    "IDS uEye / UC480",
]


@dataclass(frozen=True)
class AcquiredFramePacket:
    data: np.ndarray
    camera: CameraMetadata
    acquired_monotonic: float
    frame_index: int


class LatestFrameMailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._published = 0
        self._consumed = 0
        self._dropped = 0
        self._errors = 0
        self._acquisition_fps = 0.0
        self._last_publish = None

    def publish(self, packet):
        with self._lock:
            if self._latest is not None:
                self._dropped += 1
            now = packet.acquired_monotonic
            if self._last_publish is not None and now > self._last_publish:
                instant = 1.0 / (now - self._last_publish)
                self._acquisition_fps = instant if self._acquisition_fps <= 0 else 0.9 * self._acquisition_fps + 0.1 * instant
            self._last_publish = now
            self._latest = packet
            self._published += 1

    def take_latest(self):
        with self._lock:
            packet = self._latest
            self._latest = None
            if packet is not None:
                self._consumed += 1
            return packet

    def clear(self):
        with self._lock:
            self._latest = None

    def note_error(self):
        with self._lock:
            self._errors += 1

    def snapshot(self):
        with self._lock:
            age_ms = None if self._last_publish is None else max(0.0, (time.perf_counter() - self._last_publish) * 1000.0)
            return {
                "published": self._published, "consumed": self._consumed,
                "dropped": self._dropped, "acquisition_errors": self._errors,
                "acquisition_fps": self._acquisition_fps, "latest_frame_age_ms": age_ms,
            }


class CameraWorker(QObject):
    connected = pyqtSignal(object)
    disconnected = pyqtSignal()
    error = pyqtSignal(str, str)
    dark_frame_ready = pyqtSignal(object, object)
    dark_frame_failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, mailbox):
        super().__init__()
        self._mailbox = mailbox
        self._camera = None
        self._backend_name = None
        self._device_text = ""
        self._exposure_ms = 5.0
        self._running = False
        self._stopping = False
        self._frame_index = 0
        self._timer = None
        self._dark_target = 0
        self._dark_frames = []

    @pyqtSlot()
    def initialize(self):
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._acquire_once)

    def _schedule_next(self):
        if self._timer is not None and self._running and not self._stopping:
            self._timer.start(1)

    def _make_backend(self, spec):
        backend_name = spec["backend_name"]
        device_id = spec.get("device_id")
        if backend_name == "Generic Webcam / OpenCV":
            return OpenCVCameraBackend(camera_index=device_id if device_id is not None else 0)
        if backend_name == "Thorlabs Scientific / Zelux":
            return ThorlabsScientificCameraBackend(serial=device_id)
        if backend_name == "IDS uEye / UC480":
            return UC480CameraBackend(cam_id=device_id if device_id is not None else 0)
        raise RuntimeError(f"Unknown camera backend: {backend_name}")

    def _safe_close(self):
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                self._mailbox.note_error()
                self.error.emit(f"Camera close failed: {exc}", traceback.format_exc())

    @pyqtSlot(object)
    def connect_camera(self, spec):
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        self._safe_close()
        self._mailbox.clear()
        self._dark_frames.clear(); self._dark_target = 0
        try:
            camera = self._make_backend(spec)
            camera.open()
            self._camera = camera
            self._backend_name = spec["backend_name"]
            self._device_text = spec.get("device_text", "")
            self._exposure_ms = float(spec.get("exposure_ms", 5.0))
            camera.set_exposure_ms(self._exposure_ms)
            self._frame_index = 0
            info = {
                "backend_name": self._backend_name,
                "device_text": self._device_text,
                "sensor_type": getattr(camera, "sensor_type", None),
                "sensor_bit_depth": int(camera.get_bit_depth()),
                "sensor_max_raw": int(camera.get_sensor_max_raw()),
                "pixel_pitch_um": camera.get_pixel_pitch_um(),
                "detector_size": camera.get_detector_size(),
                "info_text": camera.get_info_text(),
            }
            self._running = True
            self.connected.emit(info)
            self._schedule_next()
        except Exception as exc:
            self._safe_close()
            self._mailbox.note_error()
            self.error.emit(str(exc), traceback.format_exc())

    @pyqtSlot()
    def disconnect_camera(self):
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        self._safe_close()
        self._mailbox.clear()
        self.disconnected.emit()

    @pyqtSlot(float)
    def set_exposure_ms(self, exposure_ms):
        self._exposure_ms = max(0.01, float(exposure_ms))
        try:
            if self._camera is not None:
                self._camera.set_exposure_ms(self._exposure_ms)
        except Exception as exc:
            self._mailbox.note_error()
            self.error.emit(f"Exposure update failed: {exc}", traceback.format_exc())

    @pyqtSlot(int)
    def request_dark_frame(self, count):
        if self._camera is None:
            self.dark_frame_failed.emit("Connect a camera first.")
            return
        self._dark_target = max(2, int(count))
        self._dark_frames = []

    @pyqtSlot()
    def stop(self):
        self._stopping = True
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        self._safe_close()
        self._mailbox.clear()
        self.stopped.emit()
        QThread.currentThread().quit()

    @pyqtSlot()
    def _acquire_once(self):
        if not self._running or self._stopping or self._camera is None:
            return
        try:
            frame = self._camera.get_frame()
            if frame is not None:
                arr = np.asarray(frame)
                if arr.ndim != 2 or arr.size == 0 or arr.dtype.kind not in "ui":
                    raise ValueError(f"Invalid frame: shape={arr.shape}, dtype={arr.dtype}")
                owned = np.ascontiguousarray(arr).copy()
                self._frame_index += 1
                meta = CameraMetadata(
                    camera_backend=self._backend_name,
                    camera_model=str(getattr(self._camera, "model", None) or getattr(self._camera, "name", None) or self._device_text),
                    resolution=(owned.shape[1], owned.shape[0]), dtype=str(owned.dtype),
                    sensor_bit_depth=int(self._camera.get_bit_depth()),
                    sensor_max_raw=int(self._camera.get_sensor_max_raw()),
                    exposure_ms=float(self._exposure_ms),
                    gain=getattr(self._camera, "gain", None),
                    pixel_format=getattr(self._camera, "pixel_format", None) or str(owned.dtype),
                )
                packet = AcquiredFramePacket(owned, meta, time.perf_counter(), self._frame_index)
                self._mailbox.publish(packet)
                if self._dark_target:
                    self._dark_frames.append(owned.copy())
                    if len(self._dark_frames) >= self._dark_target:
                        first = self._dark_frames[0]
                        if any(f.shape != first.shape or f.dtype != first.dtype for f in self._dark_frames):
                            self.dark_frame_failed.emit("Dark-frame sequence changed shape or dtype.")
                        else:
                            averaged = np.rint(np.mean(np.stack(self._dark_frames).astype(np.float64), axis=0))
                            info = np.iinfo(first.dtype)
                            averaged = np.clip(averaged, info.min, info.max).astype(first.dtype)
                            self.dark_frame_ready.emit(averaged, {
                                "camera_backend": meta.camera_backend,
                                "camera_model": meta.camera_model,
                                "resolution": tuple(first.shape),
                                "dtype": str(first.dtype),
                                "sensor_bit_depth": meta.sensor_bit_depth,
                                "exposure_ms": meta.exposure_ms,
                                "gain": meta.gain,
                                "pixel_format": meta.pixel_format,
                            })
                        self._dark_frames = []; self._dark_target = 0
        except Exception as exc:
            self._mailbox.note_error()
            self.error.emit(f"Frame acquisition failed: {exc}", traceback.format_exc())
        finally:
            self._schedule_next()

class PersistentSubWindow(QMdiSubWindow):

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.toggle_action = None

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        if self.toggle_action is not None:
            self.toggle_action.setChecked(False)


class CameraPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        self.snapButton = QPushButton("Snap to Beam")
        row1.addWidget(self.snapButton)
        row1.addStretch()
        row1.addWidget(QLabel("Zoom:"))
        self.zoomSpinBox = QSpinBox()
        self.zoomSpinBox.setRange(10, 500)
        self.zoomSpinBox.setValue(100)
        self.zoomSpinBox.setSuffix("%")
        row1.addWidget(self.zoomSpinBox)
        row1.addWidget(QLabel("Grid Label Size:"))
        self.gridLabelSizeSpinBox = QSpinBox()
        self.gridLabelSizeSpinBox.setRange(20, 150)
        self.gridLabelSizeSpinBox.setValue(50)
        self.gridLabelSizeSpinBox.setSuffix("%")
        row1.addWidget(self.gridLabelSizeSpinBox)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.centroidCrosshairCheckbox = QCheckBox("Centroid Crosshair")
        self.centroidCrosshairCheckbox.setChecked(True)
        row2.addWidget(self.centroidCrosshairCheckbox)
        self.peakCrosshairCheckbox = QCheckBox("Peak Crosshair")
        self.peakCrosshairCheckbox.setChecked(True)
        row2.addWidget(self.peakCrosshairCheckbox)
        row2.addStretch()
        self.fpsLabel = QLabel("Measured: -- FPS | -- ms/frame")
        row2.addWidget(self.fpsLabel)
        layout.addLayout(row2)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(False)

        self.videoLabel = QLabel("Loading Camera...")
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoLabel.setWordWrap(True)
        self.videoLabel.setMinimumSize(320, 240)

        self.scrollArea.setWidget(self.videoLabel)
        layout.addWidget(self.scrollArea)


class GraphPanel(QWidget):
    def __init__(self, y_max=SENSOR_ASSUMED_MAX_RAW * 1.05):
        super().__init__()
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(3.4, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_title("Live Beam Profile", fontsize=9)
        self.ax.set_xlabel("Pixel Position", fontsize=8)
        self.ax.set_ylim(0, y_max)
        self.ax.tick_params(labelsize=7)
        self.ax.grid(True)

        self.raw_data_line, = self.ax.plot([], [], 'k.', markersize=5,
                                            alpha=DEFAULT_RAW_DATA_OPACITY_PERCENT / 100.0,
                                            label="Raw Laser Data")
        self.fit_curve_line, = self.ax.plot([], [], 'r-', label="FWHM (50%)")
        self.e2_curve_line, = self.ax.plot([], [], 'b-', label="1/e² (13.5%)")
        self.gaussian_curve_line, = self.ax.plot([], [], 'g--', label="Ideal Gaussian")

        self.ax.legend(fontsize=7)
        layout.addWidget(self.canvas)

        self.percentage_mode = False
        self.raw_max = y_max / 1.05

        self.background = None
        self.canvas.mpl_connect('resize_event', self._on_resize)

    def _on_resize(self, _event):
        self.background = None

    def set_xlim_once(self, width):
        xlim = self.ax.get_xlim()
        if xlim != (0, width):
            self.ax.set_xlim(0, width)
            self.background = None

    def set_intensity_mode(self, percentage, raw_max):
        self.percentage_mode = percentage
        self.raw_max = raw_max
        if percentage:
            self.ax.set_ylim(0, 105)
        else:
            self.ax.set_ylim(0, raw_max * 1.05)
        self.background = None

    def scale_values(self, values):
        if self.percentage_mode and self.raw_max:
            return np.asarray(values, dtype=np.float64) / self.raw_max * 100.0
        return values

    def set_theme(self, dark):
        face = "#2b2b2b" if dark else "white"
        text_color = "#e0e0e0" if dark else "black"
        grid_color = "#555555" if dark else "#b0b0b0"
        self.fig.patch.set_facecolor(face)
        self.ax.set_facecolor(face)
        for spine in self.ax.spines.values():
            spine.set_color(text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.title.set_color(text_color)
        self.ax.tick_params(colors=text_color)
        self.ax.grid(True, color=grid_color)
        self.raw_data_line.set_color("white" if dark else "black")
        legend = self.ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(face)
            for text in legend.get_texts():
                text.set_color(text_color)
        self.background = None
        self.canvas.draw()

    def _set_lines_visible(self, visible):
        for line in (self.raw_data_line, self.fit_curve_line,
                     self.e2_curve_line, self.gaussian_curve_line):
            line.set_visible(visible)

    def _capture_clean_background(self):
        self._set_lines_visible(False)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self._set_lines_visible(True)

    def blit_draw(self):
        if self.background is None:
            self._capture_clean_background()
        self.canvas.restore_region(self.background)
        for line in (self.raw_data_line, self.fit_curve_line,
                     self.e2_curve_line, self.gaussian_curve_line):
            self.ax.draw_artist(line)
        self.canvas.blit(self.ax.bbox)


class StabilityPanel(QWidget):

    def __init__(self, max_points=300):
        super().__init__()
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Beam Pointing Stability")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Centroid Offset from Sensor Center (µm)")
        self.ax.grid(True)

        self.x_line, = self.ax.plot([], [], 'r-', label="ΔX")
        self.y_line, = self.ax.plot([], [], 'b-', label="ΔY")
        self.ax.legend()
        layout.addWidget(self.canvas)

        self.max_points = max_points
        self.t_data = collections.deque(maxlen=max_points)
        self.x_data = collections.deque(maxlen=max_points)
        self.y_data = collections.deque(maxlen=max_points)
        self.t0 = time.perf_counter()

    def append_sample(self, dx_um, dy_um):
        t = time.perf_counter() - self.t0
        self.t_data.append(t)
        self.x_data.append(dx_um)
        self.y_data.append(dy_um)

        self.x_line.set_data(self.t_data, self.x_data)
        self.y_line.set_data(self.t_data, self.y_data)

        if len(self.t_data) > 1:
            self.ax.set_xlim(self.t_data[0], self.t_data[-1])
        all_vals = list(self.x_data) + list(self.y_data)
        if all_vals:
            span = max(1.0, max(abs(v) for v in all_vals) * 1.2)
            self.ax.set_ylim(-span, span)

        self.canvas.draw_idle()

    def set_theme(self, dark):
        face = "#2b2b2b" if dark else "white"
        text_color = "#e0e0e0" if dark else "black"
        grid_color = "#555555" if dark else "#b0b0b0"
        self.fig.patch.set_facecolor(face)
        self.ax.set_facecolor(face)
        for spine in self.ax.spines.values():
            spine.set_color(text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.title.set_color(text_color)
        self.ax.tick_params(colors=text_color)
        self.ax.grid(True, color=grid_color)
        legend = self.ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(face)
            for text in legend.get_texts():
                text.set_color(text_color)
        self.canvas.draw()


class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        def section_label(text):
            lbl = QLabel(text)
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
            return lbl

        self.percentModeCheckbox = QCheckBox("Show Graphs as % of Full Scale (uses measured bit depth)")
        layout.addWidget(self.percentModeCheckbox)

        layout.addWidget(section_label("Camera Controls"))

        self.captureButton = QPushButton("Capture Image")
        layout.addWidget(self.captureButton)

        self.sliderLabel = QLabel("Camera Exposure: Medium")
        layout.addWidget(self.sliderLabel)
        exposure_row = QHBoxLayout()
        self.exposureSlider = QSlider(Qt.Orientation.Horizontal)
        self.exposureSlider.setRange(1, 100000)
        self.exposureSlider.setValue(500)
        exposure_row.addWidget(self.exposureSlider)
        self.exposureSpinBox = QDoubleSpinBox()
        self.exposureSpinBox.setDecimals(2)
        self.exposureSpinBox.setRange(0.01, 1000.0)
        self.exposureSpinBox.setSingleStep(0.01)
        self.exposureSpinBox.setSuffix(" ms")
        self.exposureSpinBox.setValue(5.0)
        exposure_row.addWidget(self.exposureSpinBox)
        layout.addLayout(exposure_row)
        self.exposureSlider.valueChanged.connect(self._sync_exposure_from_slider)
        self.exposureSpinBox.valueChanged.connect(self._sync_exposure_from_spinbox)

        self.cmapLabel = QLabel("Color Map:")
        layout.addWidget(self.cmapLabel)
        self.cmapCombo = QComboBox()
        self.cmapCombo.addItems(["Inferno", "Jet", "Hot", "Magma"])
        layout.addWidget(self.cmapCombo)

        self.zoomLabel = QLabel("Zoom Level:")
        layout.addWidget(self.zoomLabel)
        zoom_row = QHBoxLayout()
        self.zoomSlider = QSlider(Qt.Orientation.Horizontal)
        self.zoomSlider.setRange(10, 500)
        self.zoomSlider.setValue(100)
        zoom_row.addWidget(self.zoomSlider)
        self.zoomSpinBox = QSpinBox()
        self.zoomSpinBox.setRange(10, 500)
        self.zoomSpinBox.setValue(100)
        self.zoomSpinBox.setSuffix("%")
        zoom_row.addWidget(self.zoomSpinBox)
        layout.addLayout(zoom_row)
        self.zoomSlider.valueChanged.connect(self.zoomSpinBox.setValue)
        self.zoomSpinBox.valueChanged.connect(self.zoomSlider.setValue)

        self.overlayHeightLabel = QLabel(f"Overlay Curve Height: {OVERLAY_CURVE_HEIGHT}px")
        layout.addWidget(self.overlayHeightLabel)
        self.overlayHeightSlider = QSlider(Qt.Orientation.Horizontal)
        self.overlayHeightSlider.setRange(10, 400)
        self.overlayHeightSlider.setValue(OVERLAY_CURVE_HEIGHT)
        layout.addWidget(self.overlayHeightSlider)

        layout.addWidget(section_label("Graph"))

        self.opacityLabel = QLabel(f"Raw Data Opacity: {DEFAULT_RAW_DATA_OPACITY_PERCENT}%")
        layout.addWidget(self.opacityLabel)
        self.opacitySlider = QSlider(Qt.Orientation.Horizontal)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setValue(DEFAULT_RAW_DATA_OPACITY_PERCENT)
        layout.addWidget(self.opacitySlider)

        layout.addWidget(section_label("Camera Selection"))
        self.cameraBackendLabel = QLabel("Camera Backend:")
        layout.addWidget(self.cameraBackendLabel)
        self.cameraBackendCombo = QComboBox()
        self.cameraBackendCombo.addItems(CAMERA_BACKEND_NAMES)
        layout.addWidget(self.cameraBackendCombo)

        self.scanCamerasButton = QPushButton("Scan for Cameras")
        layout.addWidget(self.scanCamerasButton)
        self.deviceLabel = QLabel("Device:")
        layout.addWidget(self.deviceLabel)
        self.deviceCombo = QComboBox()
        layout.addWidget(self.deviceCombo)

        self.connectCameraButton = QPushButton("Connect Camera")
        layout.addWidget(self.connectCameraButton)

    def _sync_exposure_from_slider(self, value):
        self.exposureSpinBox.setValue(value / 100.0)

    def _sync_exposure_from_spinbox(self, ms):
        self.exposureSlider.setValue(int(round(ms * 100)))


class AnalyzePanel(QWidget):

    def __init__(self):
        super().__init__()
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(12)
        row = [0]

        def add_pair(left, right):
            grid.addWidget(left, row[0], 0)
            grid.addWidget(right, row[0], 1)
            row[0] += 1

        def add_full(widget):
            grid.addWidget(widget, row[0], 0, 1, 2)
            row[0] += 1

        pitch_layout = QHBoxLayout()
        pitch_label = QLabel("Pixel Pitch (µm):")
        self.pixel_pitch_input = QDoubleSpinBox()
        self.pixel_pitch_input.setDecimals(2)
        self.pixel_pitch_input.setRange(0.1, 100.0)
        self.pixel_pitch_input.setValue(3.45)
        self.pixel_pitch_input.setSingleStep(0.1)
        pitch_layout.addWidget(pitch_label)
        pitch_layout.addWidget(self.pixel_pitch_input)
        pitch_container = QWidget()
        pitch_container.setLayout(pitch_layout)
        add_full(pitch_container)


        self.saturation_label = QLabel("A/D Saturation: N/A")

        self.e2_x_label = QLabel("X 1/e² Threshold: N/A")
        self.e2_y_label = QLabel("Y 1/e² Threshold: N/A")

        self.gauss_diam_x_label = QLabel("X Gaussian Diameter: N/A")
        self.gauss_diam_y_label = QLabel("Y Gaussian Diameter: N/A")

        self.fit_quality_x_label = QLabel("X Fit Shape: N/A")
        self.fit_quality_y_label = QLabel("Y Fit Shape: N/A")

        self.centroid_position_label = QLabel("Centroid Position: N/A")
        self.peak_position_label = QLabel("Peak Position: N/A")

        self.fwhm_x_label = QLabel("X FWHM: N/A")
        self.fwhm_y_label = QLabel("Y FWHM: N/A")

        self.treatAsWaistCheckbox = QCheckBox("Treat current plane as beam waist (required for divergence estimate)")
        self.treatAsWaistCheckbox.setChecked(False)

        self.divergence_x_label = QLabel("X Divergence: enable checkbox")
        self.divergence_y_label = QLabel("Y Divergence: enable checkbox")
        for lbl in (self.divergence_x_label, self.divergence_y_label):
            lbl.setStyleSheet("background-color: rgba(58,123,213,40); border-radius: 4px; padding: 3px;")


        self.d4sigma_x_label = QLabel("X ISO D4σ: N/A")
        self.d4sigma_y_label = QLabel("Y ISO D4σ: N/A")
        self.d4sigma_principal_label = QLabel("Principal D4σ (major/minor, angle): N/A")
        self.d4sigma_ellipticity_label = QLabel("Ellipticity (minor/major D4σ): N/A")

        self.quality_label = QLabel("Signal Quality: N/A")
        self.background_label = QLabel("Background: N/A")
        self.power_label = QLabel("Total Power: Not Calibrated")

        for label in [self.saturation_label,
                      self.e2_x_label, self.e2_y_label,
                      self.gauss_diam_x_label, self.gauss_diam_y_label,
                      self.fit_quality_x_label, self.fit_quality_y_label,
                      self.centroid_position_label, self.peak_position_label,
                      self.fwhm_x_label, self.fwhm_y_label,
                      self.divergence_x_label, self.divergence_y_label,
                      self.d4sigma_x_label, self.d4sigma_y_label, self.d4sigma_principal_label,
                      self.d4sigma_ellipticity_label,
                      self.quality_label, self.background_label, self.power_label]:
            font = label.font()
            font.setBold(True)
            font.setPointSize(8)
            label.setFont(font)

        add_full(self.saturation_label)
        add_pair(self.e2_x_label, self.e2_y_label)
        add_pair(self.gauss_diam_x_label, self.gauss_diam_y_label)
        add_pair(self.fit_quality_x_label, self.fit_quality_y_label)
        add_pair(self.centroid_position_label, self.peak_position_label)
        add_pair(self.fwhm_x_label, self.fwhm_y_label)
        add_full(self.treatAsWaistCheckbox)
        add_pair(self.divergence_x_label, self.divergence_y_label)
        add_pair(self.d4sigma_x_label, self.d4sigma_y_label)
        add_full(self.d4sigma_principal_label)
        add_full(self.d4sigma_ellipticity_label)
        add_full(self.quality_label)
        add_full(self.background_label)
        add_full(self.power_label)

        self.logButton = QPushButton("Start Logging")
        self.logButton.setCheckable(True)
        add_full(self.logButton)
        self.logStatusLabel = QLabel("Logging: Off")
        add_full(self.logStatusLabel)


class AdvancedSettingsPanel(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.resetDefaultsButton = QPushButton("Reset to Defaults")
        layout.addWidget(self.resetDefaultsButton)

        layout.addWidget(QLabel("--- Background / Signal Quality ---"))
        dark_row = QHBoxLayout()
        self.captureDarkButton = QPushButton("Capture Dark (8 frames)")
        self.darkEnableCheckbox = QCheckBox("Apply Dark Subtraction")
        self.darkEnableCheckbox.setChecked(False)
        dark_row.addWidget(self.captureDarkButton)
        dark_row.addWidget(self.darkEnableCheckbox)
        layout.addLayout(dark_row)
        self.darkStatusLabel = QLabel("Dark frame: not captured")
        layout.addWidget(self.darkStatusLabel)
        snr_row = QHBoxLayout()
        snr_row.addWidget(QLabel("Minimum Peak SNR:"))
        self.minSnrSpinBox = QDoubleSpinBox()
        self.minSnrSpinBox.setRange(0.0, 10000.0)
        self.minSnrSpinBox.setDecimals(1)
        self.minSnrSpinBox.setValue(10.0)
        snr_row.addWidget(self.minSnrSpinBox)
        layout.addLayout(snr_row)
        self.badPixelMaskCheckbox = QCheckBox("Mask isolated hot pixels for analysis")
        self.badPixelMaskCheckbox.setChecked(False)
        layout.addWidget(self.badPixelMaskCheckbox)

        self.enableAdvancedMetricsCheckbox = QCheckBox("Enable D4σ / background / hot-pixel analysis")
        self.enableAdvancedMetricsCheckbox.setChecked(True)
        layout.addWidget(self.enableAdvancedMetricsCheckbox)

        r2_row = QHBoxLayout()
        r2_row.addWidget(QLabel("Gaussian Fit Threshold (R\u00B2):"))
        self.gaussianR2SpinBox = QDoubleSpinBox()
        self.gaussianR2SpinBox.setRange(0.5, 0.999)
        self.gaussianR2SpinBox.setDecimals(3)
        self.gaussianR2SpinBox.setSingleStep(0.001)
        self.gaussianR2SpinBox.setValue(GAUSSIAN_R2_MIN)
        r2_row.addWidget(self.gaussianR2SpinBox)
        layout.addLayout(r2_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Max Display Rate (FPS):"))
        self.maxFpsSpinBox = QSpinBox()
        self.maxFpsSpinBox.setRange(1, 60)
        self.maxFpsSpinBox.setValue(DEFAULT_MAX_DISPLAY_FPS)
        fps_row.addWidget(self.maxFpsSpinBox)
        layout.addLayout(fps_row)

        avg_row = QHBoxLayout()
        avg_row.addWidget(QLabel("Averaged Frames:"))
        self.averagingSpinBox = QSpinBox()
        self.averagingSpinBox.setRange(1, 20)
        self.averagingSpinBox.setValue(1)
        avg_row.addWidget(self.averagingSpinBox)
        layout.addLayout(avg_row)

        self.calcAreaCheckbox = QCheckBox("Restrict Calculation Area")
        layout.addWidget(self.calcAreaCheckbox)
        calc_row = QHBoxLayout()
        calc_row.addWidget(QLabel("Half-Size (px):"))
        self.calcAreaSpinBox = QSpinBox()
        self.calcAreaSpinBox.setRange(20, 2000)
        self.calcAreaSpinBox.setValue(200)
        calc_row.addWidget(self.calcAreaSpinBox)
        layout.addLayout(calc_row)

        clip_row = QHBoxLayout()
        clip_row.addWidget(QLabel("Clip Level (% of peak):"))
        self.clipLevelSpinBox = QSpinBox()
        self.clipLevelSpinBox.setRange(1, 99)
        self.clipLevelSpinBox.setValue(DEFAULT_CLIP_LEVEL_PERCENT)
        clip_row.addWidget(self.clipLevelSpinBox)
        layout.addLayout(clip_row)

        hold_row = QHBoxLayout()
        self.holdMaxCheckbox = QCheckBox("Hold Maximum (Peak Hold)")
        hold_row.addWidget(self.holdMaxCheckbox)
        self.resetHoldButton = QPushButton("Reset Hold")
        hold_row.addWidget(self.resetHoldButton)
        layout.addLayout(hold_row)

        psf_row = QHBoxLayout()
        psf_row.addWidget(QLabel("Instrument PSF (µm):"))
        self.psfSpinBox = QDoubleSpinBox()
        self.psfSpinBox.setRange(0.0, 1000.0)
        self.psfSpinBox.setDecimals(2)
        self.psfSpinBox.setValue(0.0)
        psf_row.addWidget(self.psfSpinBox)
        layout.addLayout(psf_row)

        layout.addWidget(QLabel("--- Power / Wavelength (manual calibration) ---"))
        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Wavelength (nm):"))
        self.wavelengthSpinBox = QDoubleSpinBox()
        self.wavelengthSpinBox.setRange(100.0, 3000.0)
        self.wavelengthSpinBox.setValue(780.0)
        wl_row.addWidget(self.wavelengthSpinBox)
        layout.addLayout(wl_row)

        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("Power Cal. Factor (µW/count):"))
        self.powerCalSpinBox = QDoubleSpinBox()
        self.powerCalSpinBox.setRange(0.0, 1000.0)
        self.powerCalSpinBox.setDecimals(6)
        self.powerCalSpinBox.setValue(0.0)
        cal_row.addWidget(self.powerCalSpinBox)
        layout.addLayout(cal_row)

        self.powerUnitCombo = QComboBox()
        self.powerUnitCombo.addItems(["µW", "mW", "dBm"])
        layout.addWidget(self.powerUnitCombo)


class PropagationPanel(QWidget):

    HEADERS = [
        "Use", "Z (mm)", "D4σ X (µm)", "D4σ Y (µm)",
        "Gaussian X (µm)", "Gaussian Y (µm)", "Sat. (%)",
        "Clipped", "Frames", "CV (%)"
    ]

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.samples = []
        root = QVBoxLayout(self)

        howto = QLabel(
            "<b>How to use this:</b> 1) Move the camera (or the beam/focusing lens) to a new "
            "Z position along the beam path and enter that position below. 2) Wait for a "
            "steady, unsaturated beam. 3) Click \u201cLog Stable Current Measurement\u201d -- it "
            "needs 5 consistent live frames within the last 2 seconds, so it will tell you to "
            "wait if the beam just moved or is still fluctuating. 4) Repeat at several "
            "different Z positions -- at least 4 for a provisional fit, ideally 10+ spread on "
            "both sides of the waist for real coverage. 5) Read the fit results and residual "
            "plot on the right."
        )
        howto.setWordWrap(True)
        howto.setStyleSheet("background:#eaf2fb;color:#1a3a5c;padding:7px;border:1px solid #a9c6e8;")
        root.addWidget(howto)

        warning = QLabel(
            "This tool fits a beam caustic from multiple planes. It is not automatically "
            "ISO 11146 compliant. Use one consistent width method, adequate axial coverage, "
            "background correction, no clipping/saturation, and a known wavelength."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("background:#fff3cd;color:#5f4700;padding:7px;border:1px solid #d8bd68;")
        root.addWidget(warning)

        config = QGroupBox("Measurement configuration")
        form = QFormLayout(config)
        self.zSpin = QDoubleSpinBox()
        self.zSpin.setRange(-1_000_000.0, 1_000_000.0)
        self.zSpin.setDecimals(4)
        self.zSpin.setSuffix(" mm")
        form.addRow("Current Z position:", self.zSpin)

        self.methodCombo = QComboBox()
        self.methodCombo.addItems([
            "D4σ camera axes (2D second moment)",
            "Gaussian-fit 1/e² diameters (non-ISO comparison)",
        ])
        self.methodCombo.currentIndexChanged.connect(self.refit)
        form.addRow("Fit width method:", self.methodCombo)

        self.wavelengthSpin = QDoubleSpinBox()
        self.wavelengthSpin.setRange(100.0, 3000.0)
        self.wavelengthSpin.setDecimals(2)
        self.wavelengthSpin.setSuffix(" nm")
        self.wavelengthSpin.setValue(main_window.advanced_panel.wavelengthSpinBox.value())
        self.wavelengthSpin.valueChanged.connect(self.refit)
        form.addRow("Wavelength:", self.wavelengthSpin)

        self.directionCombo = QComboBox()
        self.directionCombo.addItems(["Increasing Z is downstream", "Increasing Z is upstream"])
        form.addRow("Stage convention:", self.directionCombo)

        self.lensEdit = QDoubleSpinBox()
        self.lensEdit.setRange(0.0, 100000.0)
        self.lensEdit.setDecimals(2)
        self.lensEdit.setSuffix(" mm")
        self.lensEdit.setSpecialValueText("Not recorded")
        form.addRow("Focusing-lens focal length:", self.lensEdit)
        root.addWidget(config)

        body = QHBoxLayout()
        left = QVBoxLayout()
        button_row = QHBoxLayout()
        self.logButton = QPushButton("Log Stable Current Measurement")
        self.logButton.clicked.connect(self.log_current)
        self.deleteButton = QPushButton("Delete Selected")
        self.deleteButton.clicked.connect(self.delete_selected)
        self.clearButton = QPushButton("Clear All")
        self.clearButton.clicked.connect(self.clear_all)
        button_row.addWidget(self.logButton)
        button_row.addWidget(self.deleteButton)
        button_row.addWidget(self.clearButton)
        left.addLayout(button_row)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._table_changed)
        left.addWidget(self.table)

        export_row = QHBoxLayout()
        self.saveSessionButton = QPushButton("Save Session JSON")
        self.loadSessionButton = QPushButton("Load Session JSON")
        self.exportCsvButton = QPushButton("Export CSV")
        self.saveSessionButton.clicked.connect(self.save_session)
        self.loadSessionButton.clicked.connect(self.load_session)
        self.exportCsvButton.clicked.connect(self.export_csv)
        export_row.addWidget(self.saveSessionButton)
        export_row.addWidget(self.loadSessionButton)
        export_row.addWidget(self.exportCsvButton)
        left.addLayout(export_row)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(211)
        self.resid_ax = self.figure.add_subplot(212)
        right.addWidget(self.canvas, 4)
        self.resultLabel = QLabel("Add at least four distinct Z positions for a provisional fit.")
        self.resultLabel.setWordWrap(True)
        self.resultLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right.addWidget(self.resultLabel, 2)
        body.addLayout(right, 4)
        root.addLayout(body)

    def log_current(self):
        z = float(self.zSpin.value())
        if any(abs((s.z_mm or 0.0) - z) < 1e-9 for s in self.samples):
            answer = QMessageBox.question(
                self, "Duplicate Z position",
                "A point already exists at this Z. Add a replicate anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        sample, message = self.main_window.get_stable_propagation_sample(z)
        if sample is None:
            QMessageBox.warning(self, "Measurement not logged", message)
            return
        if sample.clipped or sample.saturation_fraction >= 0.95:
            answer = QMessageBox.question(
                self, "Quality warning",
                "The current sample is clipped or saturated. Log it as excluded for documentation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            sample.valid = False
        self.samples.append(sample)
        self._append_row(sample)
        self.refit()

    def _append_row(self, sample):
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        use_item = QTableWidgetItem("")
        use_item.setFlags(use_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        use_item.setCheckState(Qt.CheckState.Checked if sample.valid else Qt.CheckState.Unchecked)
        self.table.setItem(row, 0, use_item)
        values = [
            sample.z_mm, sample.d4sigma_x_um, sample.d4sigma_y_um,
            sample.gaussian_x_um, sample.gaussian_y_um,
            100.0 * sample.saturation_fraction,
            "Yes" if sample.clipped else "No", sample.source_frame_count,
            sample.width_cv_percent,
        ]
        for col, value in enumerate(values, start=1):
            if value is None:
                text = ""
            elif isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            item = QTableWidgetItem(text)
            if col != 1:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, col, item)
        self.table.blockSignals(False)

    def _table_changed(self, item):
        row = item.row()
        if not (0 <= row < len(self.samples)):
            return
        if item.column() == 0:
            self.samples[row].valid = item.checkState() == Qt.CheckState.Checked
        elif item.column() == 1:
            try:
                self.samples[row].z_mm = float(item.text())
            except ValueError:
                pass
        self.refit()

    def delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            del self.samples[row]
        self.refit()

    def clear_all(self):
        if self.samples and QMessageBox.question(
            self, "Clear propagation session", "Remove every logged point?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.samples.clear()
        self.table.setRowCount(0)
        self.refit()

    def _selected_data(self):
        use_d4 = self.methodCombo.currentIndex() == 0
        rows = []
        for row, sample in enumerate(self.samples):
            checked = self.table.item(row, 0).checkState() == Qt.CheckState.Checked
            if not checked or sample.z_mm is None:
                continue
            x = sample.d4sigma_x_um if use_d4 else sample.gaussian_x_um
            y = sample.d4sigma_y_um if use_d4 else sample.gaussian_y_um
            if x is not None and y is not None and x > 0 and y > 0:
                rows.append((sample.z_mm, x, y))
        return rows

    def refit(self, *args):
        rows = self._selected_data()
        self.ax.clear()
        self.resid_ax.clear()
        self.ax.set_xlabel("Z position (mm)")
        self.ax.set_ylabel("Full beam diameter (µm)")
        self.resid_ax.set_xlabel("Z position (mm)")
        self.resid_ax.set_ylabel("Radius residual (µm)")
        if len(rows) < 4 or len({r[0] for r in rows}) < 4:
            self.resultLabel.setText(
                f"Accepted points: {len(rows)}. Add at least four distinct Z positions for a provisional fit; "
                "ten appropriately distributed planes are a later coverage target, not a guarantee of compliance."
            )
            if rows:
                z, x, y = map(np.asarray, zip(*rows))
                self.ax.scatter(z, x, marker="o", label="X")
                self.ax.scatter(z, y, marker="s", label="Y")
                self.ax.legend()
            self.canvas.draw_idle()
            return

        z, dx, dy = map(np.asarray, zip(*rows))
        wavelength = float(self.wavelengthSpin.value())
        fit_x = fit_propagation_axis(z, dx, wavelength)
        fit_y = fit_propagation_axis(z, dy, wavelength)
        self.ax.scatter(z, dx, marker="o", label="X diameter")
        self.ax.scatter(z, dy, marker="s", label="Y diameter")
        messages = []
        for name, fit, color in (("X", fit_x, "C0"), ("Y", fit_y, "C1")):
            if fit is None:
                messages.append(f"{name}: fit failed or insufficient independent data.")
                continue
            z_grid = np.linspace(float(z.min()), float(z.max()), 500)
            radius_grid = propagation_radius_model(
                z_grid, fit["w0_um"], fit["z0_mm"], fit["m2"], wavelength
            )
            self.ax.plot(z_grid, 2.0 * radius_grid, color=color, label=f"{name} fit")
            observed = fit["observed_radius_um"]
            predicted = fit["predicted_radius_um"]
            self.resid_ax.scatter(fit["z"], observed - predicted, color=color, label=name)
            coverage_ok = (fit["near_count"] >= 5 and fit["far_count"] >= 5 and
                           fit["left_sampled"] and fit["right_sampled"])
            status = "adequate ISO-style coverage guidance" if coverage_ok else "coverage incomplete"
            physical = "physical" if fit["physical"] else "PHYSICALLY INCONSISTENT"
            messages.append(
                f"<b>{name}</b>: w₀={fit['w0_um']:.3f}±{fit['w0_error_um']:.3f} µm; "
                f"z₀={fit['z0_mm']:.4g}±{fit['z0_error_mm']:.3g} mm; "
                f"M²={fit['m2']:.4f}±{fit['m2_error']:.4f}; "
                f"zR={fit['z_r_mm']:.4g} mm; half/full divergence="
                f"{fit['theta_half_mrad']:.4g}/{fit['theta_full_mrad']:.4g} mrad; "
                f"NRMSE={100*fit['nrmse']:.3g}%; near/far={fit['near_count']}/{fit['far_count']}; "
                f"{status}; {physical}."
            )
        self.ax.legend()
        self.ax.grid(True, alpha=0.25)
        self.resid_ax.axhline(0.0, linewidth=1)
        self.resid_ax.legend()
        self.resid_ax.grid(True, alpha=0.25)
        if len(rows) < 10:
            messages.insert(0, f"<b>Provisional fit:</b> only {len(rows)} accepted points. Do not treat this as a final M² result.")
        messages.append(
            "Coverage is evaluated from the current fitted z₀ and zR and is guidance only. "
            "A good-looking curve does not prove ISO compliance or remove background/sampling uncertainty."
        )
        self.resultLabel.setText("<br>".join(messages))
        self.canvas.draw_idle()

    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save propagation session", "propagation_session.json", "JSON (*.json)")
        if not path:
            return
        payload = {
            "schema_version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "wavelength_nm": self.wavelengthSpin.value(),
            "method": self.methodCombo.currentText(),
            "direction": self.directionCombo.currentText(),
            "lens_focal_length_mm": self.lensEdit.value() or None,
            "samples": [asdict(s) for s in self.samples],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load propagation session", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded = [PropagationSample(**row) for row in payload.get("samples", [])]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.samples = loaded
        self.table.setRowCount(0)
        for sample in self.samples:
            self._append_row(sample)
        self.wavelengthSpin.setValue(float(payload.get("wavelength_nm", self.wavelengthSpin.value())))
        self.refit()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export propagation CSV", "propagation_points.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(PropagationSample(timestamp="")).keys()))
                writer.writeheader()
                for sample in self.samples:
                    writer.writerow(asdict(sample))
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


class CameraCommandBridge(QObject):
    connect_requested = pyqtSignal(object)
    disconnect_requested = pyqtSignal()
    exposure_requested = pyqtSignal(float)
    dark_requested = pyqtSignal(int)
    stop_requested = pyqtSignal()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi(resource_path("GUI1.ui"), self)
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")

        icon_path = resource_path("app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.presets_path = resource_path("beam_profiler_presets.json")

        self.camera_panel = CameraPanel()
        self.control_panel = ControlPanel()
        self.analyze_panel = AnalyzePanel()
        self.advanced_panel = AdvancedSettingsPanel()
        self.stability_panel = StabilityPanel()
        self.propagation_panel = PropagationPanel(self)

        self.camera_panel.snapButton.clicked.connect(self.snap_to_beam)

        self.graph_panel_x = GraphPanel()
        self.graph_panel_x.ax.set_title("Live Beam Profile (X-Axis)", fontsize=10)

        self.graph_panel_y = GraphPanel()
        self.graph_panel_y.ax.set_title("Live Beam Profile (Y-Axis)", fontsize=10)

        self.sub_camera = self.add_panel(self.camera_panel, "Camera Panel", 20, 20, 500, 450)
        self.sub_graph_x = self.add_panel(self.graph_panel_x, "Beam Profile (X)", 20, 480, 340, 210)
        self.sub_graph_y = self.add_panel(self.graph_panel_y, "Beam Profile (Y)", 370, 480, 340, 210)
        self.sub_controls = self.add_panel(self.control_panel, "Controls", 1020, 20, 360, 580)
        self.sub_waist = self.add_panel(self.analyze_panel, "Calculations", 530, 20, 480, 450)
        self.sub_advanced = self.add_panel(self.advanced_panel, "Advanced Settings", 950, 20, 300, 500, visible=False)
        self.sub_stability = self.add_panel(self.stability_panel, "Beam Stability", 970, 60, 320, 260, visible=False)
        self.sub_propagation = self.add_panel(self.propagation_panel, "Propagation / M² Analysis", 60, 60, 1080, 680, visible=False)

        self.control_panel.connectCameraButton.clicked.connect(self.connect_selected_camera)
        self.control_panel.scanCamerasButton.clicked.connect(self.scan_for_cameras)
        self.control_panel.captureButton.clicked.connect(self.capture_image)
        self.control_panel.exposureSpinBox.valueChanged.connect(self.update_camera_settings)
        self.control_panel.opacitySlider.valueChanged.connect(self.update_raw_data_opacity)
        self.control_panel.overlayHeightSlider.valueChanged.connect(self.update_overlay_height)
        self.camera_panel.gridLabelSizeSpinBox.valueChanged.connect(self.update_overlay_text_scale)

        self.camera_panel.zoomSpinBox.valueChanged.connect(self.control_panel.zoomSpinBox.setValue)
        self.control_panel.zoomSpinBox.valueChanged.connect(self.camera_panel.zoomSpinBox.setValue)
        self.analyze_panel.logButton.toggled.connect(self.toggle_logging)
        self.control_panel.percentModeCheckbox.toggled.connect(self.update_intensity_mode)

        self.advanced_panel.averagingSpinBox.valueChanged.connect(self.update_averaging)
        self.advanced_panel.resetHoldButton.clicked.connect(self.reset_hold_maximum)
        self.advanced_panel.resetDefaultsButton.clicked.connect(self.reset_advanced_defaults)
        self.advanced_panel.captureDarkButton.clicked.connect(self.capture_dark_frame)

        self.overlay_curve_height = OVERLAY_CURVE_HEIGHT
        self.overlay_text_scale = 0.5

        self.camera = None
        self._camera_connected = False
        self._camera_info_text = "No camera connected."
        self._camera_detector_size = None
        self.current_camera_backend_name = None
        self.current_exposure_ms = 5.0
        self.photo_count = 0

        self._frame_mailbox = LatestFrameMailbox()
        self._camera_thread = QThread(self)
        self._camera_worker = CameraWorker(self._frame_mailbox)
        self._camera_worker.moveToThread(self._camera_thread)
        self._camera_bridge = CameraCommandBridge(self)
        self._camera_thread.started.connect(self._camera_worker.initialize)
        self._camera_bridge.connect_requested.connect(self._camera_worker.connect_camera)
        self._camera_bridge.disconnect_requested.connect(self._camera_worker.disconnect_camera)
        self._camera_bridge.exposure_requested.connect(self._camera_worker.set_exposure_ms)
        self._camera_bridge.dark_requested.connect(self._camera_worker.request_dark_frame)
        self._camera_bridge.stop_requested.connect(self._camera_worker.stop)
        self._camera_worker.connected.connect(self._on_camera_connected)
        self._camera_worker.disconnected.connect(self._on_camera_disconnected)
        self._camera_worker.error.connect(self._on_camera_error)
        self._camera_worker.dark_frame_ready.connect(self._on_dark_frame_ready)
        self._camera_worker.dark_frame_failed.connect(self._on_dark_frame_failed)
        self._camera_worker.stopped.connect(self._camera_thread.quit)
        self._camera_thread.start()

        self.sensor_type = None
        self.sensor_bit_depth = 10
        self.sensor_max_raw = SENSOR_ASSUMED_MAX_RAW
        self.sensor_display_divisor = SENSOR_DISPLAY_DIVISOR
        self.min_signal_above_background = max(1, int(0.2 * self.sensor_max_raw))

        self._last_frame_ts = None
        self._fps_smoothed = None
        self._frame_time_smoothed = None
        self._display_fps_smoothed = None
        self._last_display_ts = None
        self._analysis_ms_smoothed = None
        self._last_worker_error = ""

        self._last_h_info = None
        self._last_v_info = None
        self._last_second_moment = None
        self._latest_measurement = None
        self._latest_quality = None
        self._latest_propagation_sample = None
        self._propagation_history = collections.deque(maxlen=120)

        self._log_file = None
        self._log_writer = None
        self._log_row_count = 0

        self.last_cx = 0
        self.last_cy = 0

        self._numbers_x_cache = None
        self._numbers_y_cache = None

        self._last_gaussian_popt_x = None
        self._last_gaussian_popt_y = None
        self._last_gaussian_r2_x = None
        self._last_gaussian_r2_y = None

        self._warned_possible_clipping = False
        self._warned_no_tifffile = False

        self._colormap_luts = {}

        self._frame_buffer = collections.deque(maxlen=1)
        self._frame_sum = None
        self._analysis_buffer = None

        self._hold_max_frame = None

        self._latest_raw_frame = None
        self._latest_quality = None
        self._latest_measurement = None
        self._dark_frame = None
        self._dark_metadata = None

        self._last_advanced_metrics_ts = None
        self._advanced_metrics_age_ms = None
        self._cached_background_level = 0.0
        self._cached_background_noise = 0.0
        self._cached_percentile_background = 0.0
        self._cached_hot_mask = None
        self._cached_hot_count = 0
        self._cached_hot_coords = (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
        self._cached_sensor_edge = False
        self._cached_roi_edge = False
        self._cached_moment = None

        self._latest_processed_frame = None

        self._last_video_label_size = (None, None)

        self.frame_counter = 0
        self._consecutive_bad_frames = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.render_video = True
        self.render_x_plot = True
        self.render_y_plot = True
        self.render_stability = True

        self.dark_mode = False

        self._build_menu_and_toolbar()
        self._load_settings()
        self.scan_for_cameras()
        self.connect_selected_camera(show_errors=False)
        if not self.timer.isActive():
            self.timer.start(33)


    def _build_menu_and_toolbar(self):
        self.toolbar = self.addToolBar("View Controls")

        self.action_video = QAction("Live Video", self)
        self.action_video.setCheckable(True)
        self.action_video.setChecked(True)
        self.action_video.toggled.connect(self.toggle_video)

        self.action_x = QAction("X Profile", self)
        self.action_x.setCheckable(True)
        self.action_x.setChecked(True)
        self.action_x.toggled.connect(self.toggle_x_plot)

        self.action_y = QAction("Y Profile", self)
        self.action_y.setCheckable(True)
        self.action_y.setChecked(True)
        self.action_y.toggled.connect(self.toggle_y_plot)

        self.action_controls = QAction("Controls", self)
        self.action_controls.setCheckable(True)
        self.action_controls.setChecked(True)
        self.action_controls.toggled.connect(self.toggle_controls)

        self.action_waist = QAction("Calculations", self)
        self.action_waist.setCheckable(True)
        self.action_waist.setChecked(True)
        self.action_waist.toggled.connect(self.toggle_waist)

        self.action_advanced = QAction("Advanced Settings", self)
        self.action_advanced.setCheckable(True)
        self.action_advanced.setChecked(False)
        self.action_advanced.toggled.connect(self.toggle_advanced)

        self.action_stability = QAction("Beam Stability", self)
        self.action_stability.setCheckable(True)
        self.action_stability.setChecked(False)
        self.action_stability.toggled.connect(self.toggle_stability)

        self.action_propagation = QAction("Propagation / M² Analysis", self)
        self.action_propagation.setCheckable(True)
        self.action_propagation.setChecked(False)
        self.action_propagation.toggled.connect(self.toggle_propagation)

        panel_actions = (self.action_video, self.action_x, self.action_y, self.action_controls,
                          self.action_waist, self.action_advanced, self.action_stability,
                          self.action_propagation)
        for action in panel_actions:
            self.toolbar.addAction(action)

        self.toolbar.addSeparator()

        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        capture_action = QAction("Capture Image", self)
        capture_action.setShortcut("Ctrl+S")
        capture_action.triggered.connect(self.capture_image)
        file_menu.addAction(capture_action)

        self.action_logging_menu = QAction("Start/Stop Logging", self)
        self.action_logging_menu.setCheckable(True)
        self.action_logging_menu.toggled.connect(self.analyze_panel.logButton.setChecked)
        self.analyze_panel.logButton.toggled.connect(self.action_logging_menu.setChecked)
        file_menu.addAction(self.action_logging_menu)

        file_menu.addSeparator()
        save_preset_action = QAction("Save Preset As...", self)
        save_preset_action.triggered.connect(self.save_preset_as)
        file_menu.addAction(save_preset_action)

        self.presets_menu = file_menu.addMenu("Load Preset")

        delete_preset_action = QAction("Delete Preset...", self)
        delete_preset_action.triggered.connect(self.delete_preset)
        file_menu.addAction(delete_preset_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menu_bar.addMenu("&View")
        for action in panel_actions:
            view_menu.addAction(action)
        view_menu.addSeparator()

        self.action_dark_mode = QAction("Dark Mode", self)
        self.action_dark_mode.setCheckable(True)
        self.action_dark_mode.toggled.connect(self.toggle_theme)
        view_menu.addAction(self.action_dark_mode)

        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction(self.action_propagation)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        status_bar = self.statusBar()
        self.status_camera_label = QLabel("Camera: Not connected")
        self.status_capture_label = QLabel("Last capture: none")
        status_bar.addWidget(self.status_camera_label)
        self.status_performance_label = QLabel("Acq -- | Display -- | Analysis -- | Drop 0 | Age -- | Errors 0")
        status_bar.addPermanentWidget(self.status_performance_label)
        status_bar.addPermanentWidget(self.status_capture_label)

        self._rebuild_presets_menu()

        self.sub_camera.toggle_action = self.action_video
        self.sub_graph_x.toggle_action = self.action_x
        self.sub_graph_y.toggle_action = self.action_y
        self.sub_controls.toggle_action = self.action_controls
        self.sub_waist.toggle_action = self.action_waist
        self.sub_advanced.toggle_action = self.action_advanced
        self.sub_stability.toggle_action = self.action_stability
        self.sub_propagation.toggle_action = self.action_propagation


    def get_stable_propagation_sample(self, z_mm):
        now = time.time()
        recent = [s for ts, s in self._propagation_history if now - ts <= 2.0 and s.valid]
        if len(recent) < 5:
            return None, "Need at least five valid live frames from the last two seconds. Wait for a stable beam and try again."
        fields = ("d4sigma_x_um", "d4sigma_y_um", "d4sigma_major_um", "d4sigma_minor_um",
                  "orientation_deg", "gaussian_x_um", "gaussian_y_um", "r2_x", "r2_y")
        values = {}
        for field in fields:
            arr = np.asarray([getattr(s, field) for s in recent if getattr(s, field) is not None], dtype=float)
            values[field] = float(np.median(arr)) if arr.size else None
        cvs = []
        for field in ("d4sigma_x_um", "d4sigma_y_um"):
            arr = np.asarray([getattr(s, field) for s in recent if getattr(s, field) is not None], dtype=float)
            if arr.size >= 5 and np.mean(arr) > 0:
                cvs.append(100.0 * float(np.std(arr, ddof=1)) / float(np.mean(arr)))
        cv = max(cvs) if cvs else None
        sample = PropagationSample(
            timestamp=datetime.now().isoformat(timespec="milliseconds"), z_mm=float(z_mm),
            saturation_fraction=max(s.saturation_fraction for s in recent),
            clipped=any(s.clipped for s in recent), valid=True,
            exposure_ms=float(np.median([s.exposure_ms for s in recent])),
            pixel_pitch_um=float(np.median([s.pixel_pitch_um for s in recent])),
            wavelength_nm=float(np.median([s.wavelength_nm for s in recent])),
            source_frame_count=len(recent), width_cv_percent=cv, **values,
        )
        if sample.d4sigma_x_um is None or sample.d4sigma_y_um is None:
            sample.valid = False
            return None, "No valid 2D second-moment widths are available. Check signal, background, and clipping."
        if cv is not None and cv > 2.0:
            answer = QMessageBox.question(
                self, "Beam is fluctuating",
                f"Recent D4σ widths have a {cv:.2f}% coefficient of variation. Log this unstable point anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None, "Point not logged because the recent beam width was unstable."
        return sample, ""

    def toggle_video(self, checked):
        self.render_video = checked
        self.sub_camera.setVisible(checked)

    def toggle_x_plot(self, checked):
        self.render_x_plot = checked
        self.sub_graph_x.setVisible(checked)

    def toggle_y_plot(self, checked):
        self.render_y_plot = checked
        self.sub_graph_y.setVisible(checked)

    def toggle_controls(self, checked):
        self.sub_controls.setVisible(checked)

    def toggle_waist(self, checked):
        self.sub_waist.setVisible(checked)

    def toggle_advanced(self, checked):
        self.sub_advanced.setVisible(checked)

    def toggle_stability(self, checked):
        self.render_stability = checked
        self.sub_stability.setVisible(checked)

    def toggle_propagation(self, checked):
        self.sub_propagation.setVisible(checked)

    def add_panel(self, widget, title, x, y, w, h, visible=True):
        sub = PersistentSubWindow()
        sub.setWidget(widget)
        self.mdiArea.addSubWindow(sub)
        sub.setWindowTitle(title)
        sub.setGeometry(x, y, w, h)
        sub.setVisible(visible)
        return sub


    def toggle_theme(self, checked):
        self.dark_mode = checked
        app = QApplication.instance()
        app.setStyleSheet(DARK_QSS if checked else LIGHT_QSS)
        for panel in (self.graph_panel_x, self.graph_panel_y, self.stability_panel):
            panel.set_theme(checked)

    def show_about(self):
        if self._camera_connected:
            cam_text = self._camera_info_text
            size = self._camera_detector_size
            if size is not None:
                w_px, h_px = size
                pitch = self.analyze_panel.pixel_pitch_input.value()
                fov_w_um = w_px * pitch; fov_h_um = h_px * pitch
                cam_text += (f"\nDetected sensor size: {w_px} x {h_px} px\n"
                             f"Pixel pitch in use: {pitch:.2f} µm/px\n"
                             f"Measurable field of view: {fov_w_um:.0f} x {fov_h_um:.0f} µm "
                             f"({fov_w_um/1000:.2f} x {fov_h_um/1000:.2f} mm)")
            else:
                cam_text += "\nDetected sensor size: unavailable for this backend"
        else:
            cam_text = "No camera connected."
        text = (f"{APP_NAME} {APP_VERSION}\n\n{cam_text}\n\n"
                "Developed by Jason Liang | VOILA Lab, UC Berkeley\n\nCredits:\nPrimary Mentor: "
                "Yuanlong Zhao\nSpecial Thanks: Vasilisa, Xue, and Dr. Meng for technical guidance and support!")
        QMessageBox.about(self, f"About {APP_NAME}", text)


    def scan_for_cameras(self):
        backend_name = self.control_panel.cameraBackendCombo.currentText()
        self.control_panel.deviceCombo.clear()

        if backend_name == "Generic Webcam / OpenCV":
            devices = OpenCVCameraBackend.list_available_devices()
        elif backend_name == "Thorlabs Scientific / Zelux":
            devices = ThorlabsScientificCameraBackend.list_available_devices()
        elif backend_name == "IDS uEye / UC480":
            devices = UC480CameraBackend.list_available_devices()
        else:
            devices = []

        if not devices:
            self.control_panel.deviceCombo.addItem("No devices found (will use default)", None)
        else:
            for display_name, identifier in devices:
                self.control_panel.deviceCombo.addItem(display_name, identifier)

    def connect_selected_camera(self, show_errors=True):
        backend_name = self.control_panel.cameraBackendCombo.currentText()
        device_id = self.control_panel.deviceCombo.currentData()
        device_text = self.control_panel.deviceCombo.currentText()
        if device_id is None and device_text.startswith("No devices found"):
            self._camera_bridge.disconnect_requested.emit()
            self._camera_connected = False
            self.status_camera_label.setText(f"Camera: No {backend_name} device detected")
            self._show_no_signal()
            return
        self._camera_connected = False
        self.current_camera_backend_name = backend_name
        self.status_camera_label.setText(f"Camera: Connecting ({backend_name})")
        self.camera_panel.videoLabel.setText(f"Connecting: {backend_name}")
        self._frame_mailbox.clear()
        self._camera_bridge.connect_requested.emit({
            "backend_name": backend_name, "device_id": device_id,
            "device_text": device_text, "exposure_ms": self.control_panel.exposureSpinBox.value(),
            "show_errors": bool(show_errors),
        })

    @pyqtSlot(object)
    def _on_camera_connected(self, info):
        self._camera_connected = True
        self.current_camera_backend_name = info["backend_name"]
        self.sensor_type = info.get("sensor_type")
        self.sensor_bit_depth = int(info["sensor_bit_depth"])
        self.sensor_max_raw = int(info["sensor_max_raw"])
        self.sensor_display_divisor = (self.sensor_max_raw + 1) / 256.0
        self.min_signal_above_background = max(1, int(0.2 * self.sensor_max_raw))
        self._camera_info_text = info.get("info_text", "")
        self._camera_detector_size = info.get("detector_size")
        if info.get("pixel_pitch_um") is not None:
            self.analyze_panel.pixel_pitch_input.setValue(float(info["pixel_pitch_um"]))
        self.graph_panel_x.set_intensity_mode(self.control_panel.percentModeCheckbox.isChecked(), self.sensor_max_raw)
        self.graph_panel_y.set_intensity_mode(self.control_panel.percentModeCheckbox.isChecked(), self.sensor_max_raw)
        self._reset_camera_session_state()
        self.camera_panel.videoLabel.setText(f"Connected: {self.current_camera_backend_name}")
        self.status_camera_label.setText(f"Camera: Connected ({self.current_camera_backend_name})")

    def _reset_camera_session_state(self):
        self._last_frame_ts = None; self._fps_smoothed = None; self._frame_time_smoothed = None
        self._display_fps_smoothed = None; self._last_display_ts = None; self._analysis_ms_smoothed = None
        self._hold_max_frame = None; self._latest_raw_frame = None
        self._frame_buffer = collections.deque(maxlen=self.advanced_panel.averagingSpinBox.value())
        self._frame_sum = None; self._analysis_buffer = None
        self._numbers_x_cache = None; self._numbers_y_cache = None
        self._last_gaussian_popt_x = None; self._last_gaussian_popt_y = None
        self._last_gaussian_r2_x = None; self._last_gaussian_r2_y = None
        self._last_h_info = None; self._last_v_info = None; self._latest_measurement = None
        self._latest_quality = None; self._warned_possible_clipping = False; self._consecutive_bad_frames = 0

    @pyqtSlot()
    def _on_camera_disconnected(self):
        self._camera_connected = False
        self.current_camera_backend_name = None
        self.status_camera_label.setText("Camera: Disconnected")
        self._show_no_signal()

    @pyqtSlot(str, str)
    def _on_camera_error(self, message, detail):
        self._last_worker_error = message
        print(f"[camera worker] {message}\n{detail}")
        self.status_camera_label.setText(f"Camera error: {message}")

    @pyqtSlot(object, object)
    def _on_dark_frame_ready(self, dark, metadata):
        self._dark_frame = np.asarray(dark).copy()
        self._dark_metadata = dict(metadata)
        self.advanced_panel.darkEnableCheckbox.setChecked(True)
        self.advanced_panel.darkStatusLabel.setText("Dark frame: valid and enabled")
        self._frame_buffer.clear(); self._frame_sum = None; self._analysis_buffer = None; self._hold_max_frame = None

    @pyqtSlot(str)
    def _on_dark_frame_failed(self, message):
        QMessageBox.warning(self, "Dark Frame", message)


    def _collect_settings_dict(self):
        return {
            "camera_backend": self.control_panel.cameraBackendCombo.currentText(),
            "exposure_ms": self.control_panel.exposureSpinBox.value(),
            "grid_label_size": self.camera_panel.gridLabelSizeSpinBox.value(),
            "pixel_pitch": self.analyze_panel.pixel_pitch_input.value(),
            "colormap": self.control_panel.cmapCombo.currentText(),
            "opacity": self.control_panel.opacitySlider.value(),
            "zoom": self.control_panel.zoomSlider.value(),
            "overlay_height": self.overlay_curve_height,
            "percent_mode": self.control_panel.percentModeCheckbox.isChecked(),
            "max_display_fps": self.advanced_panel.maxFpsSpinBox.value(),
            "gaussian_r2_min": self.advanced_panel.gaussianR2SpinBox.value(),
            "averaging": self.advanced_panel.averagingSpinBox.value(),
            "clip_level": self.advanced_panel.clipLevelSpinBox.value(),
            "calc_area_enabled": self.advanced_panel.calcAreaCheckbox.isChecked(),
            "calc_area_half_size": self.advanced_panel.calcAreaSpinBox.value(),
            "psf_um": self.advanced_panel.psfSpinBox.value(),
            "wavelength_nm": self.advanced_panel.wavelengthSpinBox.value(),
            "power_cal_factor": self.advanced_panel.powerCalSpinBox.value(),
            "power_unit": self.advanced_panel.powerUnitCombo.currentText(),
            "dark_frame_enabled": self.advanced_panel.darkEnableCheckbox.isChecked(),
            "minimum_snr": self.advanced_panel.minSnrSpinBox.value(),
            "bad_pixel_mask_enabled": self.advanced_panel.badPixelMaskCheckbox.isChecked(),
            "hold_maximum_enabled": self.advanced_panel.holdMaxCheckbox.isChecked(),
            "dark_mode": self.dark_mode,
        }

    def _apply_settings_dict(self, d):
        backend = d.get("camera_backend", "Thorlabs Scientific / Zelux")
        idx0 = self.control_panel.cameraBackendCombo.findText(backend)
        if idx0 >= 0:
            self.control_panel.cameraBackendCombo.setCurrentIndex(idx0)

        self.control_panel.exposureSpinBox.setValue(float(d.get("exposure_ms", 5.0)))
        self.camera_panel.gridLabelSizeSpinBox.setValue(int(d.get("grid_label_size", 50)))
        self.analyze_panel.pixel_pitch_input.setValue(float(d.get("pixel_pitch", 3.45)))

        cmap = d.get("colormap", "Inferno")
        idx = self.control_panel.cmapCombo.findText(cmap)
        if idx >= 0:
            self.control_panel.cmapCombo.setCurrentIndex(idx)

        self.control_panel.opacitySlider.setValue(int(d.get("opacity", DEFAULT_RAW_DATA_OPACITY_PERCENT)))
        self.control_panel.zoomSlider.setValue(int(d.get("zoom", 100)))
        self.control_panel.overlayHeightSlider.setValue(int(d.get("overlay_height", OVERLAY_CURVE_HEIGHT)))
        self.control_panel.percentModeCheckbox.setChecked(bool(d.get("percent_mode", False)))

        self.advanced_panel.maxFpsSpinBox.setValue(int(d.get("max_display_fps", DEFAULT_MAX_DISPLAY_FPS)))
        self.advanced_panel.gaussianR2SpinBox.setValue(float(d.get("gaussian_r2_min", GAUSSIAN_R2_MIN)))
        self.advanced_panel.averagingSpinBox.setValue(int(d.get("averaging", 1)))
        self.advanced_panel.clipLevelSpinBox.setValue(int(d.get("clip_level", DEFAULT_CLIP_LEVEL_PERCENT)))
        self.advanced_panel.calcAreaCheckbox.setChecked(bool(d.get("calc_area_enabled", False)))
        self.advanced_panel.calcAreaSpinBox.setValue(int(d.get("calc_area_half_size", 200)))
        self.advanced_panel.psfSpinBox.setValue(float(d.get("psf_um", 0.0)))
        self.advanced_panel.wavelengthSpinBox.setValue(float(d.get("wavelength_nm", 780.0)))
        self.advanced_panel.powerCalSpinBox.setValue(float(d.get("power_cal_factor", 0.0)))
        self.advanced_panel.darkEnableCheckbox.setChecked(bool(d.get("dark_frame_enabled", False)))
        self.advanced_panel.minSnrSpinBox.setValue(float(d.get("minimum_snr", 10.0)))
        self.advanced_panel.badPixelMaskCheckbox.setChecked(bool(d.get("bad_pixel_mask_enabled", False)))
        self.advanced_panel.holdMaxCheckbox.setChecked(bool(d.get("hold_maximum_enabled", False)))

        power_unit = d.get("power_unit", "µW")
        idx2 = self.advanced_panel.powerUnitCombo.findText(power_unit)
        if idx2 >= 0:
            self.advanced_panel.powerUnitCombo.setCurrentIndex(idx2)

        self.action_dark_mode.setChecked(bool(d.get("dark_mode", False)))

    def _load_settings(self):
        s = self.settings
        defaults = {
            "camera_backend": "Thorlabs Scientific / Zelux", "exposure_ms": 5.0,
            "grid_label_size": 50, "pixel_pitch": 3.45, "colormap": "Inferno",
            "opacity": DEFAULT_RAW_DATA_OPACITY_PERCENT, "zoom": 100,
            "overlay_height": OVERLAY_CURVE_HEIGHT, "percent_mode": False,
            "max_display_fps": DEFAULT_MAX_DISPLAY_FPS, "gaussian_r2_min": GAUSSIAN_R2_MIN, "averaging": 1,
            "clip_level": DEFAULT_CLIP_LEVEL_PERCENT, "calc_area_enabled": False,
            "calc_area_half_size": 200, "psf_um": 0.0, "wavelength_nm": 780.0,
            "power_cal_factor": 0.0, "power_unit": "µW",
            "dark_frame_enabled": False, "minimum_snr": 10.0,
            "bad_pixel_mask_enabled": False, "hold_maximum_enabled": False,
            "dark_mode": False,
        }
        d = {key: s.value(key, default, type=type(default)) for key, default in defaults.items()}
        self._apply_settings_dict(d)

    def _save_settings(self):
        s = self.settings
        for key, value in self._collect_settings_dict().items():
            s.setValue(key, value)


    def _load_presets_file(self):
        if not os.path.exists(self.presets_path):
            return {}
        try:
            with open(self.presets_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[beam profiler] Could not read presets file: {e}")
            return {}

    def _save_presets_file(self, presets):
        try:
            with open(self.presets_path, "w") as f:
                json.dump(presets, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Preset Save Failed", f"Could not write presets file:\n{e}")

    def _rebuild_presets_menu(self):
        self.presets_menu.clear()
        presets = self._load_presets_file()
        if not presets:
            empty_action = QAction("(no presets saved)", self)
            empty_action.setEnabled(False)
            self.presets_menu.addAction(empty_action)
            return
        for name in sorted(presets.keys()):
            action = QAction(name, self)
            action.triggered.connect(lambda checked=False, n=name: self.load_preset(n))
            self.presets_menu.addAction(action)

    def save_preset_as(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = self._load_presets_file()
        presets[name] = self._collect_settings_dict()
        self._save_presets_file(presets)
        self._rebuild_presets_menu()
        QMessageBox.information(self, "Preset Saved", f"Saved preset '{name}'.")

    def load_preset(self, name):
        presets = self._load_presets_file()
        if name not in presets:
            QMessageBox.warning(self, "Preset Not Found", f"No preset named '{name}'.")
            return
        self._apply_settings_dict(presets[name])
        self.connect_selected_camera()

    def delete_preset(self):
        presets = self._load_presets_file()
        if not presets:
            QMessageBox.information(self, "No Presets", "There are no saved presets to delete.")
            return
        names = sorted(presets.keys())
        name, ok = QInputDialog.getItem(self, "Delete Preset", "Choose a preset to delete:", names, 0, False)
        if not ok or not name:
            return
        confirm = QMessageBox.question(self, "Confirm Delete", f"Delete preset '{name}'? This can't be undone.")
        if confirm == QMessageBox.StandardButton.Yes:
            del presets[name]
            self._save_presets_file(presets)
            self._rebuild_presets_menu()


    def update_raw_data_opacity(self, value):
        alpha_val = value / 100.0
        self.control_panel.opacityLabel.setText(f"Raw Data Opacity: {value}%")
        self.graph_panel_x.raw_data_line.set_alpha(alpha_val)
        self.graph_panel_y.raw_data_line.set_alpha(alpha_val)

    def update_overlay_height(self, value):
        self.overlay_curve_height = value
        self.control_panel.overlayHeightLabel.setText(f"Overlay Curve Height: {value}px")

    def update_overlay_text_scale(self, value):
        self.overlay_text_scale = value / 100.0

    def update_intensity_mode(self, checked):
        self.graph_panel_x.set_intensity_mode(checked, self.sensor_max_raw)
        self.graph_panel_y.set_intensity_mode(checked, self.sensor_max_raw)

    def update_averaging(self, value):
        self._frame_buffer = collections.deque(maxlen=value)
        self._frame_sum = None; self._analysis_buffer = None

    def reset_hold_maximum(self):
        self._hold_max_frame = None

    def reset_advanced_defaults(self):
        self.advanced_panel.gaussianR2SpinBox.setValue(GAUSSIAN_R2_MIN)
        self.advanced_panel.maxFpsSpinBox.setValue(DEFAULT_MAX_DISPLAY_FPS)
        self.advanced_panel.averagingSpinBox.setValue(1)
        self.advanced_panel.calcAreaCheckbox.setChecked(False)
        self.advanced_panel.calcAreaSpinBox.setValue(200)
        self.advanced_panel.clipLevelSpinBox.setValue(DEFAULT_CLIP_LEVEL_PERCENT)
        self.advanced_panel.holdMaxCheckbox.setChecked(False)
        self.reset_hold_maximum()
        self.advanced_panel.psfSpinBox.setValue(0.0)
        self.advanced_panel.wavelengthSpinBox.setValue(780.0)
        self.advanced_panel.powerCalSpinBox.setValue(0.0)
        self.advanced_panel.powerUnitCombo.setCurrentIndex(0)

    def toggle_logging(self, checked):
        if checked:
            filename = f"beam_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self._log_file = open(filename, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow([
                "timestamp", "centroid_x_px", "centroid_y_px",
                "fwhm_x_um", "fwhm_x_px", "fwhm_y_um", "fwhm_y_px",
                "e2_x_um", "e2_x_px", "e2_y_um", "e2_y_px",
                "gaussian_r2_x", "gaussian_r2_y", "exposure_ms", "saturation_percent",
                "d4sigma_x_um", "d4sigma_x_px", "d4sigma_y_um", "d4sigma_y_px",
                "d4sigma_major_um", "d4sigma_major_px", "d4sigma_minor_um", "d4sigma_minor_px",
                "orientation_deg", "ellipticity", "moment_truncated",
            ])
            self._log_row_count = 0
            self._set_text_if_changed(self.analyze_panel.logButton, "Stop Logging")
            self._set_text_if_changed(self.analyze_panel.logStatusLabel, f"Logging: {filename}")
        else:
            if self._log_file is not None:
                self._log_file.close()
            self._log_file = None
            self._log_writer = None
            self._set_text_if_changed(self.analyze_panel.logButton, "Start Logging")
            self._set_text_if_changed(self.analyze_panel.logStatusLabel, f"Logging: Off ({self._log_row_count} rows saved)")

    def snap_to_beam(self):
        zoom = self.control_panel.zoomSlider.value() / 100.0

        target_x = int(self.last_cx * zoom)
        target_y = int(self.last_cy * zoom)

        view_w = self.camera_panel.scrollArea.viewport().width()
        view_h = self.camera_panel.scrollArea.viewport().height()

        self.camera_panel.scrollArea.horizontalScrollBar().setValue(target_x - (view_w // 2))
        self.camera_panel.scrollArea.verticalScrollBar().setValue(target_y - (view_h // 2))

    def _get_colormap_lut(self, cv_cmap, cmap_choice):
        lut = self._colormap_luts.get(cv_cmap)
        if lut is not None:
            return lut
        ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
        bgr_lut = cv2.applyColorMap(ramp, cv_cmap)
        rgb_lut = np.ascontiguousarray(cv2.cvtColor(bgr_lut, cv2.COLOR_BGR2RGB))
        if cmap_choice == "Jet":
            rgb_lut[0, :15, :] = 0
        self._colormap_luts[cv_cmap] = rgb_lut
        return rgb_lut

    def _fit_gaussian(self, numbers, values, info, center_guess):
        if not (info and info.get("fwhm_width")):
            return None, None

        n = len(values)
        c_guess = max(info["fwhm_width"] / 2.3548, 0.5)
        center_guess = min(max(center_guess, 0), n - 1)
        p0 = [info["signal"], center_guess, c_guess, info["background"]]
        lower = [0.0, 0.0, 0.25, 0.0]
        upper = [self.sensor_max_raw * 2.0, float(n), float(n), self.sensor_max_raw]

        FIT_MAX_POINTS = 400
        if n > FIT_MAX_POINTS:
            idx = np.linspace(0, n - 1, FIT_MAX_POINTS).astype(np.int64)
            fit_numbers, fit_values = numbers[idx], values[idx]
        else:
            fit_numbers, fit_values = numbers, values

        try:
            popt, _ = curve_fit(gaussian_model, fit_numbers, fit_values, p0=p0,
                                bounds=(lower, upper), maxfev=2000)
        except (RuntimeError, ValueError):
            return None, None

        fitted = gaussian_model(numbers, *popt)
        return popt, r_squared(values, fitted)

    def _current_acquisition_metadata(self, frame=None):
        shape = tuple(frame.shape) if frame is not None else None
        dtype = str(frame.dtype) if frame is not None else None
        camera_model = self.control_panel.deviceCombo.currentText()
        gain = None
        pixel_format = dtype
        return {
            "camera_backend": self.current_camera_backend_name, "camera_model": camera_model,
            "resolution": shape, "dtype": dtype, "sensor_bit_depth": self.sensor_bit_depth,
            "exposure_ms": float(self.current_exposure_ms), "gain": gain, "pixel_format": pixel_format,
        }

    def capture_dark_frame(self):
        if not self._camera_connected:
            QMessageBox.warning(self, "Dark Frame", "Connect a camera first.")
            return
        self.advanced_panel.darkStatusLabel.setText("Dark frame: capturing 8 frames...")
        self._camera_bridge.dark_requested.emit(8)


    def _acquire_frame_stage(self):
        if not self._camera_connected:
            self._show_no_signal(); return None
        packet = self._frame_mailbox.take_latest()
        if packet is None:
            self._consecutive_bad_frames += 1
            stats = self._frame_mailbox.snapshot()
            if stats["latest_frame_age_ms"] is None or stats["latest_frame_age_ms"] >= 500.0:
                if self._consecutive_bad_frames >= STALE_FRAME_LIMIT:
                    self._show_no_signal()
            return None
        self._consecutive_bad_frames = 0
        return RawFrame(packet.data, packet.camera, time.time(), packet.frame_index)

    def _process_frame_stage(self, raw_frame):
        gray_raw = raw_frame.data
        self._latest_raw_frame = gray_raw
        current_dark_meta = {
            "camera_backend": raw_frame.camera.camera_backend,
            "camera_model": raw_frame.camera.camera_model,
            "resolution": tuple(gray_raw.shape), "dtype": str(gray_raw.dtype),
            "sensor_bit_depth": raw_frame.camera.sensor_bit_depth,
            "exposure_ms": raw_frame.camera.exposure_ms, "gain": raw_frame.camera.gain,
            "pixel_format": raw_frame.camera.pixel_format,
        }
        dark_valid = False; dark_reason = ""; dark_applied = False
        if self._dark_frame is not None and self._dark_metadata is not None:
            dark_valid, dark_reason = dark_metadata_match(self._dark_metadata, current_dark_meta)
        if self.advanced_panel.darkEnableCheckbox.isChecked() and dark_valid:
            corrected = subtract_dark_frame(gray_raw, self._dark_frame); dark_applied = True
            self.advanced_panel.darkStatusLabel.setText("Dark frame: valid and applied")
        else:
            corrected = gray_raw.astype(np.float32)
            if self.advanced_panel.darkEnableCheckbox.isChecked() and self._dark_frame is not None and not dark_valid:
                self.advanced_panel.darkStatusLabel.setText("Dark frame disabled: " + dark_reason)
            elif self._dark_frame is None:
                self.advanced_panel.darkStatusLabel.setText("Dark frame: not captured")
        if self._frame_buffer.maxlen and self._frame_buffer.maxlen > 1:
            if self._frame_sum is None or self._frame_sum.shape != corrected.shape:
                self._frame_sum = np.zeros_like(corrected); self._frame_buffer.clear()
                self._analysis_buffer = None
            if len(self._frame_buffer) == self._frame_buffer.maxlen:
                self._frame_sum -= self._frame_buffer.popleft()
            self._frame_buffer.append(corrected); self._frame_sum += corrected
            if self._analysis_buffer is None or self._analysis_buffer.shape != self._frame_sum.shape:
                self._analysis_buffer = np.empty_like(self._frame_sum)
            np.divide(self._frame_sum, len(self._frame_buffer), out=self._analysis_buffer)
            analysis = self._analysis_buffer
        else:
            analysis = corrected
        if self.advanced_panel.holdMaxCheckbox.isChecked():
            if self._hold_max_frame is None or self._hold_max_frame.shape != analysis.shape:
                self._hold_max_frame = analysis.copy()
            else:
                np.maximum(self._hold_max_frame, analysis, out=self._hold_max_frame)
            analysis = self._hold_max_frame
        else:
            self._hold_max_frame = None
        display = np.clip(analysis / self.sensor_display_divisor, 0, 255).astype(np.uint8)
        h_img, w_img = analysis.shape
        if self.advanced_panel.calcAreaCheckbox.isChecked():
            half=self.advanced_panel.calcAreaSpinBox.value(); cy0,cx0=self.last_cy,self.last_cx
            y0=max(0,cy0-half); y1=min(h_img,cy0+half); x0=max(0,cx0-half); x1=min(w_img,cx0+half)
            roi=analysis[y0:y1,x0:x1]; bounds=(x0,y0,x1,y1); ox,oy=x0,y0
        else:
            roi=analysis; bounds=None; ox=oy=0
        return ProcessedFrame(raw_frame, corrected, analysis, display, roi, bounds, ox, oy,
                              dark_applied, dark_valid, dark_reason)

    @staticmethod
    def _set_text_if_changed(label, text):
        if label.text() != text:
            label.setText(text)

    def _measure_frame_stage(self, processed, force_advanced=False):
        gray_raw=processed.raw.data; measure_img=processed.analysis; calc_img=processed.roi
        offset_x,offset_y=processed.offset_x,processed.offset_y

        advanced_enabled = self.advanced_panel.enableAdvancedMetricsCheckbox.isChecked()

        now = time.monotonic()
        hot_mask_shape_ok = (not advanced_enabled) or (
            self._cached_hot_mask is not None and self._cached_hot_mask.shape == calc_img.shape)
        refresh_advanced = (
            force_advanced
            or self._last_advanced_metrics_ts is None
            or not hot_mask_shape_ok
            or (now - self._last_advanced_metrics_ts) >= ADVANCED_METRICS_INTERVAL_S
        )

        if refresh_advanced:
            if advanced_enabled:
                background_level, background_noise = robust_background_stats(calc_img)
                hot_mask = detect_hot_pixels(calc_img, background_level, background_noise)
                hot_count = int(np.count_nonzero(hot_mask))
                hot_coords = np.nonzero(hot_mask) if hot_count else (
                    np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
            else:
                background_level, background_noise = 0.0, 0.0
                hot_mask, hot_count = None, 0
                hot_coords = (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
            sensor_edge, roi_edge = edge_truncation_flags(measure_img, processed.roi_bounds)
            self._cached_background_level = background_level
            self._cached_background_noise = background_noise
            self._cached_hot_mask = hot_mask
            self._cached_hot_count = hot_count
            self._cached_hot_coords = hot_coords
            self._cached_sensor_edge = sensor_edge
            self._cached_roi_edge = roi_edge
            self._last_advanced_metrics_ts = now
            self._advanced_metrics_age_ms = 0.0
        else:
            background_level = self._cached_background_level
            background_noise = self._cached_background_noise
            hot_mask = self._cached_hot_mask
            hot_count = self._cached_hot_count
            hot_coords = self._cached_hot_coords
            sensor_edge = self._cached_sensor_edge
            roi_edge = self._cached_roi_edge
            self._advanced_metrics_age_ms = (now - self._last_advanced_metrics_ts) * 1000.0

        corrected_peak=max(0.0,float(calc_img.max())-background_level)
        snr_peak=corrected_peak/background_noise if background_noise>0 else (float("inf") if corrected_peak>0 else 0.0)
        if self.advanced_panel.badPixelMaskCheckbox.isChecked() and hot_count:
            calc_img_f32 = calc_img if calc_img.dtype == np.float32 else calc_img.astype(np.float32)
            med = cv2.medianBlur(calc_img_f32, 3)
            calc_img = calc_img_f32.copy()
            hy, hx = hot_coords
            calc_img[hy, hx] = med[hy, hx]

        raw_peak = float(gray_raw.max())
        central=False; isolated=False; sat_count=0; sat_fraction=0.0
        if raw_peak >= self.sensor_max_raw:
            raw_sat=gray_raw>=self.sensor_max_raw; sat_count=int(np.count_nonzero(raw_sat)); sat_fraction=sat_count/float(gray_raw.size)
            if sat_count:
                sy,sx=np.nonzero(raw_sat)
                spread=max(float(np.ptp(sx)) if len(sx) else 0.0,float(np.ptp(sy)) if len(sy) else 0.0)
                isolated=sat_count<=4 and spread<=2.0; central=not isolated
        low=snr_peak<self.advanced_panel.minSnrSpinBox.value()
        _,roi_max,_,maxloc=cv2.minMaxLoc(calc_img)
        if roi_max<=0: return None
        peak_x=maxloc[0]+offset_x; peak_y=maxloc[1]+offset_y
        clip=self.advanced_panel.clipLevelSpinBox.value()/100.0

        if refresh_advanced:
            percentile_background = float(np.percentile(calc_img, BACKGROUND_PERCENTILE))
            self._cached_percentile_background = percentile_background
        else:
            percentile_background = self._cached_percentile_background

        cent=intensity_weighted_centroid(calc_img,percentile_background,clip)
        if cent is None: cx=peak_x; cy=peak_y; cxf=float(cx); cyf=float(cy)
        else: cxf=cent[0]+offset_x; cyf=cent[1]+offset_y; cx=int(round(cxf)); cy=int(round(cyf))
        h,w=measure_img.shape; y0=max(0,cy-PROFILE_BAND_HALFWIDTH); y1=min(h,cy+PROFILE_BAND_HALFWIDTH+1)
        x0=max(0,cx-PROFILE_BAND_HALFWIDTH); x1=min(w,cx+PROFILE_BAND_HALFWIDTH+1)
        xv=np.mean(measure_img[y0:y1,:],axis=0); yv=np.mean(measure_img[:,x0:x1],axis=1)
        nx=np.arange(len(xv)); ny=np.arange(len(yv))
        hi=analyze_slice(xv,self.min_signal_above_background); vi=analyze_slice(yv,self.min_signal_above_background)
        warnings=MeasurementWarnings(low,central,isolated,sensor_edge,roi_edge)
        if warnings.invalidates_widths: hi=vi=None

        if warnings.invalidates_widths or not advanced_enabled:
            moment = None
        elif refresh_advanced:
            moment = second_moment_2d(calc_img, background=percentile_background)
            self._cached_moment = moment
        else:
            moment = self._cached_moment

        quality=FrameQuality(background_level,background_noise,snr_peak,raw_peak,corrected_peak,
            float(measure_img.max()),float(processed.display_u8.max()),sat_count,sat_fraction,hot_count,
            sensor_edge,roi_edge,low,central,isolated,processed.dark_applied,processed.dark_valid,
            processed.dark_mismatch_reason)
        return BeamMeasurement(peak_x,peak_y,cxf,cyf,xv,yv,nx,ny,hi,vi,calc_img,moment,quality,warnings,
            100.0*raw_peak/self.sensor_max_raw,100.0*roi_max/self.sensor_max_raw,
            float(np.clip(calc_img-background_level,0,None).sum()))

    def update_frame(self):
        t_frame_start=time.perf_counter()
        now_display = t_frame_start
        if self._last_display_ts is not None and now_display > self._last_display_ts:
            inst = 1.0 / (now_display - self._last_display_ts)
            self._display_fps_smoothed = inst if self._display_fps_smoothed is None else 0.9*self._display_fps_smoothed + 0.1*inst
        self._last_display_ts = now_display
        if self._last_frame_ts is not None:
            dt=t_frame_start-self._last_frame_ts
            if dt>0:
                instant_fps=1.0/dt
                self._fps_smoothed=instant_fps if self._fps_smoothed is None else (0.9*self._fps_smoothed+0.1*instant_fps)
        self._last_frame_ts=t_frame_start; self.frame_counter+=1
        sensor_pixel_size=self.analyze_panel.pixel_pitch_input.value()
        raw_frame=self._acquire_frame_stage()
        if raw_frame is None:
            self._update_performance_instrumentation()
            return
        gray_raw=raw_frame.data
        processed=self._process_frame_stage(raw_frame)
        self._latest_processed_frame = processed
        measure_img=processed.analysis; display_img=processed.display_u8; calc_img=processed.roi
        offset_x,offset_y=processed.offset_x,processed.offset_y
        h_img,w_img=measure_img.shape
        measurement=self._measure_frame_stage(processed)
        analysis_ms = (time.perf_counter() - t_frame_start) * 1000.0
        self._analysis_ms_smoothed = analysis_ms if self._analysis_ms_smoothed is None else 0.9*self._analysis_ms_smoothed + 0.1*analysis_ms
        if measurement is None:
            self._show_no_signal(); return
        if not self._warned_possible_clipping and measurement.quality.raw_peak >= self.sensor_max_raw:
            print(f"[beam profiler] Raw pixel value reached the sensor's reported maximum ({self.sensor_max_raw}, {self.sensor_bit_depth}-bit) -- the sensor may be saturating.")
            self._warned_possible_clipping=True
        self._latest_measurement=measurement
        peak_x,peak_y=measurement.peak_x,measurement.peak_y
        center_x_precise,center_y_precise=measurement.centroid_x,measurement.centroid_y
        center_x,center_y=int(round(center_x_precise)),int(round(center_y_precise))
        saturation_pct=measurement.raw_saturation_percent; processed_peak_pct=measurement.processed_peak_percent
        scale=1.0; disp_cx,disp_cy=center_x,center_y; disp_px,disp_py=peak_x,peak_y
        self.last_cx,self.last_cy=disp_cx,disp_cy
        x_values,y_values=measurement.x_values,measurement.y_values
        numbers_x,numbers_y=measurement.numbers_x,measurement.numbers_y
        self._numbers_x_cache=numbers_x; self._numbers_y_cache=numbers_y
        h_info,v_info=measurement.x_profile,measurement.y_profile
        self._last_h_info,self._last_v_info=h_info,v_info
        quality_invalid=measurement.warnings.invalidates_widths
        background_level=measurement.quality.background_level; background_noise=measurement.quality.background_noise_mad
        corrected_peak=measurement.quality.corrected_peak; snr_peak=measurement.quality.snr_peak
        hot_pixel_count=measurement.quality.hot_pixel_count; saturated_pixel_count=measurement.quality.saturated_pixel_count
        saturated_fraction=measurement.quality.saturated_pixel_fraction
        sensor_edge_truncated=measurement.warnings.sensor_edge_truncated; roi_edge_truncated=measurement.warnings.roi_edge_truncated
        low_snr=measurement.warnings.low_snr; central_saturation=measurement.warnings.central_saturation
        isolated_hot_saturation=measurement.warnings.isolated_hot_pixel_saturation
        dark_applied=processed.dark_applied; dark_valid=processed.dark_valid; dark_reason=processed.dark_mismatch_reason
        calc_img=measurement.analysis_roi
        moment=measurement.second_moment
        pitch = self.analyze_panel.pixel_pitch_input.value()
        psf_um = self.advanced_panel.psfSpinBox.value()

        self._last_second_moment = moment
        d4sigma_x_um = d4sigma_y_um = None
        d4sigma_major_um = d4sigma_minor_um = orientation_deg = None
        moment_ellipticity = None
        moment_truncated = False
        if moment is not None:
            d4sigma_x_um = moment["d4sigma_x_px"] * pitch
            d4sigma_y_um = moment["d4sigma_y_px"] * pitch
            d4sigma_major_um = moment["d4sigma_major_px"] * pitch
            d4sigma_minor_um = moment["d4sigma_minor_px"] * pitch
            orientation_deg = moment["orientation_deg"]
            moment_ellipticity = moment["ellipticity"]
            moment_truncated = moment["truncated"]
            edge_warn = "  \u26A0 beam extends past analysis window (D4σ unreliable)" if moment_truncated else ""
            self._set_text_if_changed(self.analyze_panel.d4sigma_x_label, 
                f"X ISO D4σ: {d4sigma_x_um:.1f} µm  ({moment['d4sigma_x_px']:.1f} px){edge_warn}")
            self._set_text_if_changed(self.analyze_panel.d4sigma_y_label, 
                f"Y ISO D4σ: {d4sigma_y_um:.1f} µm  ({moment['d4sigma_y_px']:.1f} px){edge_warn}")
            self._set_text_if_changed(self.analyze_panel.d4sigma_principal_label, 
                f"Principal D4σ: {d4sigma_major_um:.1f} / {d4sigma_minor_um:.1f} µm; angle {orientation_deg:.1f}°{edge_warn}")
            if moment_ellipticity is not None:
                self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, 
                    f"Ellipticity (minor/major D4σ): {moment_ellipticity:.3f}{edge_warn}")
            else:
                self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, "Ellipticity (minor/major D4σ): N/A")
        else:
            self._set_text_if_changed(self.analyze_panel.d4sigma_x_label, "X ISO D4σ: N/A")
            self._set_text_if_changed(self.analyze_panel.d4sigma_y_label, "Y ISO D4σ: N/A")
            self._set_text_if_changed(self.analyze_panel.d4sigma_principal_label, "Principal D4σ (major/minor, angle): N/A")
            self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, "Ellipticity (minor/major D4σ): N/A")

        if h_info is not None:
            fwhm_x = h_info.get("fwhm_width")
            if fwhm_x is not None:
                fwhm_x_um = correct_for_instrument_psf(fwhm_x * pitch, psf_um)
                edge_warn = "  \u26A0 may be truncated" if h_info.get("fwhm_touches_edge") else ""
                self._set_text_if_changed(self.analyze_panel.fwhm_x_label, f"X FWHM: {fwhm_x_um:.1f} µm  ({fwhm_x:.1f} px){edge_warn}")
            if "e2_width" in h_info:
                e2_x = h_info["e2_width"]
                e2_x_um = correct_for_instrument_psf(e2_x * pitch, psf_um)
                edge_warn = "  \u26A0 may be truncated" if h_info.get("e2_touches_edge") else ""
                self._set_text_if_changed(self.analyze_panel.e2_x_label, f"X 1/e² Threshold: {e2_x_um:.1f} µm  ({e2_x:.1f} px){edge_warn}")
        else:
            self._set_text_if_changed(self.analyze_panel.fwhm_x_label, "X FWHM: N/A")
            self._set_text_if_changed(self.analyze_panel.e2_x_label, "X 1/e² Threshold: N/A")

        if v_info is not None:
            fwhm_y = v_info.get("fwhm_width")
            if fwhm_y is not None:
                fwhm_y_um = correct_for_instrument_psf(fwhm_y * pitch, psf_um)
                edge_warn = "  \u26A0 may be truncated" if v_info.get("fwhm_touches_edge") else ""
                self._set_text_if_changed(self.analyze_panel.fwhm_y_label, f"Y FWHM: {fwhm_y_um:.1f} µm  ({fwhm_y:.1f} px){edge_warn}")
            if "e2_width" in v_info:
                e2_y = v_info["e2_width"]
                e2_y_um = correct_for_instrument_psf(e2_y * pitch, psf_um)
                edge_warn = "  \u26A0 may be truncated" if v_info.get("e2_touches_edge") else ""
                self._set_text_if_changed(self.analyze_panel.e2_y_label, f"Y 1/e² Threshold: {e2_y_um:.1f} µm  ({e2_y:.1f} px){edge_warn}")
        else:
            self._set_text_if_changed(self.analyze_panel.fwhm_y_label, "Y FWHM: N/A")
            self._set_text_if_changed(self.analyze_panel.e2_y_label, "Y 1/e² Threshold: N/A")

        if h_info is None:
            self._last_gaussian_popt_x = None
            self._last_gaussian_r2_x = None
        if v_info is None:
            self._last_gaussian_popt_y = None
            self._last_gaussian_r2_y = None

        need_gaussian = (self.render_x_plot or self.render_y_plot
                         or (self.render_video and DRAW_LIVE_OVERLAY)
                         or self.sub_waist.isVisible()
                         or self._log_writer is not None)
        if need_gaussian and self.frame_counter % GAUSSIAN_FIT_INTERVAL == 0:
            self._last_gaussian_popt_x, self._last_gaussian_r2_x = self._fit_gaussian(
                numbers_x, x_values, h_info, center_x)
            self._last_gaussian_popt_y, self._last_gaussian_r2_y = self._fit_gaussian(
                numbers_y, y_values, v_info, center_y)

        r2_threshold = self.advanced_panel.gaussianR2SpinBox.value()
        good_fit_x = self._last_gaussian_popt_x is not None and (self._last_gaussian_r2_x or 0) >= r2_threshold
        good_fit_y = self._last_gaussian_popt_y is not None and (self._last_gaussian_r2_y or 0) >= r2_threshold

        if self._last_gaussian_r2_x is None:
            self._set_text_if_changed(self.analyze_panel.fit_quality_x_label, "X Fit Shape: N/A")
        else:
            shape_x = "Gaussian" if good_fit_x else "Below fit threshold"
            self._set_text_if_changed(self.analyze_panel.fit_quality_x_label, f"X Fit Shape: {shape_x}  (R²={self._last_gaussian_r2_x:.3f})")

        if self._last_gaussian_r2_y is None:
            self._set_text_if_changed(self.analyze_panel.fit_quality_y_label, "Y Fit Shape: N/A")
        else:
            shape_y = "Gaussian" if good_fit_y else "Below fit threshold"
            self._set_text_if_changed(self.analyze_panel.fit_quality_y_label, f"Y Fit Shape: {shape_y}  (R²={self._last_gaussian_r2_y:.3f})")

        wavelength_nm = self.advanced_panel.wavelengthSpinBox.value()
        treat_as_waist = self.analyze_panel.treatAsWaistCheckbox.isChecked()
        diam_x_um = None
        diam_y_um = None

        if good_fit_x:
            diam_x_px = gaussian_1e2_diameter(self._last_gaussian_popt_x)
            diam_x_um = correct_for_instrument_psf(diam_x_px * pitch, psf_um)
            self._set_text_if_changed(self.analyze_panel.gauss_diam_x_label, 
                f"X Gaussian Diameter: {diam_x_um:.1f} µm  ({diam_x_px:.1f} px)")
            if treat_as_waist:
                div_x_mrad = divergence_half_angle_mrad(diam_x_um / 2.0, wavelength_nm)
                self._set_text_if_changed(self.analyze_panel.divergence_x_label, 
                    f"X Min. Divergence (IF waist, M²=1): {div_x_mrad:.3f} mrad" if div_x_mrad is not None
                    else "X Divergence: N/A")
            else:
                self._set_text_if_changed(self.analyze_panel.divergence_x_label, "X Divergence: enable checkbox")
        else:
            self._set_text_if_changed(self.analyze_panel.gauss_diam_x_label, "X Gaussian Diameter: N/A")
            self._set_text_if_changed(self.analyze_panel.divergence_x_label, "X Divergence: enable checkbox")

        if good_fit_y:
            diam_y_px = gaussian_1e2_diameter(self._last_gaussian_popt_y)
            diam_y_um = correct_for_instrument_psf(diam_y_px * pitch, psf_um)
            self._set_text_if_changed(self.analyze_panel.gauss_diam_y_label, 
                f"Y Gaussian Diameter: {diam_y_um:.1f} µm  ({diam_y_px:.1f} px)")
            if treat_as_waist:
                div_y_mrad = divergence_half_angle_mrad(diam_y_um / 2.0, wavelength_nm)
                self._set_text_if_changed(self.analyze_panel.divergence_y_label, 
                    f"Y Min. Divergence (IF waist, M²=1): {div_y_mrad:.3f} mrad" if div_y_mrad is not None
                    else "Y Divergence: N/A")
            else:
                self._set_text_if_changed(self.analyze_panel.divergence_y_label, "Y Divergence: enable checkbox")
        else:
            self._set_text_if_changed(self.analyze_panel.gauss_diam_y_label, "Y Gaussian Diameter: N/A")
            self._set_text_if_changed(self.analyze_panel.divergence_y_label, "Y Divergence: enable checkbox")

        self._set_text_if_changed(self.analyze_panel.peak_position_label, 
            f"Peak Position: ({peak_x * pitch:.1f}, {peak_y * pitch:.1f}) µm")
        self._set_text_if_changed(self.analyze_panel.centroid_position_label, 
            f"Centroid Position: ({center_x_precise * pitch:.1f}, {center_y_precise * pitch:.1f}) µm")

        if abs(saturation_pct - processed_peak_pct) >= 1.0:
            self._set_text_if_changed(self.analyze_panel.saturation_label, 
                f"A/D Saturation: {saturation_pct:.1f}%")
        else:
            self._set_text_if_changed(self.analyze_panel.saturation_label, f"A/D Saturation: {saturation_pct:.1f}%")
        self.analyze_panel.saturation_label.setStyleSheet("color: red;" if saturation_pct >= 95 else "")

        displayed_peak = measurement.quality.displayed_peak
        averaged_peak = measurement.quality.averaged_peak
        raw_peak = measurement.quality.raw_peak
        if self.advanced_panel.enableAdvancedMetricsCheckbox.isChecked():
            self._set_text_if_changed(self.analyze_panel.background_label, 
                f"Background: median={background_level:.2f}, MAD σ={background_noise:.2f}, "
                f"corrected counts={measurement.background_subtracted_counts:.1f}")
        else:
            self._set_text_if_changed(self.analyze_panel.background_label, 
                "Background: disabled (Advanced Settings > Enable D4σ/background/hot-pixel analysis)")
        flags = []
        if low_snr: flags.append("LOW SNR")
        if central_saturation: flags.append("CENTRAL SATURATION")
        if isolated_hot_saturation: flags.append("ISOLATED HOT-PIXEL SATURATION")
        if sensor_edge_truncated: flags.append("SENSOR-EDGE TRUNCATION")
        if roi_edge_truncated: flags.append("ROI-EDGE TRUNCATION")
        if self.advanced_panel.enableAdvancedMetricsCheckbox.isChecked():
            self._set_text_if_changed(self.analyze_panel.quality_label, 
                f"Signal Quality: SNR={snr_peak:.1f}; raw/corrected/avg/display peaks="
                f"{raw_peak:.1f}/{corrected_peak:.1f}/{averaged_peak:.1f}/{displayed_peak:.1f}"
                + ("; " + ", ".join(flags) if flags else "; OK"))
        else:
            self._set_text_if_changed(self.analyze_panel.quality_label, 
                f"Signal Quality: SNR N/A (background analysis disabled); raw/avg/display peaks="
                f"{raw_peak:.1f}/{averaged_peak:.1f}/{displayed_peak:.1f}"
                + ("; " + ", ".join(flags) if flags else "; OK"))
        self._latest_quality = measurement.quality

        clipped = bool((h_info and (h_info.get("fwhm_touches_edge") or h_info.get("e2_touches_edge"))) or
                       (v_info and (v_info.get("fwhm_touches_edge") or v_info.get("e2_touches_edge"))) or
                       moment_truncated)
        current_sample = PropagationSample(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            d4sigma_x_um=d4sigma_x_um, d4sigma_y_um=d4sigma_y_um,
            d4sigma_major_um=d4sigma_major_um, d4sigma_minor_um=d4sigma_minor_um,
            orientation_deg=orientation_deg, ellipticity=moment_ellipticity,
            d4sigma_truncated=moment_truncated,
            gaussian_x_um=diam_x_um, gaussian_y_um=diam_y_um,
            r2_x=self._last_gaussian_r2_x, r2_y=self._last_gaussian_r2_y,
            saturation_fraction=saturated_fraction, clipped=clipped,
            valid=(moment is not None and not clipped and not quality_invalid and h_info is not None and v_info is not None),
            exposure_ms=self.current_exposure_ms, pixel_pitch_um=pitch, wavelength_nm=wavelength_nm,
        )
        self._latest_propagation_sample = current_sample
        self._propagation_history.append((time.time(), current_sample))

        power_factor = self.advanced_panel.powerCalSpinBox.value()
        if power_factor > 0:
            total_counts = measurement.background_subtracted_counts
            power_uw = total_counts * power_factor
            unit = self.advanced_panel.powerUnitCombo.currentText()
            if unit == "µW":
                self._set_text_if_changed(self.analyze_panel.power_label, f"Total Power: {power_uw:.2f} µW")
            elif unit == "mW":
                self._set_text_if_changed(self.analyze_panel.power_label, f"Total Power: {power_uw / 1000.0:.4f} mW")
            else:
                dbm = power_to_dbm(power_uw / 1000.0)
                self._set_text_if_changed(self.analyze_panel.power_label, 
                    f"Total Power: {dbm:.1f} dBm" if dbm is not None else "Total Power: N/A")
        else:
            self._set_text_if_changed(self.analyze_panel.power_label, "Total Power: Not Calibrated")

        if self._log_writer is not None and self.frame_counter % LOG_EVERY_N_FRAMES == 0:
            fwhm_x_px = h_info.get("fwhm_width") if h_info else None
            fwhm_y_px = v_info.get("fwhm_width") if v_info else None
            e2_x_px = h_info.get("e2_width") if h_info else None
            e2_y_px = v_info.get("e2_width") if v_info else None
            self._log_writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                self.last_cx, self.last_cy,
                (fwhm_x_px * pitch) if fwhm_x_px is not None else "", fwhm_x_px if fwhm_x_px is not None else "",
                (fwhm_y_px * pitch) if fwhm_y_px is not None else "", fwhm_y_px if fwhm_y_px is not None else "",
                (e2_x_px * pitch) if e2_x_px is not None else "", e2_x_px if e2_x_px is not None else "",
                (e2_y_px * pitch) if e2_y_px is not None else "", e2_y_px if e2_y_px is not None else "",
                self._last_gaussian_r2_x if self._last_gaussian_r2_x is not None else "",
                self._last_gaussian_r2_y if self._last_gaussian_r2_y is not None else "",
                self.current_exposure_ms,
                f"{saturation_pct:.2f}",
                d4sigma_x_um if d4sigma_x_um is not None else "",
                moment["d4sigma_x_px"] if moment is not None else "",
                d4sigma_y_um if d4sigma_y_um is not None else "",
                moment["d4sigma_y_px"] if moment is not None else "",
                d4sigma_major_um if d4sigma_major_um is not None else "",
                moment["d4sigma_major_px"] if moment is not None else "",
                d4sigma_minor_um if d4sigma_minor_um is not None else "",
                moment["d4sigma_minor_px"] if moment is not None else "",
                orientation_deg if orientation_deg is not None else "",
                moment_ellipticity if moment_ellipticity is not None else "",
                moment_truncated,
            ])
            self._log_row_count += 1
            if self._log_row_count % 10 == 0:
                self._log_file.flush()

        if self.render_stability and self.frame_counter % LOG_EVERY_N_FRAMES == 0:
            dx_um = center_x_precise * pitch - (w_img / 2.0) * pitch
            dy_um = center_y_precise * pitch - (h_img / 2.0) * pitch
            self.stability_panel.append_sample(dx_um, dy_um)

        has_beam = h_info is not None and v_info is not None

        rgb_image = None
        if self.render_video:
            zoom_pct = self.control_panel.zoomSlider.value()
            zoom_factor = zoom_pct / 100.0

            cmap_choice = self.control_panel.cmapCombo.currentText()
            cv_cmap = {
                "Inferno": cv2.COLORMAP_INFERNO,
                "Jet": cv2.COLORMAP_JET,
                "Hot": cv2.COLORMAP_HOT,
                "Magma": cv2.COLORMAP_MAGMA,
            }.get(cmap_choice, cv2.COLORMAP_INFERNO)

            color_lut = self._get_colormap_lut(cv_cmap, cmap_choice)
            rgb_image = cv2.LUT(cv2.merge([display_img, display_img, display_img]), color_lut)

            if has_beam:
                show_centroid = self.camera_panel.centroidCrosshairCheckbox.isChecked()
                show_peak = self.camera_panel.peakCrosshairCheckbox.isChecked()
                if DRAW_LIVE_OVERLAY:
                    centers = {"mass": (disp_cx, disp_cy), "peak": (disp_px, disp_py)}
                    self._draw_live_overlay(
                        rgb_image, scale, centers,
                        numbers_x, numbers_y,
                        self._last_gaussian_popt_x if good_fit_x else None,
                        self._last_gaussian_popt_y if good_fit_y else None,
                        sensor_pixel_size, zoom_factor,
                        show_centroid, show_peak
                    )
                else:
                    if show_centroid:
                        cv2.drawMarker(rgb_image, (disp_cx, disp_cy), (0, 255, 0), cv2.MARKER_CROSS, 34, 3)
                    if show_peak:
                        cv2.drawMarker(rgb_image, (disp_px, disp_py), (255, 0, 0), cv2.MARKER_CROSS, 28, 1)

            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            new_w = int(w * zoom_factor)
            new_h = int(h * zoom_factor)

            if new_w == w and new_h == h:
                scaled_pixmap = pixmap
            else:
                scaled_pixmap = pixmap.scaled(
                    new_w, new_h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )

            if (new_w, new_h) != self._last_video_label_size:
                self.camera_panel.videoLabel.setFixedSize(new_w, new_h)
                self._last_video_label_size = (new_w, new_h)
            self.camera_panel.videoLabel.setPixmap(scaled_pixmap)

        if self.render_x_plot:
            gp = self.graph_panel_x
            gp.set_xlim_once(len(x_values))
            gp.raw_data_line.set_data(numbers_x, gp.scale_values(x_values))

            if h_info is not None:
                gp.fit_curve_line.set_data([h_info["fwhm_left"], h_info["fwhm_right"]],
                                            gp.scale_values([h_info["half_max_level"]] * 2))
                if "e2_width" in h_info:
                    gp.e2_curve_line.set_data([h_info["e2_left"], h_info["e2_right"]],
                                               gp.scale_values([h_info["e2_level"]] * 2))
            else:
                gp.fit_curve_line.set_data([], [])
                gp.e2_curve_line.set_data([], [])

            if good_fit_x:
                gp.gaussian_curve_line.set_data(
                    numbers_x, gp.scale_values(gaussian_model(numbers_x, *self._last_gaussian_popt_x)))
            else:
                gp.gaussian_curve_line.set_data([], [])

            gp.blit_draw()

        if self.render_y_plot:
            gp = self.graph_panel_y
            gp.set_xlim_once(len(y_values))
            gp.raw_data_line.set_data(numbers_y, gp.scale_values(y_values))

            if v_info is not None:
                gp.fit_curve_line.set_data([v_info["fwhm_left"], v_info["fwhm_right"]],
                                            gp.scale_values([v_info["half_max_level"]] * 2))
                if "e2_width" in v_info:
                    gp.e2_curve_line.set_data([v_info["e2_left"], v_info["e2_right"]],
                                               gp.scale_values([v_info["e2_level"]] * 2))
            else:
                gp.fit_curve_line.set_data([], [])
                gp.e2_curve_line.set_data([], [])

            if good_fit_y:
                gp.gaussian_curve_line.set_data(
                    numbers_y, gp.scale_values(gaussian_model(numbers_y, *self._last_gaussian_popt_y)))
            else:
                gp.gaussian_curve_line.set_data([], [])

            gp.blit_draw()

        frame_ms = (time.perf_counter() - t_frame_start) * 1000.0
        self._frame_time_smoothed = frame_ms if self._frame_time_smoothed is None else (0.9 * self._frame_time_smoothed + 0.1 * frame_ms)
        if self._fps_smoothed is not None:
            self.camera_panel.fpsLabel.setText(f"Measured: {self._fps_smoothed:.1f} FPS | {self._frame_time_smoothed:.1f} ms/frame")
        self._update_performance_instrumentation()

    def _update_performance_instrumentation(self):
        stats = self._frame_mailbox.snapshot()
        acq = stats["acquisition_fps"]
        disp = self._display_fps_smoothed or 0.0
        analysis = self._analysis_ms_smoothed
        age = stats["latest_frame_age_ms"]
        self.status_performance_label.setText(
            f"Acq {acq:.1f} FPS | Display {disp:.1f} FPS | Analysis "
            f"{analysis:.1f} ms | Drop {stats['dropped']} | Age "
            f"{age:.0f} ms | Errors {stats['acquisition_errors']}" if analysis is not None and age is not None else
            f"Acq {acq:.1f} FPS | Display {disp:.1f} FPS | Analysis -- | Drop {stats['dropped']} | Age -- | Errors {stats['acquisition_errors']}"
        )

    def _show_no_signal(self):
        label = self.camera_panel.videoLabel
        label.setPixmap(QPixmap())
        viewport_size = self.camera_panel.scrollArea.viewport().size()
        label.setFixedSize(max(320, viewport_size.width()),
                           max(240, viewport_size.height()))
        label.setWordWrap(True)
        label.setText("No live frame from camera")

        self._set_text_if_changed(self.analyze_panel.fwhm_x_label, "X FWHM: N/A")
        self._set_text_if_changed(self.analyze_panel.fwhm_y_label, "Y FWHM: N/A")
        self._set_text_if_changed(self.analyze_panel.e2_x_label, "X 1/e² Threshold: N/A")
        self._set_text_if_changed(self.analyze_panel.e2_y_label, "Y 1/e² Threshold: N/A")
        self._set_text_if_changed(self.analyze_panel.gauss_diam_x_label, "X Gaussian Diameter: N/A")
        self._set_text_if_changed(self.analyze_panel.gauss_diam_y_label, "Y Gaussian Diameter: N/A")
        self._set_text_if_changed(self.analyze_panel.d4sigma_x_label, "X ISO D4σ: N/A")
        self._set_text_if_changed(self.analyze_panel.d4sigma_y_label, "Y ISO D4σ: N/A")
        self._set_text_if_changed(self.analyze_panel.d4sigma_principal_label, "Principal D4σ (major/minor, angle): N/A")
        self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, "Ellipticity (minor/major D4σ): N/A")
        self._set_text_if_changed(self.analyze_panel.divergence_x_label, "X Divergence: enable checkbox")
        self._set_text_if_changed(self.analyze_panel.divergence_y_label, "Y Divergence: enable checkbox")
        self._set_text_if_changed(self.analyze_panel.fit_quality_x_label, "X Fit Shape: N/A")
        self._set_text_if_changed(self.analyze_panel.fit_quality_y_label, "Y Fit Shape: N/A")

        self._set_text_if_changed(self.analyze_panel.peak_position_label, "Peak Position: N/A")
        self._set_text_if_changed(self.analyze_panel.centroid_position_label, "Centroid Position: N/A")
        self._set_text_if_changed(self.analyze_panel.saturation_label, "A/D Saturation: N/A")
        self.analyze_panel.saturation_label.setStyleSheet("")

        for panel in (self.graph_panel_x, self.graph_panel_y):
            panel.raw_data_line.set_data([], [])
            panel.fit_curve_line.set_data([], [])
            panel.e2_curve_line.set_data([], [])
            panel.gaussian_curve_line.set_data([], [])
            panel.blit_draw()

        self._last_gaussian_popt_x = None
        self._last_gaussian_popt_y = None
        self._last_gaussian_r2_x = None
        self._last_gaussian_r2_y = None
        self._last_h_info = None
        self._last_v_info = None
        self._last_second_moment = None
        self._latest_measurement = None
        self._latest_quality = None
        self._latest_propagation_sample = None
        self.camera_panel.fpsLabel.setText("Measured: -- FPS | -- ms/frame")

    def _draw_live_overlay(self, image, scale, centers, numbers_x, numbers_y, popt_x, popt_y, pixel_size,
                           zoom_factor, show_centroid=True, show_peak=True):
        cx, cy = centers["mass"]
        px, py = centers["peak"]

        h, w = image.shape[:2]
        overlay = image.copy()

        inv_zoom = 1.0 / zoom_factor if zoom_factor > 0 else 1.0
        ui_scale = inv_zoom * getattr(self, "overlay_text_scale", 0.5)
        grid_line_px = max(1, int(round(OVERLAY_GRID_LINE_BASE_PX * ui_scale)))
        curve_line_px = max(1, int(round(OVERLAY_CURVE_LINE_BASE_PX * ui_scale)))
        tick_len = max(2, int(round(OVERLAY_TICK_LEN_BASE_PX * ui_scale)))
        font_scale = OVERLAY_FONT_SCALE_BASE * ui_scale
        font_thick = max(1, int(round(font_scale * OVERLAY_FONT_THICK_MULTIPLIER)))
        crosshair_big = max(4, int(round(OVERLAY_CROSSHAIR_BIG_BASE_PX * ui_scale)))
        crosshair_small = max(3, int(round(OVERLAY_CROSSHAIR_SMALL_BASE_PX * ui_scale)))
        crosshair_centroid_thick = max(1, int(round(OVERLAY_CROSSHAIR_CENTROID_THICK_BASE_PX * ui_scale)))
        crosshair_thick = max(1, int(round(OVERLAY_CROSSHAIR_THICK_BASE_PX * ui_scale)))

        sensor_cx = w // 2
        sensor_cy = h // 2

        beamxfordirection = popt_x[1] if popt_x is not None else cx
        beamyfordirection = popt_y[1] if popt_y is not None else cy

        x_amp_sign = -1 if beamyfordirection < sensor_cy else 1
        y_amp_sign = 1 if beamxfordirection > sensor_cx else -1

        if popt_x is not None:
            fitted_x = gaussian_model(numbers_x, *popt_x)
            fitted_x = fitted_x - np.min(fitted_x)
            max_x = np.max(fitted_x)

            if max_x > 0:
                fitted_x_norm = fitted_x / max_x

                xs = (numbers_x * scale).astype(np.int32)
                ys = (sensor_cy + x_amp_sign * fitted_x_norm * self.overlay_curve_height).astype(np.int32)
                mask_x = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                pts_x = np.stack([xs[mask_x], ys[mask_x]], axis=1)

                if len(pts_x) > 1:
                    cv2.polylines(
                        overlay,
                        [pts_x.reshape(-1, 1, 2)],
                        False,
                        (255, 255, 0),
                        curve_line_px,
                        cv2.LINE_AA
                    )

        if popt_y is not None:
            fitted_y = gaussian_model(numbers_y, *popt_y)
            fitted_y = fitted_y - np.min(fitted_y)
            max_y = np.max(fitted_y)

            if max_y > 0:
                fitted_y_norm = fitted_y / max_y

                ys_curve = (numbers_y * scale).astype(np.int32)
                xs_curve = (sensor_cx + y_amp_sign * fitted_y_norm * self.overlay_curve_height).astype(np.int32)

                mask_y = (xs_curve >= 0) & (xs_curve < w) & (ys_curve >= 0) & (ys_curve < h)
                pts_y = np.stack([xs_curve[mask_y], ys_curve[mask_y]], axis=1)

                if len(pts_y) > 1:
                    cv2.polylines(
                        overlay,
                        [pts_y.reshape(-1, 1, 2)],
                        False,
                        (255, 255, 0),
                        curve_line_px,
                        cv2.LINE_AA
                    )

        if pixel_size is None or pixel_size <= 0:
            pixel_size = 3.45

        tick_um = 500
        tick_px = int(tick_um / pixel_size)

        cv2.line(overlay, (0, sensor_cy), (w, sensor_cy), (255, 255, 255), grid_line_px)
        for i, x in enumerate(range(sensor_cx, w, tick_px)):
            cv2.line(overlay, (x, sensor_cy - tick_len), (x, sensor_cy + tick_len), (255, 255, 255), grid_line_px)
            if i > 0:
                cv2.putText(overlay, f"{i * tick_um}", (x + 2, sensor_cy - tick_len - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
        for i, x in enumerate(range(sensor_cx, -1, -tick_px)):
            cv2.line(overlay, (x, sensor_cy - tick_len), (x, sensor_cy + tick_len), (255, 255, 255), grid_line_px)
            if i > 0:
                cv2.putText(overlay, f"-{i * tick_um}", (x + 2, sensor_cy - tick_len - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

        cv2.line(overlay, (sensor_cx, 0), (sensor_cx, h), (255, 255, 255), grid_line_px)
        for i, y in enumerate(range(sensor_cy, h, tick_px)):
            cv2.line(overlay, (sensor_cx - tick_len, y), (sensor_cx + tick_len, y), (255, 255, 255), grid_line_px)
            if i > 0:
                cv2.putText(overlay, f"{i * tick_um}", (sensor_cx + tick_len + 3, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
        for i, y in enumerate(range(sensor_cy, -1, -tick_px)):
            cv2.line(overlay, (sensor_cx - tick_len, y), (sensor_cx + tick_len, y), (255, 255, 255), grid_line_px)
            if i > 0:
                cv2.putText(overlay, f"-{i * tick_um}", (sensor_cx + tick_len + 3, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

        if show_centroid:
            cv2.drawMarker(overlay, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, crosshair_big, crosshair_centroid_thick)
        if show_peak:
            cv2.drawMarker(overlay, (px, py), (255, 0, 0), cv2.MARKER_CROSS, crosshair_small, crosshair_thick)

        cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0, dst=image)

    def capture_image(self):
        if not self._camera_connected:
            return

        if self.advanced_panel.holdMaxCheckbox.isChecked() and self._hold_max_frame is not None:
            dtype = np.uint8 if self.sensor_bit_depth <= 8 else np.uint16
            gray_raw = np.clip(np.round(self._hold_max_frame), 0, self.sensor_max_raw).astype(dtype)
            capture_source = "hold_maximum_envelope"
        elif self._latest_raw_frame is not None:
            gray_raw = self._latest_raw_frame
            capture_source = "live_frame"
        else:
            QMessageBox.information(self, "Capture", "No displayed frame is available yet.")
            return

        base = f"laser_profile_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{self.photo_count:04d}"

        save_ok = cv2.imwrite(f"{base}.tif", gray_raw)
        if not save_ok:
            QMessageBox.critical(self, "Capture Failed", f"Could not write {base}.tif -- capture aborted.")
            return
        np.save(f"{base}.npy", gray_raw)

        display16_path = None
        if _HAS_TIFFFILE:
            shift = max(0, 16 - self.sensor_bit_depth)
            raw_16_display = (gray_raw.astype(np.uint16) << shift)
            display16_path = f"{base}_display16.tif"
            tifffile.imwrite(display16_path, raw_16_display)
        elif not self._warned_no_tifffile:
            print("[beam profiler] tifffile is not installed (pip install tifffile) -- "
                  "skipping the eye-viewable scaled TIFF; the raw .tif/.npy were still saved.")
            self._warned_no_tifffile = True

        fresh_quality = self._latest_quality
        fresh_second_moment = self._last_second_moment
        if self._latest_processed_frame is not None:
            try:
                fresh_measurement = self._measure_frame_stage(self._latest_processed_frame, force_advanced=True)
            except Exception:
                fresh_measurement = None
            if fresh_measurement is not None:
                fresh_quality = fresh_measurement.quality
                fresh_second_moment = fresh_measurement.second_moment

        metadata = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "camera_backend": self.current_camera_backend_name,
            "dark_correction": {
                "enabled": self.advanced_panel.darkEnableCheckbox.isChecked(),
                "applied": bool(fresh_quality.dark_applied) if fresh_quality else False,
                "valid": bool(fresh_quality.dark_valid) if fresh_quality else False,
                "mismatch_reason": fresh_quality.dark_mismatch_reason if fresh_quality else "",
                "metadata": self._dark_metadata,
            },
            "signal_quality": asdict(fresh_quality) if fresh_quality else None,
            "capture_source": capture_source,
            "exposure_ms": self.current_exposure_ms,
            "pixel_pitch_um": self.analyze_panel.pixel_pitch_input.value(),
            "colormap": self.control_panel.cmapCombo.currentText(),
            "zoom_percent": self.control_panel.zoomSlider.value(),
            "overlay_curve_height_px": self.overlay_curve_height,
            "sensor_bit_depth": self.sensor_bit_depth,
            "sensor_max_raw": self.sensor_max_raw,
            "centroid_px": {"x": self.last_cx, "y": self.last_cy},
            "gaussian_fit_r2": {"x": self._last_gaussian_r2_x, "y": self._last_gaussian_r2_y},
            "x_profile": self._last_h_info,
            "y_profile": self._last_v_info,
            "second_moment": fresh_second_moment,
            "raw_file": f"{base}.tif",
            "array_file": f"{base}.npy",
            "display16_file": display16_path,
        }
        with open(f"{base}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Captured ({capture_source}): {base}.tif / {base}.npy / {base}.json" +
              (f" / {display16_path}" if display16_path else ""))
        self.status_capture_label.setText(f"Last capture: {base}.tif ({datetime.now().strftime('%H:%M:%S')})")
        self.photo_count += 1

    def update_camera_settings(self, exposure_ms):
        if exposure_ms <= 0:
            exposure_ms = 0.01
        exposure_sec = exposure_ms / 1000.0

        max_display_fps = float(self.advanced_panel.maxFpsSpinBox.value())
        max_possible_fps = 1.0 / exposure_sec
        target_fps = min(max_display_fps, max_possible_fps - 1.0)
        if target_fps < 1:
            target_fps = 1.0

        timer_ms = int(1000.0 / target_fps)

        self.current_exposure_ms = exposure_ms
        if self._camera_connected:
            self._camera_bridge.exposure_requested.emit(float(exposure_ms))
        self.timer.start(timer_ms)

        self.control_panel.sliderLabel.setText(f"Exposure: {exposure_ms:.2f}ms | Target Display Rate: {target_fps:.1f} FPS")

    def closeEvent(self, event):
        print("Shutting down camera worker...")
        self._save_settings()
        self.timer.stop()
        self._camera_bridge.stop_requested.emit()
        if not self._camera_thread.wait(3000):
            print("[camera worker] Shutdown exceeded 3 s; requesting interruption.")
            self._camera_thread.requestInterruption()
            self._camera_thread.quit()
            if not self._camera_thread.wait(1000):
                QMessageBox.critical(self, "Shutdown Error",
                    "The camera SDK did not return from a blocked read. Disconnect the camera before closing the process.")
                event.ignore(); return
        if self._log_file is not None:
            self._log_file.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon_path = resource_path("app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec())