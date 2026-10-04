import sys
import os
import time
import csv
import json
import uuid
import collections
import threading
import traceback
from dataclasses import dataclass, asdict, field
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
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon, QColor
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
APP_VERSION = "v5.0"
ORG_NAME = "VOILA Lab"

SENSOR_ASSUMED_MAX_RAW = 1023
SENSOR_DISPLAY_DIVISOR = (SENSOR_ASSUMED_MAX_RAW + 1) / 256.0

GAUSSIAN_FIT_INTERVAL = 3
BACKGROUND_PERCENTILE = 5.0
PROFILE_BAND_HALFWIDTH = 5
GAUSSIAN_R2_MIN = 0.995
LOG_EVERY_N_FRAMES = 10
MOMENT_TRUNCATION_LEVEL = 0.135
D4SIGMA_NOISE_SIGMA_K = 3.0
ADVANCED_METRICS_INTERVAL_S = 0.20
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
# Shared by both themes, so the toolbar font is identical in light and dark mode.
TOOLBAR_FONT_QSS = """
QToolBar QToolButton { font-size: 13pt; }
"""
DARK_QSS += TOOLBAR_FONT_QSS
LIGHT_QSS += TOOLBAR_FONT_QSS


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
    """One raw, user-logged measurement at one Z position.

    Never itself an "aggregate" -- if the same Z is logged multiple times,
    each logging action produces one more PropagationSample. Aggregation
    across replicates is performed transiently by aggregate_replicates(),
    producing a separate AggregatedPropagationPoint, never written back into
    PropagationPanel.samples.
    """
    timestamp: str
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex)
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

    # Frame-to-frame repeatability at the moment this sample was logged
    # (standard deviation across the live frames used to build the stored
    # median). Explicitly NOT "uncertainty of the median" -- it's a
    # repeatability scale, used only as a relative fit weight, never as a
    # calibrated absolute uncertainty (see fit_propagation_axis docstring).
    d4sigma_x_repeatability_um: float | None = None
    d4sigma_y_repeatability_um: float | None = None
    gaussian_x_repeatability_um: float | None = None
    gaussian_y_repeatability_um: float | None = None

    saturation_fraction: float = 0.0
    clipped: bool = False
    valid: bool = False

    # Broader quality gate: anything that can make a measurement
    # stable-but-wrong ends up here, not just clipped/saturation_fraction.
    # Populated at capture time from the app's own per-frame warning flags
    # (low SNR, central saturation, isolated hot-pixel saturation,
    # sensor/ROI edge truncation, D4-sigma non-convergence). Matched against
    # QUALITY_BLOCKING_FLAGS below.
    quality_flags: list[str] = field(default_factory=list)

    exposure_ms: float | None = 0.0
    pixel_pitch_um: float = 0.0
    wavelength_nm: float = 0.0
    source_frame_count: int = 1
    width_cv_percent: float | None = None


@dataclass(frozen=True)
class AggregatedPropagationPoint:
    """A derived fitting-input point: one or more PropagationSamples that
    share (approximately) the same Z, collapsed into one point.

    Never stored as a PropagationSample. Never round-tripped through
    save/load/export as if it were raw data -- always regenerated from
    PropagationPanel.samples on demand by aggregate_replicates().
    """
    z_mm: float
    d4sigma_x_um: float | None
    d4sigma_y_um: float | None
    gaussian_x_um: float | None
    gaussian_y_um: float | None

    # Between-replicate spread when replicate_count >= 2; otherwise falls
    # back to that single sample's own frame-level repeatability, if known.
    # A repeatability scale, not a calibrated uncertainty. None if no
    # spread information is available at all.
    d4sigma_x_spread_um: float | None
    d4sigma_y_spread_um: float | None
    gaussian_x_spread_um: float | None
    gaussian_y_spread_um: float | None

    replicate_count: int
    source_sample_ids: list[str]


# Quality flags that hard-exclude a sample from aggregation/fitting
# regardless of how repeatable it looked. A stable-but-wrong measurement
# (e.g. bad background subtraction, truncated wings) must be excluded
# outright, not merely down-weighted -- weighting only handles genuine
# random repeatability, never systematic error.
QUALITY_BLOCKING_FLAGS = frozenset({
    "CENTRAL_SATURATION",
    "SENSOR_EDGE_TRUNCATED",
    "ROI_EDGE_TRUNCATED",
    "ISOLATED_HOT_PIXEL_SATURATION",
    "D4SIGMA_NON_CONVERGED",
    "LOW_SNR",
})


def passes_quality_gate(sample):
    """Hard exclusion check. Returns (passes, blocking_reasons)."""
    if not sample.valid:
        return False, ["MARKED_INVALID"]
    if sample.clipped or sample.saturation_fraction >= 0.95:
        return False, ["CLIPPED_OR_SATURATED"]
    blocking = [f for f in sample.quality_flags if f in QUALITY_BLOCKING_FLAGS]
    if blocking:
        return False, blocking
    return True, []


def _spread(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 2:
        return None
    return float(np.std(arr, ddof=1))


def _median_or_none(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _merge_replicate_group(group):
    z_mm = float(np.median([s.z_mm for s in group]))
    n = len(group)

    def merged_field(attr_name, repeatability_attr):
        values = [getattr(s, attr_name) for s in group]
        point_value = _median_or_none(values)
        if n >= 2:
            spread = _spread(values)  # between-replicate spread, preferred
        else:
            spread = getattr(group[0], repeatability_attr)  # single replicate: fall back
        return point_value, spread

    dx, dx_spread = merged_field("d4sigma_x_um", "d4sigma_x_repeatability_um")
    dy, dy_spread = merged_field("d4sigma_y_um", "d4sigma_y_repeatability_um")
    gx, gx_spread = merged_field("gaussian_x_um", "gaussian_x_repeatability_um")
    gy, gy_spread = merged_field("gaussian_y_um", "gaussian_y_repeatability_um")

    return AggregatedPropagationPoint(
        z_mm=z_mm,
        d4sigma_x_um=dx, d4sigma_y_um=dy,
        gaussian_x_um=gx, gaussian_y_um=gy,
        d4sigma_x_spread_um=dx_spread, d4sigma_y_spread_um=dy_spread,
        gaussian_x_spread_um=gx_spread, gaussian_y_spread_um=gy_spread,
        replicate_count=n,
        source_sample_ids=[s.sample_id for s in group],
    )


def aggregate_replicates(samples, z_tol_mm=0.01):
    """Group samples that pass the quality gate by Z position (within
    z_tol_mm) and collapse each group into one AggregatedPropagationPoint.

    z_tol_mm defaults to 0.01 mm (10 micron), a plausible manual/mechanical
    stage repositioning precision -- NOT an effectively-exact tolerance,
    which would treat e.g. 100.0000 vs 100.0001 mm as two different planes.
    """
    accepted = [s for s in samples if s.z_mm is not None and passes_quality_gate(s)[0]]
    accepted.sort(key=lambda s: s.z_mm)

    groups = []
    for s in accepted:
        if groups and abs(s.z_mm - groups[-1][0].z_mm) <= z_tol_mm:
            groups[-1].append(s)
        else:
            groups.append([s])

    return [_merge_replicate_group(group) for group in groups]


PROPAGATION_SCOPE_NOTE = (
    "Current propagation analysis fits camera X and Y independently and is "
    "intended for stigmatic or appropriately aligned simple-astigmatic "
    "beams. General astigmatic beams require a full second-moment "
    "treatment that is not yet implemented here."
)


def second_moment_2d(image, background=None, clip_fraction=0.0, background_noise=None,
                     noise_sigma_k=D4SIGMA_NOISE_SIGMA_K):
    """
    Calculate D4-sigma beam parameters using an iterative integration area.

    The ROI is repeatedly centered on the measured centroid and sized from
    the measured second-moment beam widths. This helps against spatially
    localized contamination (a distant reflection or hot-pixel cluster)
    while retaining real low-intensity beam wings.

    NOTE: this iterative ROI is NOT a substitute for a good background
    estimate. If the caller passes a systematically biased `background`
    (or relies on the fallback below producing one), the very first,
    whole-image iteration can already be corrupted -- in which case the
    derived ROI clamps to the full image and this degenerates to a plain
    one-shot second moment. Check the returned "converged" and
    "integration_bounds" fields: bounds spanning (nearly) the whole image
    is a sign the background estimate upstream should be investigated
    rather than trusting the number as-is.

    IMPORTANT, non-obvious point: an unbiased `background` alone is NOT
    enough. Clipping `arr - background` to [0, None] gives every purely-
    noise background pixel a nonzero *expected* positive residual --
    E[max(N(0, sigma), 0)] = sigma / sqrt(2*pi) -- even when `background`
    exactly equals the true mean. That small per-pixel expectation, summed
    over every background pixel and weighted by squared distance in the
    second moment, is what actually blows up D4-sigma on any reasonably
    sized frame; fixing only the background *level* estimator (as opposed
    to also handling this clip-induced bias) was verified to leave D4-sigma
    wrong by several hundred percent even with an unbiased background.

    `background_noise` (the background standard deviation / robust sigma)
    is used to apply a noise-floor threshold instead of a bare clip-to-zero:
    only pixels more than `noise_sigma_k` standard deviations above
    background are counted as signal. At noise_sigma_k=3.0 this suppresses
    >99% of pure background noise while costing only a small, predictable
    underestimate of very faint real beam wings (tested down to a few
    percent). If `background_noise` isn't supplied and can't be estimated,
    this falls back to a plain clip-to-zero (the old, noise-sensitive
    behavior) rather than silently doing nothing.

    clip_fraction is an additional, optional fixed fraction-of-peak cutoff
    applied on top of the noise-floor threshold; it should normally remain
    0.0 for D4-sigma.
    """

    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return None

    h, w = arr.shape

    if background is None or background_noise is None:
        # Border-median + MAD estimate rather than a fixed low percentile
        # of the whole image: a fixed low percentile (e.g. the 5th) is a
        # biased estimator of the true background whenever the background
        # has any noise (it sits ~1.6 sigma below the true mean for
        # Gaussian noise). robust_background_stats samples only the image
        # border (assumed beam-free) and uses median/MAD, which is
        # unbiased for symmetric noise and does not need the beam to
        # occupy a known fraction of the frame.
        est_background, est_noise = robust_background_stats(arr)
        if background is None:
            background = est_background
        if background_noise is None:
            background_noise = est_noise

    diff = arr - np.float32(background)

    if background_noise and background_noise > 0:
        # Noise-floor threshold: only pixels statistically distinguishable
        # from background noise count as signal. This is what actually
        # prevents the clip-to-zero bias described above from dominating
        # the second moment; a good background LEVEL estimate alone is not
        # sufficient (see note above).
        corrected = np.where(
            diff > np.float32(noise_sigma_k * background_noise),
            diff,
            np.float32(0.0)
        )
    else:
        # No usable noise estimate -- fall back to a plain clip-to-zero.
        # This reproduces the old, noise-sensitive behavior rather than
        # silently skipping the correction.
        corrected = np.clip(diff, 0.0, None)

    peak = float(corrected.max())

    if peak <= 0:
        return None

    # Do NOT normally threshold D4-sigma.
    if clip_fraction > 0:
        corrected = np.where(
            corrected >= clip_fraction * peak,
            corrected,
            np.float32(0.0)
        )

    # ----------------------------------------------------------
    # Initial estimate using the full available image.
    # ----------------------------------------------------------

    weights = corrected

    previous_result = None

    MAX_ITERATIONS = 10
    CONVERGENCE_TOLERANCE = 0.01

    # ISO-style integration-area multiplier.
    # Integration region will be approximately 3 beam widths.
    ROI_SCALE = 3.0

    final_bounds = (0, w, 0, h)

    converged = False
    iterations_used = 0

    for _ in range(MAX_ITERATIONS):

        m = cv2.moments(weights, binaryImage=False)

        total = float(m["m00"])

        if total <= 0:
            return None

        cx = float(m["m10"] / total)
        cy = float(m["m01"] / total)

        var_x = max(
            float(m["m20"] / total - cx * cx),
            0.0
        )

        var_y = max(
            float(m["m02"] / total - cy * cy),
            0.0
        )

        cov_xy = float(
            m["m11"] / total - cx * cy
        )

        d4sigma_x = 4.0 * np.sqrt(var_x)
        d4sigma_y = 4.0 * np.sqrt(var_y)

        if d4sigma_x <= 0 or d4sigma_y <= 0:
            # A degenerate iteration (e.g. an over-aggressive crop that
            # left no signal) shouldn't discard an otherwise-valid prior
            # estimate. Fall back to the last committed ROI/result instead
            # of aborting the whole measurement.
            if previous_result is None:
                return None
            break

        iterations_used += 1

        # Check convergence.
        if previous_result is not None:

            prev_x, prev_y = previous_result

            change_x = abs(d4sigma_x - prev_x) / max(prev_x, 1e-12)
            change_y = abs(d4sigma_y - prev_y) / max(prev_y, 1e-12)

            if (
                change_x < CONVERGENCE_TOLERANCE
                and
                change_y < CONVERGENCE_TOLERANCE
            ):
                converged = True
                break

        previous_result = (
            d4sigma_x,
            d4sigma_y
        )

        # ------------------------------------------------------
        # Build next integration area.
        #
        # The total ROI width is approximately
        # ROI_SCALE * D4-sigma.
        # ------------------------------------------------------

        half_width_x = max(
            int(np.ceil(0.5 * ROI_SCALE * d4sigma_x)),
            5
        )

        half_width_y = max(
            int(np.ceil(0.5 * ROI_SCALE * d4sigma_y)),
            5
        )

        x0 = max(
            0,
            int(np.floor(cx - half_width_x))
        )

        x1 = min(
            w,
            int(np.ceil(cx + half_width_x + 1))
        )

        y0 = max(
            0,
            int(np.floor(cy - half_width_y))
        )

        y1 = min(
            h,
            int(np.ceil(cy + half_width_y + 1))
        )

        final_bounds = (
            x0,
            x1,
            y0,
            y1
        )

        # Preserve original coordinate system.
        weights = np.zeros_like(corrected)

        weights[y0:y1, x0:x1] = corrected[y0:y1, x0:x1]

    # ----------------------------------------------------------
    # Final moment calculation
    # ----------------------------------------------------------

    m = cv2.moments(weights, binaryImage=False)

    total = float(m["m00"])

    if total <= 0:
        return None

    cx = float(m["m10"] / total)
    cy = float(m["m01"] / total)

    var_x = max(
        float(m["m20"] / total - cx * cx),
        0.0
    )

    var_y = max(
        float(m["m02"] / total - cy * cy),
        0.0
    )

    cov_xy = float(
        m["m11"] / total - cx * cy
    )

    # ----------------------------------------------------------
    # Principal-axis calculation
    # ----------------------------------------------------------

    cov = np.array(
        [
            [var_x, cov_xy],
            [cov_xy, var_y]
        ],
        dtype=np.float64
    )

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if (
        np.any(~np.isfinite(eigenvalues))
        or np.any(eigenvalues < -1e-9)
    ):
        return None

    eigenvalues = np.clip(
        eigenvalues,
        0.0,
        None
    )

    order = np.argsort(eigenvalues)[::-1]

    major_var = float(eigenvalues[order[0]])
    minor_var = float(eigenvalues[order[1]])

    major_vec = eigenvectors[:, order[0]]

    orientation_deg = float(
        np.degrees(
            np.arctan2(
                major_vec[1],
                major_vec[0]
            )
        )
    )

    while orientation_deg >= 90.0:
        orientation_deg -= 180.0

    while orientation_deg < -90.0:
        orientation_deg += 180.0

    d4sigma_x = 4.0 * np.sqrt(var_x)
    d4sigma_y = 4.0 * np.sqrt(var_y)

    d4sigma_major = 4.0 * np.sqrt(major_var)
    d4sigma_minor = 4.0 * np.sqrt(minor_var)

    ellipticity = (
        float(d4sigma_minor / d4sigma_major)
        if d4sigma_major > 0
        else None
    )

    # ----------------------------------------------------------
    # Truncation check
    # ----------------------------------------------------------

    x0, x1, y0, y1 = final_bounds

    # Check whether the integration area itself is touching
    # the camera/selected ROI boundary.
    touches_sensor_edge = (
        x0 <= 0
        or y0 <= 0
        or x1 >= w
        or y1 >= h
    )

    # Examine signal around the final integration-area boundary.
    roi = corrected[y0:y1, x0:x1]

    if roi.size:
        border = np.concatenate(
            [
                roi[0, :],
                roi[-1, :],
                roi[:, 0],
                roi[:, -1]
            ]
        )

        border_peak = float(border.max())
    else:
        border_peak = 0.0

    truncated = bool(
        touches_sensor_edge
        and
        border_peak >= MOMENT_TRUNCATION_LEVEL * peak
    )

    return {
        "centroid_x_px": cx,
        "centroid_y_px": cy,

        "var_x_px2": var_x,
        "var_y_px2": var_y,

        "cov_xy_px2": cov_xy,

        "major_var_px2": major_var,
        "minor_var_px2": minor_var,

        "d4sigma_x_px": d4sigma_x,
        "d4sigma_y_px": d4sigma_y,

        "d4sigma_major_px": d4sigma_major,
        "d4sigma_minor_px": d4sigma_minor,

        "orientation_deg": orientation_deg,
        "ellipticity": ellipticity,

        "truncated": truncated,
        "valid": (not truncated) and converged,

        "background": float(background),
        "background_noise": float(background_noise) if background_noise else 0.0,
        "noise_sigma_k": float(noise_sigma_k),

        "total_corrected_counts": total,

        # Diagnostics: whether the ROI iteration actually settled, and how
        # many iterations it took. If converged is False, MAX_ITERATIONS
        # was reached without meeting CONVERGENCE_TOLERANCE -- treat the
        # result with suspicion rather than assuming it's a stable number.
        "converged": converged,
        "iterations": iterations_used,

        # Useful for debugging / future display. If this rectangle spans
        # (nearly) the whole image, the adaptive ROI never actually
        # shrank -- almost always a sign of a bad upstream background
        # estimate rather than a beam that legitimately fills the frame.
        "integration_bounds": {
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1
        }
    }


def propagation_radius_model(z_mm, w0_um, z0_mm, m2, wavelength_nm):
    z_um = (np.asarray(z_mm, dtype=np.float64) - z0_mm) * 1000.0
    term = (m2 * wavelength_nm * 1e-3 * z_um) / (np.pi * w0_um * w0_um)
    return w0_um * np.sqrt(1.0 + term * term)


def bounds_hit(popt, lower, upper, rel_tol=0.01, names=("w0", "z0", "m2")):
    hits = []
    for name, val, lo, hi in zip(names, popt, lower, upper):
        span = hi - lo
        if span <= 0:
            continue
        if (val - lo) / span < rel_tol:
            hits.append((name, "lower"))
        elif (hi - val) / span < rel_tol:
            hits.append((name, "upper"))
    return hits


def identifiability_diagnostics(pcov):
    """Dimensionless identifiability diagnostics.

    Deliberately does NOT report cond(pcov) directly: w0 (~um), z0 (~mm),
    and m2 (dimensionless, O(1)) have wildly different numerical scales, so
    a raw covariance condition number is dominated by unit choice rather
    than by how genuinely degenerate the fit directions are. The
    correlation matrix is scale-free and directly interpretable; its own
    condition number is reported as a secondary, still-dimensionless
    diagnostic.
    """
    if pcov is None or pcov.shape != (3, 3) or not np.all(np.isfinite(pcov)):
        return {"covariance_ok": False, "correlation": None,
                "max_abs_correlation": None, "correlation_condition_number": None}
    d = np.sqrt(np.diag(pcov))
    if np.any(d <= 0):
        return {"covariance_ok": False, "correlation": None,
                "max_abs_correlation": None, "correlation_condition_number": None}
    corr = pcov / np.outer(d, d)
    off_diag = corr[~np.eye(3, dtype=bool)]
    return {
        "covariance_ok": True,
        "correlation": corr,
        "max_abs_correlation": float(np.max(np.abs(off_diag))) if off_diag.size else None,
        "correlation_condition_number": float(np.linalg.cond(corr)),
    }


def coverage_zones(z, z0_mm, z_r_mm):
    """Per-side near/transition/far plane counts, relative to the fitted
    waist z0 and Rayleigh range zR. Does NOT judge adequacy on its own --
    see coverage_summary() for that.
    """
    z = np.asarray(z, dtype=np.float64)
    delta = z - z0_mm
    upstream = delta <= 0
    downstream = delta >= 0
    abs_delta = np.abs(delta)
    return {
        "upstream_near": int(np.count_nonzero(upstream & (abs_delta <= z_r_mm))),
        "upstream_transition": int(np.count_nonzero(upstream & (abs_delta > z_r_mm) & (abs_delta < 2 * z_r_mm))),
        "upstream_far": int(np.count_nonzero(upstream & (abs_delta >= 2 * z_r_mm))),
        "downstream_near": int(np.count_nonzero(downstream & (abs_delta <= z_r_mm))),
        "downstream_transition": int(np.count_nonzero(downstream & (abs_delta > z_r_mm) & (abs_delta < 2 * z_r_mm))),
        "downstream_far": int(np.count_nonzero(downstream & (abs_delta >= 2 * z_r_mm))),
        "z0_mm": z0_mm, "z_r_mm": z_r_mm,
    }


def coverage_summary(zones, unique_planes):
    """Turn raw zone counts into the adequacy judgment used by the UI. Does
    NOT require far-field points on both sides of the waist -- commonly
    cited ISO-style guidance wants roughly half the planes near the waist
    and roughly half beyond 2*zR, without requiring that far group to split
    evenly across upstream/downstream. One-sided far coverage is reported
    (`one_sided_far`) but does not by itself fail `iso_style_target_met`.
    """
    if zones is None:
        return {
            "usable_provisional": unique_planes >= 4, "iso_style_target_met": False,
            "near_total": 0, "far_total": 0, "one_sided_far": False,
            "needs_more_near": True, "needs_more_far": True, "zones": None,
        }
    near_total = zones["upstream_near"] + zones["downstream_near"]
    far_total = zones["upstream_far"] + zones["downstream_far"]
    one_sided_far = (zones["upstream_far"] == 0) != (zones["downstream_far"] == 0) if far_total > 0 else False
    return {
        "usable_provisional": unique_planes >= 4,
        "iso_style_target_met": bool(unique_planes >= 10 and near_total >= 4 and far_total >= 4),
        "near_total": near_total, "far_total": far_total,
        "one_sided_far": one_sided_far,
        "needs_more_near": near_total < 4, "needs_more_far": far_total < 4,
        "zones": zones,
    }


def describe_coverage(summary):
    """Plain-language coverage sentence (near/far in words, not
    'near/far=8/0')."""
    if summary is None:
        return "Coverage could not be evaluated."
    near, far = summary["near_total"], summary["far_total"]
    if summary["iso_style_target_met"]:
        base = f"Coverage target met: {near} near-waist and {far} far-field planes."
    else:
        missing = []
        if summary["needs_more_near"]:
            missing.append(f"more near-waist planes (have {near}, target ≥ 4)")
        if summary["needs_more_far"]:
            missing.append(f"more far-field planes (have {far}, target ≥ 4)")
        reason = "; ".join(missing) if missing else "more unique Z planes overall (target ≥ 10)"
        base = f"Coverage incomplete: {reason}."
    if summary.get("one_sided_far"):
        base += " Far-field coverage is one-sided only (present on just one side of the waist)."
    return base


def describe_identifiability(fit):
    """Plain-language identifiability sentence, listing the actual reason(s)
    instead of a bare 'poorly constrained' label. Takes the full
    fit_propagation_axis() result (not just the identifiability dict)
    because a broad/unphysical-crossing M^2 confidence interval must also
    be checked -- correlation alone can miss a fit that is individually
    poorly constrained on every parameter without any pairwise correlation
    crossing a threshold.
    """
    ident = fit["identifiability"]
    reasons = []
    if not ident.get("covariance_ok", False):
        reasons.append("the fit's covariance could not be computed (numerically singular)")
    elif ident.get("max_abs_correlation") is not None and ident["max_abs_correlation"] > 0.95:
        reasons.append(f"strong correlation between fitted parameters (|r|={ident['max_abs_correlation']:.2f})")
    if fit["bounds_hit"]:
        names = ", ".join(f"{n} at {side} bound" for n, side in fit["bounds_hit"])
        reasons.append(f"parameter(s) pinned at a fit boundary ({names})")
    m2, m2_error = fit["m2"], fit["m2_error"]
    if np.isfinite(m2_error) and m2 > 0:
        relative_m2_uncertainty = m2_error / m2
        crosses_unphysical = (m2 - m2_error) <= 1.0
        if relative_m2_uncertainty > 0.5 or crosses_unphysical:
            detail = f"M² = {m2:.2f} ± {m2_error:.2f} ({100*relative_m2_uncertainty:.0f}% relative)"
            if crosses_unphysical:
                detail += "; this range includes M² ≤ 1"
            reasons.append(f"M² confidence interval is very broad ({detail})")
    if not reasons:
        return "Parameters appear reasonably well constrained by this dataset."
    return "Poorly constrained: " + "; ".join(reasons) + "."


def fit_propagation_axis(z_mm, diameter_um, wavelength_nm, spread_um=None):
    """Fit one axis's beam caustic.

    `spread_um`, if given, is used as a *relative* weight only
    (absolute_sigma=False) -- repeatability/replicate spread is not a
    calibrated absolute measurement uncertainty, so the fit's covariance
    scale still comes from the residuals, not from an unproven noise model.
    Entries that are None/non-finite/non-positive fall back to the median
    of the other provided spreads; if none are available at all, the fit is
    unweighted.

    Returns None if there isn't enough independent data to fit, otherwise a
    dict. Notable fields:

      - "model_constraints_satisfied" (renamed from the old "physical"):
        only checks the point estimate lies in the physically allowed
        region. It does NOT mean the result is validated -- see
        "identifiability" and coverage_summary() for that.
      - "identifiability": dict from identifiability_diagnostics().
      - "bounds_hit": list of (param_name, "lower"|"upper").
      - "weighted" / "sigma_kind": whether/how spread was used.
      - "zones": near/transition/far-per-side breakdown from
        coverage_zones(); adequacy is judged separately by
        coverage_summary(), not baked in here.
      - "nrmse" is retained for backward-compatible display but is
        descriptive-only -- it is NOT a validity signal (it can look
        excellent on flat, undersampled data).
    """
    z = np.asarray(z_mm, dtype=np.float64)
    d = np.asarray(diameter_um, dtype=np.float64)
    mask = np.isfinite(z) & np.isfinite(d) & (d > 0)
    z, d = z[mask], d[mask]

    weighted = False
    sigma = None
    if spread_um is not None:
        spread_arr = np.asarray(spread_um, dtype=np.float64)
        if spread_arr.shape == mask.shape:
            spread_arr = spread_arr[mask]
            known = spread_arr[np.isfinite(spread_arr) & (spread_arr > 0)]
            if known.size > 0:
                fallback = float(np.median(known))
                sigma = np.where(np.isfinite(spread_arr) & (spread_arr > 0), spread_arr, fallback)
                sigma = sigma / 2.0  # diameter-scale spread -> radius-scale sigma for the w=d/2 fit
                weighted = True

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
        popt, pcov = curve_fit(
            model, z, w, p0=[w0_guess, z0_guess, 1.2],
            bounds=(lower, upper), maxfev=30000,
            sigma=sigma, absolute_sigma=False,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    predicted = model(z, *popt)
    residuals = w - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    # Descriptive-only; do not use as a validity signal in the UI.
    nrmse = rmse / max(float(np.ptp(w)), float(np.mean(w)), 1e-12)
    weighted_rmse = float(np.sqrt(np.mean((residuals / sigma) ** 2))) if sigma is not None else None

    errors = np.sqrt(np.diag(pcov)) if pcov.shape == (3, 3) and np.all(np.isfinite(pcov)) else np.full(3, np.nan)
    w0, z0, m2 = map(float, popt)
    wavelength_um = wavelength_nm * 1e-3
    z_r_um = np.pi * w0 * w0 / (m2 * wavelength_um)
    z_r_mm = z_r_um / 1000.0
    theta_half_rad = m2 * wavelength_um / (np.pi * w0)

    zones = coverage_zones(z, z0, z_r_mm) if z_r_mm > 0 else None
    left = bool(np.any(z < z0))
    right = bool(np.any(z > z0))

    # M^2 lower bound is 0.2, not 1.0 (see `lower` above), intentionally: a
    # fit landing below 1 is diagnostic evidence of a measurement problem
    # (bad background, truncation, wrong width method) and should be
    # visible rather than clamped away by the optimizer's own bounds.
    model_constraints_satisfied = bool(m2 >= 1.0 and w0 > 0 and z_r_mm > 0 and np.all(np.isfinite(popt)))

    return {
        "w0_um": w0, "z0_mm": z0, "m2": m2,
        "w0_error_um": float(errors[0]), "z0_error_mm": float(errors[1]),
        "m2_error": float(errors[2]), "z_r_mm": z_r_mm,
        "theta_half_mrad": theta_half_rad * 1000.0,
        "theta_full_mrad": theta_half_rad * 2000.0,
        "bpp_mm_mrad": w0 / 1000.0 * theta_half_rad * 1000.0,
        "rmse_um_radius": rmse, "nrmse": nrmse, "weighted_rmse_um_radius": weighted_rmse,
        "weighted": weighted, "sigma_kind": "relative_repeatability" if weighted else None,
        "left_sampled": left, "right_sampled": right,
        "model_constraints_satisfied": model_constraints_satisfied,
        "identifiability": identifiability_diagnostics(pcov),
        "bounds_hit": bounds_hit(popt, lower, upper),
        "zones": zones, "unique_planes": int(z.size),
        "z": z, "observed_radius_um": w, "predicted_radius_um": predicted,
        "_lower_bounds": lower, "_upper_bounds": upper, "_sigma": sigma,
    }


def bootstrap_ci(z, w_radius_um, wavelength_nm, sigma, lower, upper, n_boot=300, seed=0):
    """On-demand parametric bootstrap for the caustic fit.

    Not part of the live refit() path -- only run when the user explicitly
    asks for it ("Finalize analysis"), because it means ~n_boot extra
    curve_fit calls. Parametric (not pairs/resample-points) bootstrap is
    used so every synthetic realization keeps the same Z positions as the
    real experiment -- resampling points with replacement can under-sample
    or omit whole Z regions and produce failed fits for reasons unrelated
    to the parameter uncertainty being estimated.

    Uses `sigma` as the per-point noise scale if available (same
    relative-weighting caveat as the live fit); otherwise falls back to a
    constant noise scale from the fit's own RMSE.
    """
    def model(zvals, w0, z0, m2):
        return propagation_radius_model(zvals, w0, z0, m2, wavelength_nm)

    p0_result = None
    try:
        p0_result, _ = curve_fit(model, z, w_radius_um, p0=[float(np.min(w_radius_um)) or 1.0, float(z[np.argmin(w_radius_um)]), 1.2],
                                  bounds=(lower, upper), maxfev=30000, sigma=sigma, absolute_sigma=False)
    except (RuntimeError, ValueError, FloatingPointError):
        return {"n_successful": 0}

    predicted0 = model(z, *p0_result)
    noise_scale = sigma if sigma is not None else np.full_like(z, float(np.std(w_radius_um - predicted0)) or 1e-6)

    rng = np.random.default_rng(seed)
    boot_params = []
    for _ in range(n_boot):
        synthetic = predicted0 + rng.normal(0.0, noise_scale)
        try:
            popt, _ = curve_fit(model, z, synthetic, p0=p0_result, bounds=(lower, upper),
                                 maxfev=30000, sigma=sigma, absolute_sigma=False)
            boot_params.append(popt)
        except (RuntimeError, ValueError, FloatingPointError):
            continue

    if not boot_params:
        return {"n_successful": 0}
    boot_params = np.asarray(boot_params)
    return {
        "n_successful": len(boot_params),
        "n_requested": n_boot,
        "w0_ci95_um": tuple(np.percentile(boot_params[:, 0], [2.5, 97.5]).tolist()),
        "z0_ci95_mm": tuple(np.percentile(boot_params[:, 1], [2.5, 97.5]).tolist()),
        "m2_ci95": tuple(np.percentile(boot_params[:, 2], [2.5, 97.5]).tolist()),
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

    e2_level = background + np.exp(-2.0) * signal
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

    # Capability flags, so the GUI can enable/disable acquisition- and
    # time-series-dependent controls by asking the backend what it
    # supports, rather than string-matching backend names everywhere.
    supports_exposure = True
    supports_dark_capture = True
    supports_temporal_analysis = True   # Beam Stability / propagation logging meaningful
    supports_frame_averaging = True
    supports_peak_hold = True
    # If True, this backend represents a single static observation, not an
    # ongoing live acquisition. CameraWorker stops rescheduling further
    # get_frame() calls after the first one succeeds; the GUI must use
    # reprocess_current_frame() (re-analysis of the same observation) to
    # reflect setting changes, rather than expecting new frames to arrive.
    is_static_source = False

    def open(self):
        raise NotImplementedError

    def close(self):
        pass

    def get_frame(self):
        raise NotImplementedError

    def set_exposure_ms(self, exposure_ms):
        pass

    def set_dark_mode(self, enabled):
        """Optional hook: when enabled, a backend that supports it should
        return frames representing "no signal" (e.g. a real camera would
        expect the user to block the beam; a simulated backend can just
        stop generating one). No-op by default so every existing backend
        remains a valid no-op implementer without changes."""
        pass

    def get_bit_depth(self):
        return 8

    def get_sensor_max_raw(self):
        return (2 ** self.get_bit_depth()) - 1

    def get_sensor_saturation_known(self):
        # False means get_sensor_max_raw() is a container/display range,
        # not a verified physical sensor ADC ceiling -- the GUI should
        # still use it for display scaling and container-relative
        # percentages, but must suppress saturation warnings/flags.
        return True

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


class DemoCameraBackend(CameraBackend):
    """Ideal, noiseless simulated TEM00 Gaussian beam source.

    This is deliberately NOT a noisy camera emulator: zero background,
    zero read/shot noise, zero drift, fixed peak intensity. The point is a
    known ground truth to validate the analysis pipeline against, not a
    realistic sensor simulation. Every acquisition-loop-facing behavior
    (dtype, bit depth, pacing) matches what a real backend would provide,
    so nothing downstream needs to know this isn't real hardware.

    `diameter_um` is the simulated beam's 1/e^2 FULL diameter. Internally,
    the standard TEM00 formula uses the 1/e^2 RADIUS w = diameter_um / 2:
        I(x, y) = I0 * exp(-2 * r^2 / w^2)
    So a requested "500 µm" diameter means w = 250 µm is what actually
    appears in the exponent -- this is the ground-truth calibration
    reference every other measurement in the app should be checked against.
    """

    name = "Demo / Simulated Camera"

    # Exposure deliberately affects pacing only, never brightness (see
    # set_exposure_ms below) -- so leaving the Exposure control enabled
    # would misleadingly suggest it does something to this backend.
    supports_exposure = False

    WIDTH_PX = 640
    HEIGHT_PX = 480
    PIXEL_PITCH_UM = 3.45
    BIT_DEPTH = 12
    PEAK_COUNTS = round(0.75 * ((2 ** BIT_DEPTH) - 1))  # ~3071 of 4095, fixed forever
    MAX_FPS = 60.0

    def __init__(self, diameter_um=500.0):
        self._diameter_um = float(diameter_um)
        self._exposure_ms = 5.0
        self._dark_mode = False
        # Fixed pixel grid, computed once since resolution never changes.
        yy, xx = np.mgrid[0:self.HEIGHT_PX, 0:self.WIDTH_PX]
        self._xx = xx.astype(np.float64)
        self._yy = yy.astype(np.float64)
        self._cx = 0.70 * (self.WIDTH_PX - 1)
        self._cy = 0.30 * (self.HEIGHT_PX - 1)

    def open(self):
        pass

    def close(self):
        pass

    def set_exposure_ms(self, exposure_ms):
        # Exposure affects acquisition PACING ONLY -- never the simulated
        # intensity. Coupling the two would turn a controlled reference
        # beam into a moving target every time exposure is adjusted.
        self._exposure_ms = max(0.01, float(exposure_ms))

    def set_dark_mode(self, enabled):
        self._dark_mode = bool(enabled)

    def set_beam_diameter_um(self, diameter_um):
        if diameter_um > 0:
            self._diameter_um = float(diameter_um)

    def get_frame(self):
        # Pace like a real camera would: exposure sets the floor, but never
        # faster than MAX_FPS and (given the GUI's 0.01-100ms exposure
        # range) never longer than 100ms, so the worker's Qt event loop
        # stays responsive to queued commands (diameter changes, disconnect,
        # etc.) even at the slowest configured exposure.
        frame_period_ms = max(self._exposure_ms, 1000.0 / self.MAX_FPS)
        time.sleep(frame_period_ms / 1000.0)

        if self._dark_mode:
            return np.zeros((self.HEIGHT_PX, self.WIDTH_PX), dtype=np.uint16)

        w_um = self._diameter_um / 2.0
        w_px = max(w_um / self.PIXEL_PITCH_UM, 0.5)
        r2 = (self._xx - self._cx) ** 2 + (self._yy - self._cy) ** 2
        img = self.PEAK_COUNTS * np.exp(-2.0 * r2 / (w_px * w_px))
        return np.clip(np.rint(img), 0, (2 ** self.BIT_DEPTH) - 1).astype(np.uint16)

    def get_bit_depth(self):
        return self.BIT_DEPTH

    def get_pixel_pitch_um(self):
        return self.PIXEL_PITCH_UM

    def get_detector_size(self):
        return (self.WIDTH_PX, self.HEIGHT_PX)

    def get_info_text(self):
        return (f"Backend: {self.name}\n"
                f"Ideal noiseless TEM00 Gaussian reference beam\n"
                f"Resolution: {self.WIDTH_PX}x{self.HEIGHT_PX}, "
                f"pixel pitch {self.PIXEL_PITCH_UM} um\n"
                f"Bit depth: {self.BIT_DEPTH}-bit (uint16), peak fixed at "
                f"{self.PEAK_COUNTS} counts\n"
                f"Simulated 1/e^2 diameter: {self._diameter_um:.1f} um")

    @staticmethod
    def list_available_devices():
        return [("Simulated Gaussian Beam", "demo")]


UPLOAD_MAX_DIMENSION_WARN_PX = 2048
UPLOAD_IMAGE_EXTENSIONS_NPY = (".npy",)
UPLOAD_IMAGE_EXTENSIONS_RASTER = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")


def detect_upload_metadata(path, channel="luminance"):
    """Load an image file for the Uploaded Image backend and resolve its
    analysis metadata, entirely on the calling (GUI) thread -- no camera
    I/O happens later on the worker thread for this backend; the array
    returned here is exactly what gets analyzed.

    Returns a dict with keys:
        array, pixel_pitch_um (float or None), sensor_bit_depth (int),
        sensor_max_raw (int), sensor_saturation_known (bool),
        is_rgb_source (bool), warnings (list[str])
    Raises ValueError with a clear message on anything unusable.
    """
    warnings_out = []
    ext = os.path.splitext(path)[1].lower()

    pixel_pitch_um = None
    sensor_bit_depth = None
    sensor_max_raw = None
    sensor_saturation_known = False
    is_rgb_source = False

    if ext in UPLOAD_IMAGE_EXTENSIONS_NPY:
        arr = np.load(path, allow_pickle=False)
        if arr.dtype == np.dtype(object) or np.iscomplexobj(arr):
            raise ValueError("Unsupported .npy array type (object/complex arrays are not supported).")
        if arr.ndim != 2:
            raise ValueError(f"Unsupported .npy shape {arr.shape}; expected a single-channel 2D array (H x W).")
        if not np.all(np.isfinite(arr)):
            raise ValueError(".npy array contains NaN/Inf values.")
        if np.any(arr < 0):
            raise ValueError(".npy array contains negative values; a beam-intensity array should not.")

        if arr.dtype.kind == "f":
            # Accept floating-point synthetic data (e.g. an np.exp(...)
            # beam), but only if it already looks like digitized counts --
            # round and cast directly. Never rescale/normalize by the
            # array's own max: that would silently redefine what the
            # values mean, which is worse than just requiring the caller
            # to pick a convention themselves.
            max_val = float(arr.max())
            if max_val > 65535.0:
                raise ValueError(
                    f".npy array is floating-point with max value {max_val:.4g}, which exceeds the "
                    f"supported 16-bit range (0-65535). This does not rescale/normalize float data "
                    f"(that would silently change what the values mean) -- scale your synthetic array "
                    f"to a chosen peak-count convention yourself before saving, "
                    f"e.g. beam = beam / beam.max() * 3000."
                )
            arr = np.rint(arr).astype(np.uint16)
        elif arr.dtype.kind in "ui":
            max_val = int(arr.max())
            if max_val <= 255:
                arr = arr.astype(np.uint8)
            elif max_val <= 65535:
                arr = arr.astype(np.uint16)
            else:
                raise ValueError(
                    f".npy integer values (max={max_val}) exceed the supported 16-bit range "
                    f"(0-65535); a direct dtype cast would silently wrap around rather than "
                    f"represent these values correctly."
                )
        else:
            raise ValueError(
                f".npy array has unsupported dtype {arr.dtype}; expected an unsigned/signed "
                f"integer array or a finite non-negative floating-point array."
            )

        sidecar_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r") as f:
                    meta = json.load(f)
                p = meta.get("pixel_pitch_um")
                bd = meta.get("sensor_bit_depth")
                smr = meta.get("sensor_max_raw")
                valid = (
                    isinstance(p, (int, float)) and 0 < p < 1000.0
                    and isinstance(bd, (int, float)) and 1 <= bd <= 32
                    and isinstance(smr, (int, float)) and smr > 0
                    and smr <= (2 ** int(bd)) - 1 + 1e-6
                )
                if valid:
                    pixel_pitch_um = float(p)
                    sensor_bit_depth = int(bd)
                    sensor_max_raw = int(smr)
                    sensor_saturation_known = True
                    warnings_out.append(
                        f"VOILALab sidecar detected: image scale {pixel_pitch_um:.3f} \u00b5m/px, "
                        f"{sensor_bit_depth}-bit sensor (trusted)."
                    )
                else:
                    warnings_out.append(
                        "A matching .json sidecar was found but its values did not validate "
                        "(missing/out-of-range pixel_pitch_um, sensor_bit_depth, or sensor_max_raw) "
                        "-- enter image scale manually."
                    )
            except Exception:
                warnings_out.append("A matching .json sidecar was found but could not be read -- enter image scale manually.")
        else:
            warnings_out.append("No matching VOILALab .json sidecar found -- enter image scale manually.")

    elif ext in UPLOAD_IMAGE_EXTENSIONS_RASTER:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Could not decode image file: {os.path.basename(path)}")
        if raw.ndim == 2:
            arr = raw
        elif raw.ndim == 3:
            is_rgb_source = True
            if raw.shape[2] == 4:
                warnings_out.append("Image has an alpha channel; alpha is discarded (RGB/BGR channels only are used).")
                raw = raw[:, :, :3]
            channel_map = {"blue": 0, "green": 1, "red": 2}
            if channel in channel_map:
                arr = raw[:, :, channel_map[channel]]
                warnings_out.append(
                    f"Multi-channel source -- using the {channel} channel only. Not verified to be "
                    f"radiometrically linear; treat quantitative results with caution."
                )
            else:
                arr = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                warnings_out.append(
                    "Multi-channel (RGB) source converted via standard luminance weighting. "
                    "This is NOT necessarily proportional to optical intensity (ordinary images are "
                    "typically gamma/sRGB-encoded, and a color camera's channel response depends on "
                    "wavelength) -- treat quantitative results as qualitative unless you know otherwise."
                )
        else:
            raise ValueError(f"Unsupported image array shape {raw.shape}.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Image contains NaN/Inf values after decoding.")

        if arr.dtype == np.uint8:
            warnings_out.append("8-bit source -- limited dynamic range; quantitative precision limited.")
        elif arr.dtype != np.uint16:
            arr = np.clip(arr, 0, 65535).astype(np.uint16)

        if ext in (".jpg", ".jpeg"):
            warnings_out.append(
                "JPEG source -- lossy compression and usually gamma-encoded; treat quantitative "
                "results with caution regardless of bit depth."
            )

        # No trustworthy pitch/bit-depth metadata for an arbitrary raster
        # file; sensor_max_raw below is the container's own range, not a
        # verified physical ADC ceiling.
        sensor_saturation_known = False

    else:
        raise ValueError(f"Unsupported file type: {ext or '(no extension)'}")

    if sensor_bit_depth is None:
        sensor_bit_depth = 16 if arr.dtype == np.uint16 else 8
    if sensor_max_raw is None:
        sensor_max_raw = int(np.iinfo(arr.dtype).max)

    h, w = arr.shape
    if max(h, w) > UPLOAD_MAX_DIMENSION_WARN_PX:
        warnings_out.append(
            f"Large image ({w}x{h}) -- analysis performance may be degraded. The image is used "
            f"at full resolution (never silently resized); consider cropping to the beam region."
        )

    return {
        "array": arr,
        "pixel_pitch_um": pixel_pitch_um,
        "sensor_bit_depth": sensor_bit_depth,
        "sensor_max_raw": sensor_max_raw,
        "sensor_saturation_known": sensor_saturation_known,
        "is_rgb_source": is_rgb_source,
        "warnings": warnings_out,
    }


class UploadedImageCameraBackend(CameraBackend):
    """Static, one-shot backend for an image loaded from disk.

    This is NOT a live camera: is_static_source=True, and every other
    capability is disabled (no exposure, no dark capture, no meaningful
    temporal/averaging/peak-hold behavior). All decoding happens up front
    on the GUI thread via detect_upload_metadata(); this class simply
    holds and delivers the already-resolved array and metadata once.
    Re-analysis after a settings change goes through
    MainWindow.reprocess_current_frame(), not another acquisition.
    """

    name = "Uploaded Image"

    supports_exposure = False
    supports_dark_capture = False
    supports_temporal_analysis = False
    supports_frame_averaging = False
    supports_peak_hold = False
    is_static_source = True

    def __init__(self, path, array, pixel_pitch_um, sensor_bit_depth,
                 sensor_max_raw, sensor_saturation_known, warnings=None):
        self.path = path
        self._array = array
        self._pixel_pitch_um = pixel_pitch_um
        self._sensor_bit_depth = int(sensor_bit_depth)
        self._sensor_max_raw = int(sensor_max_raw)
        self._sensor_saturation_known = bool(sensor_saturation_known)
        self.source_warnings = list(warnings or [])
        self._delivered = False

    def open(self):
        pass

    def close(self):
        pass

    def get_frame(self):
        if self._delivered:
            # Should not normally be reached again once is_static_source
            # stops the worker from rescheduling, but stay well-behaved
            # (and non-busy-looping) if it ever is.
            time.sleep(0.05)
        self._delivered = True
        return self._array.copy()

    def get_bit_depth(self):
        return self._sensor_bit_depth

    def get_sensor_max_raw(self):
        return self._sensor_max_raw

    def get_sensor_saturation_known(self):
        return self._sensor_saturation_known

    def get_pixel_pitch_um(self):
        return self._pixel_pitch_um

    def get_detector_size(self):
        h, w = self._array.shape
        return (w, h)

    def get_info_text(self):
        h, w = self._array.shape
        lines = [
            f"Backend: {self.name}",
            f"Source file: {os.path.basename(self.path)}",
            f"{w}x{h}, dtype {self._array.dtype}",
            f"Sensor saturation known: {self._sensor_saturation_known}",
        ]
        lines.extend(f"\u26A0 {msg}" for msg in self.source_warnings)
        return "\n".join(lines)

    @staticmethod
    def list_available_devices():
        return []


CAMERA_BACKEND_NAMES = [
    "Generic Webcam / OpenCV",
    "Thorlabs Scientific / Zelux",
    "IDS uEye / UC480",
    "Demo / Simulated Camera",
    "Uploaded Image",
]

# Single source of truth for every persisted setting: used both to load on
# startup (_load_settings) and to factory-reset (factory_reset_settings).
# Add a key here once and both paths automatically pick it up.
SETTINGS_DEFAULTS = {
    "camera_backend": "Thorlabs Scientific / Zelux", "exposure_ms": 5.0,
    "demo_diameter_um": 500.0,
    "grid_label_size": 50, "pixel_pitch": 3.45, "colormap": "Inferno",
    "opacity": DEFAULT_RAW_DATA_OPACITY_PERCENT, "zoom": 60,
    "overlay_height": OVERLAY_CURVE_HEIGHT, "percent_mode": False,
    "max_display_fps": DEFAULT_MAX_DISPLAY_FPS, "gaussian_r2_min": GAUSSIAN_R2_MIN, "averaging": 1,
    "show_integration_area": False, "calc_area_enabled": False,
    "show_grid": True, "show_gaussian_fit": True,
    "show_centroid_crosshair": True, "show_peak_crosshair": True,
    "calc_area_half_size": 200, "psf_um": 0.0, "wavelength_nm": 780.0,
    "power_cal_factor": 0.0, "power_unit": "µW",
    "dark_frame_enabled": False, "minimum_snr": 10.0,
    "bad_pixel_mask_enabled": False, "hold_maximum_enabled": False,
    "dark_mode": False,
}


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
        if backend_name == "Demo / Simulated Camera":
            return DemoCameraBackend(diameter_um=spec.get("demo_diameter_um", 500.0))
        if backend_name == "Uploaded Image":
            upload = spec.get("upload_data")
            if upload is None:
                raise RuntimeError("No image has been loaded for the Uploaded Image backend.")
            return UploadedImageCameraBackend(
                path=upload["path"], array=upload["array"],
                pixel_pitch_um=upload["pixel_pitch_um"],
                sensor_bit_depth=upload["sensor_bit_depth"],
                sensor_max_raw=upload["sensor_max_raw"],
                sensor_saturation_known=upload["sensor_saturation_known"],
                warnings=upload.get("warnings"),
            )
        raise RuntimeError(f"Unknown camera backend: {backend_name}")

    def _safe_close(self):
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                self._mailbox.note_error()
                self.error.emit(f"Camera close failed: {exc}", traceback.format_exc())

    def _begin_dark_capture(self, count):
        self._dark_frames = []
        self._dark_target = max(2, int(count))
        if self._camera is not None:
            self._camera.set_dark_mode(True)

    def _cancel_dark_capture(self):
        # Centralized so dark mode can never be left stuck on: called on
        # normal completion, on failure, and on every path that tears down
        # or replaces the current camera (connect, disconnect, stop).
        self._dark_frames = []
        self._dark_target = 0
        if self._camera is not None:
            self._camera.set_dark_mode(False)

    @pyqtSlot(object)
    def connect_camera(self, spec):
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        self._safe_close()
        self._mailbox.clear()
        self._cancel_dark_capture()
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
                "sensor_saturation_known": bool(camera.get_sensor_saturation_known()),
                "pixel_pitch_um": camera.get_pixel_pitch_um(),
                "detector_size": camera.get_detector_size(),
                "info_text": camera.get_info_text(),
                "supports_exposure": camera.supports_exposure,
                "supports_dark_capture": camera.supports_dark_capture,
                "supports_temporal_analysis": camera.supports_temporal_analysis,
                "supports_frame_averaging": camera.supports_frame_averaging,
                "supports_peak_hold": camera.supports_peak_hold,
                "is_static_source": camera.is_static_source,
                "source_filename": os.path.basename(getattr(camera, "path", "")) if getattr(camera, "path", None) else None,
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
        self._cancel_dark_capture()
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

    @pyqtSlot(float)
    def set_demo_beam_diameter_um(self, diameter_um):
        if isinstance(self._camera, DemoCameraBackend):
            self._camera.set_beam_diameter_um(float(diameter_um))

    @pyqtSlot(int)
    def request_dark_frame(self, count):
        if self._camera is None:
            self.dark_frame_failed.emit("Connect a camera first.")
            return
        self._begin_dark_capture(count)

    @pyqtSlot()
    def stop(self):
        self._stopping = True
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        self._safe_close()
        self._mailbox.clear()
        self._cancel_dark_capture()
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
                        self._cancel_dark_capture()
        except Exception as exc:
            if self._dark_target:
                self.dark_frame_failed.emit(f"Dark frame capture failed: {exc}")
                self._cancel_dark_capture()
            self._mailbox.note_error()
            self.error.emit(f"Frame acquisition failed: {exc}", traceback.format_exc())
        finally:
            # Static sources (e.g. an uploaded image) get exactly one
            # attempted acquisition -- on success OR failure. On failure
            # this matches live-camera behavior (an error was already
            # emitted above); on success there is nothing more to acquire,
            # since the source represents a single observation, not an
            # ongoing stream. The GUI reflects settings changes for such a
            # source via reprocess_current_frame(), not another acquisition.
            if (
                self._camera is not None
                and not getattr(self._camera, "is_static_source", False)
            ):
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
        self.zoomSpinBox.setValue(60)
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
        self.showGridCheckbox = QCheckBox("Show Grid")
        self.showGridCheckbox.setChecked(True)
        self.showGridCheckbox.setToolTip("Coordinate grid: center lines, tick marks, and µm labels.")
        row2.addWidget(self.showGridCheckbox)
        self.showGaussianFitCheckbox = QCheckBox("Show Gaussian Fit")
        self.showGaussianFitCheckbox.setChecked(True)
        self.showGaussianFitCheckbox.setToolTip("Fitted Gaussian curve overlay on the live image.")
        row2.addWidget(self.showGaussianFitCheckbox)
        self.centroidCrosshairCheckbox = QCheckBox("Centroid Crosshair")
        self.centroidCrosshairCheckbox.setChecked(True)
        row2.addWidget(self.centroidCrosshairCheckbox)
        self.peakCrosshairCheckbox = QCheckBox("Peak Crosshair")
        self.peakCrosshairCheckbox.setChecked(True)
        row2.addWidget(self.peakCrosshairCheckbox)
        row2.addStretch()
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.fpsLabel = QLabel("Measured: -- FPS | -- ms/frame")
        row3.addWidget(self.fpsLabel)
        row3.addSpacing(16)
        self.saturationLabel = QLabel("Saturation: --")
        self.saturationLabel.setToolTip(
            "Brightest pixel as a % of the camera's maximum (A/D full scale).\n"
            "Green 20-90%: good exposure.  Yellow: close to saturating (>90%) or weak signal (<20%).\n"
            "Red \u226595%: saturating, beam-size measurements are unreliable.")
        row3.addWidget(self.saturationLabel)
        row3.addStretch()          # keeps the label left-aligned
        layout.addLayout(row3)

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


# ---------------------------------------------------------------------------
# Shared panel styling helpers (Controls / Advanced Settings).
# The Calculations panel uses the same look: one uniform scrollable
# background, blue section headers, and label-left / control-right rows.
# ---------------------------------------------------------------------------
SECTION_HEADER_STYLE = ("color: #3a7bd5; border-bottom: 1px solid rgba(128,128,128,90);"
                        " padding-top: 6px; padding-bottom: 2px;")
PRIMARY_BUTTON_STYLE = ("QPushButton { background-color: #3a7bd5; color: white; border: 1px solid #5a9bf5;"
                        " border-radius: 4px; padding: 5px; font-weight: bold; }"
                        " QPushButton:hover { background-color: #4a8be5; }"
                        " QPushButton:disabled { background-color: #555555; color: #999999; }")
HINT_LABEL_STYLE = "color: gray;"


def make_scroll_layout(panel):
    """Give `panel` a single scrollable content area with one uniform
    background and return the QVBoxLayout to fill."""
    outer_layout = QVBoxLayout(panel)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    outer_layout.addWidget(scroll)
    content = QWidget()
    scroll.setWidget(content)
    layout = QVBoxLayout(content)
    layout.setSpacing(6)
    return layout


def section_header(title, point_size=12):
    header = QLabel(title)
    font = header.font()
    font.setBold(True)
    font.setPointSize(point_size)
    header.setFont(font)
    header.setStyleSheet(SECTION_HEADER_STYLE)
    return header


def make_form():
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(6)
    return form


def shrink_combos(panel, min_chars=8):
    """Let dropdowns shrink with the panel instead of forcing it as wide as
    their longest item (long names like "Demo / Simulated Camera" would
    otherwise push the right edge of the panel off screen)."""
    for combo in panel.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(min_chars)


def hbox_widget(*widgets, stretch_first=True):
    """Pack widgets side by side into one QWidget (for a form's right column)."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        row.addWidget(w, 1 if (i == 0 and stretch_first) else 0)
    return container


class ControlPanel(QWidget):
    """Controls, in the order you use them: Camera, Acquisition, Display, Graphs."""

    def __init__(self):
        super().__init__()
        layout = make_scroll_layout(self)

        # ---------------- Camera ----------------
        layout.addWidget(section_header("Camera"))
        camera_form = make_form()
        self.cameraBackendLabel = QLabel("Backend:")
        self.cameraBackendCombo = QComboBox()
        self.cameraBackendCombo.addItems(CAMERA_BACKEND_NAMES)
        camera_form.addRow(self.cameraBackendLabel, self.cameraBackendCombo)

        self.deviceLabel = QLabel("Device:")
        self.deviceCombo = QComboBox()
        self.scanCamerasButton = QPushButton("Scan")
        self.scanCamerasButton.setToolTip("Scan for cameras of the selected backend")
        camera_form.addRow(self.deviceLabel, hbox_widget(self.deviceCombo, self.scanCamerasButton))

        self.demoDiameterLabel = QLabel("Simulated 1/e\u00B2 diameter:")
        self.demoDiameterSpinBox = QDoubleSpinBox()
        self.demoDiameterSpinBox.setDecimals(1)
        self.demoDiameterSpinBox.setRange(10.0, 5000.0)
        self.demoDiameterSpinBox.setSingleStep(10.0)
        self.demoDiameterSpinBox.setSuffix(" \u00B5m")
        self.demoDiameterSpinBox.setValue(500.0)
        self.demoDiameterSpinBox.setToolTip(
            "1/e\u00B2 FULL diameter of the ideal simulated TEM00 beam "
            "(Demo / Simulated Camera backend only). Internally the 1/e\u00B2 "
            "radius w = diameter / 2 is used in I = I0*exp(-2r\u00B2/w\u00B2)."
        )
        camera_form.addRow(self.demoDiameterLabel, self.demoDiameterSpinBox)
        layout.addLayout(camera_form)

        self.connectCameraButton = QPushButton("Connect Camera")
        self.connectCameraButton.setStyleSheet(PRIMARY_BUTTON_STYLE)
        layout.addWidget(self.connectCameraButton)

        # ---------------- Acquisition ----------------
        layout.addWidget(section_header("Acquisition"))
        acq_form = make_form()
        self.exposureSlider = QSlider(Qt.Orientation.Horizontal)
        self.exposureSlider.setRange(1, 10000)
        self.exposureSlider.setValue(500)
        self.exposureSpinBox = QDoubleSpinBox()
        self.exposureSpinBox.setDecimals(2)
        self.exposureSpinBox.setRange(0.01, 100.0)
        self.exposureSpinBox.setSingleStep(0.01)
        self.exposureSpinBox.setSuffix(" ms")
        self.exposureSpinBox.setValue(5.0)
        acq_form.addRow(QLabel("Exposure:"), hbox_widget(self.exposureSlider, self.exposureSpinBox))
        layout.addLayout(acq_form)
        self.exposureSlider.valueChanged.connect(self._sync_exposure_from_slider)
        self.exposureSpinBox.valueChanged.connect(self._sync_exposure_from_spinbox)

        # Status line (the main window writes exposure / target display rate here).
        self.sliderLabel = QLabel("Exposure: 5.00 ms")
        self.sliderLabel.setStyleSheet(HINT_LABEL_STYLE)
        self.sliderLabel.setWordWrap(True)
        layout.addWidget(self.sliderLabel)

        self.captureButton = QPushButton("Capture Image")
        layout.addWidget(self.captureButton)

        # ---------------- Display ----------------
        layout.addWidget(section_header("Display"))
        display_form = make_form()
        self.cmapLabel = QLabel("Color map:")
        self.cmapCombo = QComboBox()
        self.cmapCombo.addItems(["Inferno", "Jet", "Hot", "Magma"])
        display_form.addRow(self.cmapLabel, self.cmapCombo)

        self.overlayHeightLabel = QLabel(f"Overlay Curve Height: {OVERLAY_CURVE_HEIGHT}px")
        self.overlayHeightSlider = QSlider(Qt.Orientation.Horizontal)
        self.overlayHeightSlider.setRange(10, 400)
        self.overlayHeightSlider.setValue(OVERLAY_CURVE_HEIGHT)
        display_form.addRow(self.overlayHeightLabel, self.overlayHeightSlider)
        layout.addLayout(display_form)

        # Zoom: the visible control lives in the Camera panel (next to the
        # image). This slider stays as the stored value the rest of the code
        # reads, kept in sync with the Camera panel's zoom box, but hidden
        # so the app doesn't show two zoom controls.
        self.zoomLabel = QLabel("Zoom:")
        self.zoomSlider = QSlider(Qt.Orientation.Horizontal)
        self.zoomSlider.setRange(10, 500)
        self.zoomSlider.setValue(60)
        self.zoomSpinBox = QSpinBox()
        self.zoomSpinBox.setRange(10, 500)
        self.zoomSpinBox.setValue(60)
        self.zoomSpinBox.setSuffix("%")
        self.zoomSlider.valueChanged.connect(self.zoomSpinBox.setValue)
        self.zoomSpinBox.valueChanged.connect(self.zoomSlider.setValue)
        zoom_container = hbox_widget(self.zoomLabel, self.zoomSlider, self.zoomSpinBox, stretch_first=False)
        zoom_container.setVisible(False)
        layout.addWidget(zoom_container)

        # ---------------- Graphs ----------------
        layout.addWidget(section_header("Graphs"))
        graph_form = make_form()
        self.opacityLabel = QLabel(f"Raw Data Opacity: {DEFAULT_RAW_DATA_OPACITY_PERCENT}%")
        self.opacitySlider = QSlider(Qt.Orientation.Horizontal)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setValue(DEFAULT_RAW_DATA_OPACITY_PERCENT)
        graph_form.addRow(self.opacityLabel, self.opacitySlider)
        layout.addLayout(graph_form)

        self.percentModeCheckbox = QCheckBox("Show graphs as % of full scale")
        self.percentModeCheckbox.setToolTip("Scales the graphs using the camera's measured bit depth.")
        layout.addWidget(self.percentModeCheckbox)
        layout.addStretch()
        shrink_combos(self)

    def _sync_exposure_from_slider(self, value):
        self.exposureSpinBox.setValue(value / 100.0)

    def _sync_exposure_from_spinbox(self, ms):
        self.exposureSlider.setValue(int(round(ms * 100)))


class AnalyzePanel(QWidget):
    """Calculations panel, grouped into sections:
    Beam Size (table of the four standard methods, X and Y side by side),
    Shape, Position, Signal Quality, Divergence, Setup."""

    # (key, row label, tooltip) in order of importance.
    SIZE_METHODS = (
        ("d4sigma", "D4σ (ISO 11146)",
         "Second-moment diameter, the ISO 11146 standard. Valid for any beam shape."),
        ("e2", "1/e² threshold",
         "Width where the profile falls to 1/e² (13.5%) of its peak."),
        ("gauss", "Gaussian fit",
         "1/e² diameter of a Gaussian fitted to the whole profile.\n"
         "R² shows how Gaussian the beam is; if it is low, trust D4σ."),
        ("fwhm", "FWHM",
         "Full width at half maximum: width where the profile falls to 50% of its peak."),
    )

    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(4)

        def bold(label, size=11):
            font = label.font()
            font.setBold(True)
            font.setPointSize(size)
            label.setFont(font)
            return label

        def section(title):
            header = bold(QLabel(title), 12)
            header.setStyleSheet("color: #3a7bd5; border-bottom: 1px solid rgba(128,128,128,90);"
                                 " padding-top: 6px; padding-bottom: 2px;")
            layout.addWidget(header)

        def full(text):
            label = bold(QLabel(text))
            label.setWordWrap(True)
            layout.addWidget(label)
            return label

        # ---------------- Beam Size ----------------
        section("Beam Size")
        size_grid = QGridLayout()
        size_grid.setHorizontalSpacing(16)
        size_grid.setVerticalSpacing(4)
        for col, text in enumerate(("", "X", "Y")):
            header = bold(QLabel(text))
            header.setStyleSheet("color: gray;")
            size_grid.addWidget(header, 0, col)
        self.size_labels = {}
        for row, (key, name, tip) in enumerate(self.SIZE_METHODS, start=1):
            name_label = bold(QLabel(name))
            name_label.setToolTip(tip)
            x_label = bold(QLabel("N/A"))
            y_label = bold(QLabel("N/A"))
            size_grid.addWidget(name_label, row, 0)
            size_grid.addWidget(x_label, row, 1)
            size_grid.addWidget(y_label, row, 2)
            self.size_labels[key] = (x_label, y_label)
        size_grid.setColumnStretch(1, 1)
        size_grid.setColumnStretch(2, 1)
        layout.addLayout(size_grid)
        # Same attribute names as before, so the update code can find them.
        self.d4sigma_x_label, self.d4sigma_y_label = self.size_labels["d4sigma"]
        self.e2_x_label, self.e2_y_label = self.size_labels["e2"]
        self.gauss_diam_x_label, self.gauss_diam_y_label = self.size_labels["gauss"]
        self.fwhm_x_label, self.fwhm_y_label = self.size_labels["fwhm"]

        # ---------------- Shape ----------------
        section("Shape")
        self.d4sigma_principal_label = full("Principal D4σ (major/minor, angle): N/A")
        self.d4sigma_ellipticity_label = full("Ellipticity (minor/major D4σ): N/A")

        # ---------------- Position ----------------
        section("Position")
        self.centroid_position_label = full("Centroid: N/A")
        self.peak_position_label = full("Peak: N/A")

        # ---------------- Signal Quality ----------------
        section("Signal Quality")
        self.saturation_label = full("A/D Saturation: N/A")
        self.quality_label = full("SNR: N/A")
        self.background_label = full("Background: N/A")

        # ---------------- Divergence ----------------
        section("Divergence")
        self.treatAsWaistCheckbox = QCheckBox("Treat current plane as beam waist")
        self.treatAsWaistCheckbox.setToolTip("Required for the divergence estimate: only valid if the camera "
                                             "is at the beam's narrowest point (waist).")
        self.treatAsWaistCheckbox.setChecked(False)
        layout.addWidget(self.treatAsWaistCheckbox)
        self.divergence_x_label = full("X Divergence: enable checkbox")
        self.divergence_y_label = full("Y Divergence: enable checkbox")
        for lbl in (self.divergence_x_label, self.divergence_y_label):
            lbl.setStyleSheet("background-color: rgba(58,123,213,40); border-radius: 4px; padding: 3px;")

        # ---------------- Setup ----------------
        section("Setup")
        pitch_layout = QHBoxLayout()
        pitch_label = QLabel("Pixel Pitch (µm):")
        self.pixel_pitch_input = QDoubleSpinBox()
        self.pixel_pitch_input.setDecimals(2)
        self.pixel_pitch_input.setRange(0.1, 100.0)
        self.pixel_pitch_input.setValue(3.45)
        self.pixel_pitch_input.setSingleStep(0.1)
        pitch_layout.addWidget(pitch_label)
        pitch_layout.addWidget(self.pixel_pitch_input)
        layout.addLayout(pitch_layout)
        self.power_label = full("Total Power: Not Calibrated")

        self.logButton = QPushButton("Start Logging")
        self.logButton.setCheckable(True)
        layout.addWidget(self.logButton)
        self.logStatusLabel = QLabel("Logging: Off")
        layout.addWidget(self.logStatusLabel)
        layout.addStretch()


class AdvancedSettingsPanel(QWidget):
    """Advanced settings grouped as: Analysis, Dark Frame, Calculation Area,
    Acquisition, Calibration, with Reset to Defaults at the bottom."""

    def __init__(self):
        super().__init__()
        layout = make_scroll_layout(self)

        # ---------------- Analysis ----------------
        layout.addWidget(section_header("Analysis"))
        self.enableAdvancedMetricsCheckbox = QCheckBox("Enable D4\u03c3 / background / hot-pixel analysis")
        self.enableAdvancedMetricsCheckbox.setChecked(True)
        layout.addWidget(self.enableAdvancedMetricsCheckbox)

        analysis_form = make_form()
        self.minSnrSpinBox = QDoubleSpinBox()
        self.minSnrSpinBox.setRange(0.0, 10000.0)
        self.minSnrSpinBox.setDecimals(1)
        self.minSnrSpinBox.setValue(10.0)
        analysis_form.addRow(QLabel("Minimum peak SNR:"), self.minSnrSpinBox)

        self.gaussianR2SpinBox = QDoubleSpinBox()
        self.gaussianR2SpinBox.setRange(0.5, 0.999)
        self.gaussianR2SpinBox.setDecimals(3)
        self.gaussianR2SpinBox.setSingleStep(0.001)
        self.gaussianR2SpinBox.setValue(GAUSSIAN_R2_MIN)
        analysis_form.addRow(QLabel("Gaussian fit threshold (R\u00B2):"), self.gaussianR2SpinBox)

        self.psfSpinBox = QDoubleSpinBox()
        self.psfSpinBox.setRange(0.0, 1000.0)
        self.psfSpinBox.setDecimals(2)
        self.psfSpinBox.setValue(0.0)
        self.psfSpinBox.setSuffix(" \u00B5m")
        analysis_form.addRow(QLabel("Instrument PSF:"), self.psfSpinBox)
        layout.addLayout(analysis_form)

        self.badPixelMaskCheckbox = QCheckBox("Mask isolated hot pixels for analysis")
        self.badPixelMaskCheckbox.setChecked(False)
        layout.addWidget(self.badPixelMaskCheckbox)

        self.showIntegrationAreaCheckbox = QCheckBox("Show D4\u03c3 integration area")
        self.showIntegrationAreaCheckbox.setChecked(False)
        self.showIntegrationAreaCheckbox.setToolTip(
            "Draw the adaptive D4\u03c3 integration rectangle on the live overlay. "
            "A box spanning (nearly) the whole frame is a sign the D4\u03c3 background "
            "estimate should be investigated rather than trusting the number as-is."
        )
        layout.addWidget(self.showIntegrationAreaCheckbox)

        # ---------------- Dark Frame ----------------
        layout.addWidget(section_header("Dark Frame"))
        self.captureDarkButton = QPushButton("Capture Dark (8 frames)")
        self.darkEnableCheckbox = QCheckBox("Apply dark subtraction")
        self.darkEnableCheckbox.setChecked(False)
        layout.addWidget(hbox_widget(self.captureDarkButton, self.darkEnableCheckbox, stretch_first=False))
        self.darkStatusLabel = QLabel("Dark frame: not captured")
        self.darkStatusLabel.setStyleSheet(HINT_LABEL_STYLE)
        self.darkStatusLabel.setWordWrap(True)
        layout.addWidget(self.darkStatusLabel)

        # ---------------- Calculation Area ----------------
        layout.addWidget(section_header("Calculation Area"))
        self.calcAreaCheckbox = QCheckBox("Restrict calculation area")
        layout.addWidget(self.calcAreaCheckbox)
        calc_form = make_form()
        self.calcAreaSpinBox = QSpinBox()
        self.calcAreaSpinBox.setRange(20, 2000)
        self.calcAreaSpinBox.setValue(200)
        self.calcAreaSpinBox.setSuffix(" px")
        calc_form.addRow(QLabel("Half-size:"), self.calcAreaSpinBox)
        layout.addLayout(calc_form)
        # Half-size only matters when the area is restricted.
        self.calcAreaSpinBox.setEnabled(False)
        self.calcAreaCheckbox.toggled.connect(self.calcAreaSpinBox.setEnabled)

        # ---------------- Acquisition ----------------
        layout.addWidget(section_header("Acquisition"))
        acq_form = make_form()
        self.maxFpsSpinBox = QSpinBox()
        self.maxFpsSpinBox.setRange(1, 60)
        self.maxFpsSpinBox.setValue(DEFAULT_MAX_DISPLAY_FPS)
        self.maxFpsSpinBox.setSuffix(" FPS")
        acq_form.addRow(QLabel("Max display rate:"), self.maxFpsSpinBox)

        self.averagingSpinBox = QSpinBox()
        self.averagingSpinBox.setRange(1, 20)
        self.averagingSpinBox.setValue(1)
        acq_form.addRow(QLabel("Averaged frames:"), self.averagingSpinBox)
        layout.addLayout(acq_form)

        self.holdMaxCheckbox = QCheckBox("Hold maximum (peak hold)")
        self.resetHoldButton = QPushButton("Reset Hold")
        layout.addWidget(hbox_widget(self.holdMaxCheckbox, self.resetHoldButton))

        # ---------------- Calibration ----------------
        layout.addWidget(section_header("Calibration"))
        cal_form = make_form()
        self.wavelengthSpinBox = QDoubleSpinBox()
        self.wavelengthSpinBox.setRange(100.0, 3000.0)
        self.wavelengthSpinBox.setValue(780.0)
        self.wavelengthSpinBox.setSuffix(" nm")
        cal_form.addRow(QLabel("Wavelength:"), self.wavelengthSpinBox)

        self.powerCalSpinBox = QDoubleSpinBox()
        self.powerCalSpinBox.setRange(0.0, 1000.0)
        self.powerCalSpinBox.setDecimals(6)
        self.powerCalSpinBox.setValue(0.0)
        self.powerCalSpinBox.setToolTip("Manual calibration: microwatts per background-subtracted count. "
                                        "Leave at 0 if not calibrated.")
        cal_form.addRow(QLabel("Power factor (\u00B5W/count):"), self.powerCalSpinBox)

        self.powerUnitCombo = QComboBox()
        self.powerUnitCombo.addItems(["\u00B5W", "mW", "dBm"])
        cal_form.addRow(QLabel("Show power in:"), self.powerUnitCombo)
        layout.addLayout(cal_form)

        # ---------------- Reset (last: destructive) ----------------
        layout.addSpacing(10)
        self.resetDefaultsButton = QPushButton("Reset to Defaults")
        layout.addWidget(self.resetDefaultsButton)
        layout.addStretch()
        shrink_combos(self)


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
        self._last_fit_x = None
        self._last_fit_y = None
        self._last_fit_inputs = {}
        self._bootstrap_results = {}
        self._current_backend_name = None
        self._current_pixel_pitch_um = None
        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        notice = QLabel(
            "Log measurements at several Z positions, then read the coverage status, results, "
            "and diagnostics on the right."
        )
        notice.setWordWrap(True)
        top_row.addWidget(notice, 4)
        self.detailsButton = QPushButton("ⓘ How to use / caveats")
        self.detailsButton.clicked.connect(self._show_details_dialog)
        top_row.addWidget(self.detailsButton, 1)
        root.addLayout(top_row)

        self.calibrationWarningLabel = QLabel("")
        self.calibrationWarningLabel.setWordWrap(True)
        self.calibrationWarningLabel.setStyleSheet("background:#fff3cd;color:#5f4700;padding:5px;border:1px solid #d8bd68;")
        self.calibrationWarningLabel.setVisible(False)
        root.addWidget(self.calibrationWarningLabel)

        config = QGroupBox("Measurement configuration")
        configLayout = QHBoxLayout(config)
        configLayout.addWidget(QLabel("Z position:"))
        self.zSpin = QDoubleSpinBox()
        self.zSpin.setRange(-1_000_000.0, 1_000_000.0)
        self.zSpin.setDecimals(4)
        self.zSpin.setSuffix(" mm")
        configLayout.addWidget(self.zSpin)

        configLayout.addWidget(QLabel("Wavelength:"))
        self.wavelengthSpin = QDoubleSpinBox()
        self.wavelengthSpin.setRange(100.0, 3000.0)
        self.wavelengthSpin.setDecimals(2)
        self.wavelengthSpin.setSuffix(" nm")
        self.wavelengthSpin.setValue(main_window.advanced_panel.wavelengthSpinBox.value())
        self.wavelengthSpin.valueChanged.connect(self.refit)
        configLayout.addWidget(self.wavelengthSpin)
        self.wavelengthLockLabel = QLabel("")
        self.wavelengthLockLabel.setToolTip(
            "Wavelength is locked once measurements exist, so existing points are never "
            "silently reinterpreted under a different wavelength. Use Clear All to change it."
        )
        configLayout.addWidget(self.wavelengthLockLabel)

        configLayout.addWidget(QLabel("Fit width method:"))
        self.methodCombo = QComboBox()
        self.methodCombo.addItems([
            "D4σ camera axes (2D second moment)",
            "Gaussian-fit 1/e² diameters (non-ISO comparison)",
        ])
        self.methodCombo.currentIndexChanged.connect(self.refit)
        configLayout.addWidget(self.methodCombo)
        configLayout.addStretch(1)
        root.addWidget(config)

        advancedConfig = QGroupBox("Advanced configuration (stage convention, focal length) — click to expand")
        advancedConfig.setCheckable(True)
        advancedConfig.setChecked(False)
        advancedOuter = QVBoxLayout(advancedConfig)
        advancedContent = QWidget()
        advancedForm = QFormLayout(advancedContent)
        self.directionCombo = QComboBox()
        self.directionCombo.addItems(["Increasing Z is downstream", "Increasing Z is upstream"])
        advancedForm.addRow("Stage convention (label only — does not change the fit):", self.directionCombo)
        self.lensEdit = QDoubleSpinBox()
        self.lensEdit.setRange(0.0, 100000.0)
        self.lensEdit.setDecimals(2)
        self.lensEdit.setSuffix(" mm")
        self.lensEdit.setSpecialValueText("Not recorded")
        advancedForm.addRow("Focusing-lens focal length (reference only — M² is invariant\n"
                             "through ideal optics, so this is not applied to the fit):", self.lensEdit)
        advancedOuter.addWidget(advancedContent)
        advancedContent.setVisible(False)
        advancedConfig.toggled.connect(advancedContent.setVisible)
        root.addWidget(advancedConfig)

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
        self.table.setToolTip(
            "Rows sharing a highlighted tint were logged at (approximately) the same Z and are "
            "aggregated into one plane for fitting. The currently inactive width method's "
            "columns are shown in italics."
        )
        left.addWidget(self.table)

        export_row = QHBoxLayout()
        self.saveSessionButton = QPushButton("Save Session JSON")
        self.loadSessionButton = QPushButton("Load Session JSON")
        self.exportCsvButton = QPushButton("Export Raw CSV")
        self.exportAggregatedCsvButton = QPushButton("Export Aggregated CSV")
        self.saveSessionButton.clicked.connect(self.save_session)
        self.loadSessionButton.clicked.connect(self.load_session)
        self.exportCsvButton.clicked.connect(self.export_csv)
        self.exportAggregatedCsvButton.clicked.connect(self.export_aggregated_csv)
        export_row.addWidget(self.saveSessionButton)
        export_row.addWidget(self.loadSessionButton)
        export_row.addWidget(self.exportCsvButton)
        export_row.addWidget(self.exportAggregatedCsvButton)
        left.addLayout(export_row)

        analysis_row = QHBoxLayout()
        self.finalizeButton = QPushButton("Finalize Analysis (bootstrap 95% CI)")
        self.finalizeButton.setToolTip(
            "Runs an on-demand parametric bootstrap for a more robust confidence interval than "
            "the live linearized ±. Not run automatically on every edit -- only when requested."
        )
        self.finalizeButton.clicked.connect(self.finalize_analysis)
        analysis_row.addWidget(self.finalizeButton)
        left.addLayout(analysis_row)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        self.statusLabel = QLabel(self._overall_status_html(None, None))
        self.statusLabel.setWordWrap(True)
        right.addWidget(self.statusLabel)

        self.figure = Figure(figsize=(7, 6), tight_layout=True)
        self.main_canvas = FigureCanvas(self.figure)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1])
        self.ax = self.figure.add_subplot(gs[0])
        self.zone_ax = self.figure.add_subplot(gs[1])
        right.addWidget(self.main_canvas, 6)

        cards_row = QHBoxLayout()
        self.xCard = QLabel("")
        self.yCard = QLabel("")
        for card in (self.xCard, self.yCard):
            card.setWordWrap(True)
            card.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            card.setStyleSheet("border:1px solid #7fa0c0;border-radius:6px;padding:8px;")
            card.setAlignment(Qt.AlignmentFlag.AlignTop)
        cards_row.addWidget(self.xCard)
        cards_row.addWidget(self.yCard)
        right.addLayout(cards_row, 3)

        self.diagnosticsGroup = QGroupBox("Fit Diagnostics (residuals, correlation, boundary checks) — click to expand")
        self.diagnosticsGroup.setCheckable(True)
        self.diagnosticsGroup.setChecked(False)
        diagOuter = QVBoxLayout(self.diagnosticsGroup)
        self.diagnosticsContent = QWidget()
        diagContentLayout = QVBoxLayout(self.diagnosticsContent)
        self.diag_figure = Figure(figsize=(6, 2.5), tight_layout=True)
        self.diag_canvas = FigureCanvas(self.diag_figure)
        self.resid_ax = self.diag_figure.add_subplot(111)
        diagContentLayout.addWidget(self.diag_canvas)
        self.diagnosticsLabel = QLabel("")
        self.diagnosticsLabel.setWordWrap(True)
        self.diagnosticsLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        diagContentLayout.addWidget(self.diagnosticsLabel)
        diagOuter.addWidget(self.diagnosticsContent)
        self.diagnosticsContent.setVisible(False)
        self.diagnosticsGroup.toggled.connect(self.diagnosticsContent.setVisible)
        right.addWidget(self.diagnosticsGroup, 3)

        body.addLayout(right, 4)
        root.addLayout(body)
        self._sync_wavelength_lock()

    # -- details / scope dialog -------------------------------------------------

    def _show_details_dialog(self):
        QMessageBox.information(
            self, "How to use this / caveats",
            "<b>How to use this:</b> 1) Move the camera (or the beam/focusing lens) to a new "
            "Z position along the beam path and enter that position above. 2) Wait for a "
            "steady, unsaturated beam. 3) Click “Log Stable Current Measurement” -- it "
            "needs 5 consistent live frames within the last 2 seconds, so it will tell you to "
            "wait if the beam just moved or is still fluctuating. 4) Repeat at several "
            "different Z positions -- at least 4 for a provisional fit, roughly half near the "
            "waist and half beyond twice the Rayleigh range for real coverage.<br><br>"
            "<b>Caveats:</b> This tool fits a beam caustic from multiple planes. It is not "
            "automatically ISO 11146 compliant. Use one consistent width method, adequate "
            "axial coverage, background correction, no clipping/saturation, and a known "
            "wavelength. A good-looking curve fit does not by itself prove ISO compliance or "
            "remove background/sampling uncertainty.<br><br>"
            f"<b>Scope:</b> {PROPAGATION_SCOPE_NOTE}"
        )

    # -- environment / session-consistency hooks --------------------------------

    def note_environment_change(self, pixel_pitch_um=None, backend_name=None):
        """Called by MainWindow when the camera backend (re)connects. Stores
        the latest environment for reference; the actual mixed-calibration
        warning is raised at log time in log_current(), compared against the
        first sample already logged this session (pixel pitch is already
        stored per-sample, so old points don't need to be reinterpreted --
        they just get flagged as inconsistent with newer ones)."""
        self._current_backend_name = backend_name
        self._current_pixel_pitch_um = pixel_pitch_um

    def _sync_wavelength_lock(self):
        locked = len(self.samples) > 0
        self.wavelengthSpin.setEnabled(not locked)
        self.wavelengthLockLabel.setText("\U0001f512 locked (Clear All to change)" if locked else "")

    # -- logging / table ----------------------------------------------------

    def log_current(self):
        z = float(self.zSpin.value())
        if any(abs((s.z_mm or 0.0) - z) < 1e-9 for s in self.samples):
            answer = QMessageBox.question(
                self, "Duplicate Z position",
                "A point already exists at this Z. Add a replicate anyway? "
                "(Replicates are aggregated into one plane for fitting, not treated as "
                "additional independent Z coverage.)",
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
        if (self.samples and sample.pixel_pitch_um and self.samples[0].pixel_pitch_um
                and abs(sample.pixel_pitch_um - self.samples[0].pixel_pitch_um) > 1e-6):
            self.calibrationWarningLabel.setText(
                "⚠ Mixed calibration session: this point's pixel pitch "
                f"({sample.pixel_pitch_um:.3f} µm/px) differs from the first logged point "
                f"({self.samples[0].pixel_pitch_um:.3f} µm/px). Widths are already converted "
                "to µm at capture time, so old points are not reinterpreted -- but mixing "
                "calibrations in one fit is not recommended. Consider starting a new session "
                "(Clear All)."
            )
            self.calibrationWarningLabel.setVisible(True)
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
        self.calibrationWarningLabel.setVisible(False)
        self._bootstrap_results = {}
        self.refit()

    def _checked_samples(self):
        checked = []
        for row, sample in enumerate(self.samples):
            item = self.table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                checked.append(sample)
        return checked

    def _selected_data(self):
        use_d4 = self.methodCombo.currentIndex() == 0
        points = aggregate_replicates(self._checked_samples())
        rows = []
        for p in points:
            x = p.d4sigma_x_um if use_d4 else p.gaussian_x_um
            y = p.d4sigma_y_um if use_d4 else p.gaussian_y_um
            x_spread = p.d4sigma_x_spread_um if use_d4 else p.gaussian_x_spread_um
            y_spread = p.d4sigma_y_spread_um if use_d4 else p.gaussian_y_spread_um
            if x is not None and y is not None and x > 0 and y > 0:
                rows.append((p.z_mm, x, y, x_spread, y_spread))
        return rows, points

    def _recolor_table(self):
        use_d4 = self.methodCombo.currentIndex() == 0
        active_cols = (2, 3) if use_d4 else (4, 5)
        inactive_cols = (4, 5) if use_d4 else (2, 3)
        for row, sample in enumerate(self.samples):
            ok, _ = passes_quality_gate(sample)
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                font = item.font()
                font.setItalic(col in inactive_cols)
                font.setBold(col in active_cols)
                item.setFont(font)
            z_item = self.table.item(row, 1)
            if z_item is not None:
                z_item.setBackground(QColor(255, 90, 90, 60) if not ok else QColor(0, 0, 0, 0))

        # Replicate/plane grouping tint on the "Use" column, matching
        # aggregate_replicates()'s own z-tolerance grouping.
        order = sorted(
            (i for i, s in enumerate(self.samples) if s.z_mm is not None),
            key=lambda i: self.samples[i].z_mm,
        )
        group_id = -1
        prev_z = None
        group_of = {}
        for i in order:
            z = self.samples[i].z_mm
            if prev_z is None or abs(z - prev_z) > 0.01:
                group_id += 1
            group_of[i] = group_id
            prev_z = z
        for row in range(len(self.samples)):
            gid = group_of.get(row)
            item = self.table.item(row, 0)
            if item is None or gid is None:
                continue
            item.setBackground(QColor(120, 170, 220, 35) if gid % 2 == 0 else QColor(0, 0, 0, 0))

    # -- coverage bar ---------------------------------------------------------

    def _draw_zone_bar(self, ax, fit, y_pos, z_all):
        if fit is None or fit.get("zones") is None:
            return
        z0, zr = fit["z0_mm"], fit["z_r_mm"]
        if zr <= 0:
            return
        zmin, zmax = float(np.min(z_all)), float(np.max(z_all))
        lo = min(zmin, z0 - 2.5 * zr)
        hi = max(zmax, z0 + 2.5 * zr)
        bands = [
            (lo, z0 - 2 * zr, "tab:red"),
            (z0 - 2 * zr, z0 - zr, "gold"),
            (z0 - zr, z0 + zr, "tab:green"),
            (z0 + zr, z0 + 2 * zr, "gold"),
            (z0 + 2 * zr, hi, "tab:red"),
        ]
        for start, end, color in bands:
            if end > start:
                ax.broken_barh([(start, end - start)], (y_pos - 0.35, 0.7), facecolors=color, alpha=0.30)
        ax.scatter(z_all, np.full_like(z_all, y_pos, dtype=float), marker="|", color="black", s=200, zorder=5)

    # -- status / card text ---------------------------------------------------

    def _overall_status_html(self, fit_x, fit_y):
        fits = [f for f in (fit_x, fit_y) if f is not None]
        if not fits:
            return ('<span style="color:#b36b00;"><b>⚠ Add at least four distinct Z planes '
                    '(after replicate aggregation) for a provisional fit.</b></span>')
        all_ok = all(
            f["model_constraints_satisfied"]
            and coverage_summary(f["zones"], f["unique_planes"])["iso_style_target_met"]
            and describe_identifiability(f).startswith("Parameters appear")
            for f in fits
        )
        if all_ok:
            return '<span style="color:#2e7d32;"><b>✓ Coverage target met; fit checks passed on both axes.</b></span>'
        return '<span style="color:#b36b00;"><b>⚠ Provisional — see per-axis details below for what is missing.</b></span>'

    def _card_html(self, name, fit):
        if fit is None:
            return f"<b>{name}</b>: fit failed or insufficient independent data."
        use_d4 = self.methodCombo.currentIndex() == 0
        mode_header = (f"<b>{name} — D4σ / M² propagation analysis</b>" if use_d4 else
                       f'<b><span style="color:#8a5300;">{name} — Gaussian-derived M² estimate '
                       f'(non-ISO comparison)</span></b>')
        summary = coverage_summary(fit["zones"], fit["unique_planes"])
        ident_text = describe_identifiability(fit)
        cov_text = describe_coverage(summary)
        status_ok = (fit["model_constraints_satisfied"] and summary["iso_style_target_met"]
                     and ident_text.startswith("Parameters appear"))
        status_html = ('<span style="color:#2e7d32;">✓ Fit checks passed</span>' if status_ok else
                       '<span style="color:#b36b00;">⚠ Provisional</span>')
        weighted_note = (" (weighted by replicate/frame repeatability -- relative weighting only, "
                          "not a calibrated absolute uncertainty)" if fit["weighted"] else " (unweighted)")
        lines = [
            mode_header,
            status_html,
            f"M² = {fit['m2']:.3f} ± {fit['m2_error']:.3f}{weighted_note}",
            f"Waist diameter = {2 * fit['w0_um'] / 1000:.4f} mm  (w₀ = {fit['w0_um']:.1f} µm)",
            f"Waist position z₀ = {fit['z0_mm']:.4g} ± {fit['z0_error_mm']:.3g} mm",
            f"Full divergence = {fit['theta_full_mrad']:.4g} mrad (half = {fit['theta_half_mrad']:.4g} mrad)",
            f"zR = {fit['z_r_mm']:.4g} mm",
            cov_text,
            ident_text,
            f"Fit constraints satisfied: {fit['model_constraints_satisfied']}",
        ]
        boot = self._bootstrap_results.get(name)
        if boot and boot.get("n_successful", 0) > 0:
            m2_lo, m2_hi = boot["m2_ci95"]
            lines.append(
                f"Bootstrap 95% CI (n={boot['n_successful']}/{boot['n_requested']}, parametric): "
                f"M² ∈ [{m2_lo:.2f}, {m2_hi:.2f}]"
            )
        elif boot is not None:
            lines.append("Bootstrap CI: did not converge on enough resamples.")
        return "<br>".join(lines)

    def _diagnostics_html(self, name, fit):
        if fit is None:
            return f"<b>{name}</b>: n/a"
        ident = fit["identifiability"]
        corr_text = "n/a (covariance not available)"
        if ident.get("correlation") is not None:
            c = ident["correlation"]
            corr_text = f"corr(w₀,z₀)={c[0,1]:.2f}, corr(w₀,M²)={c[0,2]:.2f}, corr(z₀,M²)={c[1,2]:.2f}"
        bounds_text = ", ".join(f"{n}@{side}" for n, side in fit["bounds_hit"]) or "none"
        weighted_rmse_text = (f", weighted RMSE={fit['weighted_rmse_um_radius']:.3g} µm (radius)"
                               if fit["weighted_rmse_um_radius"] is not None else "")
        return (f"<b>{name}</b>: RMSE={fit['rmse_um_radius']:.3g} µm (radius); "
                f"NRMSE={100*fit['nrmse']:.3g}% (descriptive only -- not a validity signal)"
                f"{weighted_rmse_text}<br>Parameter correlation: {corr_text}<br>Boundary hits: {bounds_text}")

    def _render_results(self):
        self.statusLabel.setText(self._overall_status_html(self._last_fit_x, self._last_fit_y))
        self.xCard.setText(self._card_html("X", self._last_fit_x))
        self.yCard.setText(self._card_html("Y", self._last_fit_y))
        diag_lines = [self._diagnostics_html("X", self._last_fit_x), self._diagnostics_html("Y", self._last_fit_y)]
        diag_lines.append(f"<i>{PROPAGATION_SCOPE_NOTE}</i>")
        self.diagnosticsLabel.setText("<br><br>".join(diag_lines))

    # -- fitting --------------------------------------------------------------

    def refit(self, *args):
        self._recolor_table()
        self._sync_wavelength_lock()
        rows, points = self._selected_data()
        self.ax.clear()
        self.zone_ax.clear()
        self.resid_ax.clear()
        self.ax.set_ylabel("Full beam diameter (µm)")
        self.zone_ax.set_xlabel("Z position (mm)")
        self.resid_ax.set_xlabel("Z position (mm)")
        self.resid_ax.set_ylabel("Radius residual (µm)")

        if len(rows) < 4:
            self._last_fit_x = self._last_fit_y = None
            self._last_fit_inputs = {}
            self.statusLabel.setText(self._overall_status_html(None, None))
            self.xCard.setText(
                f"Accepted: {len(rows)} unique plane(s) after replicate aggregation "
                f"({len(self._checked_samples())} raw row(s) checked)."
            )
            self.yCard.setText("")
            self.diagnosticsLabel.setText("")
            if rows:
                z = np.array([r[0] for r in rows])
                dx = np.array([r[1] for r in rows])
                dy = np.array([r[2] for r in rows])
                self.ax.scatter(z, dx, marker="o", label="X")
                self.ax.scatter(z, dy, marker="s", label="Y")
                self.ax.legend()
            self.main_canvas.draw_idle()
            self.diag_canvas.draw_idle()
            return

        z = np.array([r[0] for r in rows], dtype=float)
        dx = np.array([r[1] for r in rows], dtype=float)
        dy = np.array([r[2] for r in rows], dtype=float)
        dx_spread = np.array([r[3] if r[3] is not None else np.nan for r in rows], dtype=float)
        dy_spread = np.array([r[4] if r[4] is not None else np.nan for r in rows], dtype=float)
        wavelength = float(self.wavelengthSpin.value())

        fit_x = fit_propagation_axis(z, dx, wavelength, spread_um=dx_spread)
        fit_y = fit_propagation_axis(z, dy, wavelength, spread_um=dy_spread)

        # Raw (pre-aggregation) points shown lightly for context; the
        # aggregated points actually being fitted are shown bold on top.
        use_d4 = self.methodCombo.currentIndex() == 0
        raw_checked = [s for s in self._checked_samples() if s.z_mm is not None]
        raw_z = np.array([s.z_mm for s in raw_checked], dtype=float)
        raw_dx = np.array([(s.d4sigma_x_um if use_d4 else s.gaussian_x_um) or np.nan for s in raw_checked], dtype=float)
        raw_dy = np.array([(s.d4sigma_y_um if use_d4 else s.gaussian_y_um) or np.nan for s in raw_checked], dtype=float)
        if raw_z.size:
            self.ax.scatter(raw_z, raw_dx, marker=".", color="C0", alpha=0.35, s=25)
            self.ax.scatter(raw_z, raw_dy, marker=".", color="C1", alpha=0.35, s=25)
        self.ax.scatter(z, dx, marker="o", color="C0", label="X (aggregated)")
        self.ax.scatter(z, dy, marker="s", color="C1", label="Y (aggregated)")

        self._last_fit_inputs = {}
        for name, fit, color in (("X", fit_x, "C0"), ("Y", fit_y, "C1")):
            if fit is None:
                continue
            z_grid = np.linspace(float(z.min()), float(z.max()), 500)
            radius_grid = propagation_radius_model(z_grid, fit["w0_um"], fit["z0_mm"], fit["m2"], wavelength)
            self.ax.plot(z_grid, 2.0 * radius_grid, color=color, label=f"{name} fit")
            self.resid_ax.scatter(fit["z"], fit["observed_radius_um"] - fit["predicted_radius_um"],
                                   color=color, label=name)
            self._last_fit_inputs[name] = {
                "z": fit["z"], "w": fit["observed_radius_um"], "wavelength": wavelength,
                "sigma": fit["_sigma"], "lower": fit["_lower_bounds"], "upper": fit["_upper_bounds"],
            }

        self.ax.legend(fontsize=8)
        self.ax.grid(True, alpha=0.25)

        self._draw_zone_bar(self.zone_ax, fit_x, 1.0, z)
        self._draw_zone_bar(self.zone_ax, fit_y, 0.0, z)
        self.zone_ax.set_yticks([0.0, 1.0])
        self.zone_ax.set_yticklabels(["Y", "X"])
        self.zone_ax.set_ylim(-0.6, 1.6)
        self.zone_ax.set_title("Coverage: green = near-waist, gold = transition, red = far-field target", fontsize=8)

        self.resid_ax.axhline(0.0, linewidth=1)
        self.resid_ax.legend(fontsize=8)
        self.resid_ax.grid(True, alpha=0.25)

        self._last_fit_x, self._last_fit_y = fit_x, fit_y
        self._bootstrap_results = {}  # invalidate any cached CI now that the fit input changed
        self._render_results()

        self.main_canvas.draw_idle()
        self.diag_canvas.draw_idle()

    # -- finalize / bootstrap --------------------------------------------------

    def finalize_analysis(self):
        if not self._last_fit_inputs:
            QMessageBox.information(self, "Nothing to finalize", "Fit at least one axis first.")
            return
        self.finalizeButton.setEnabled(False)
        self.finalizeButton.setText("Running bootstrap (n=300)…")
        QApplication.processEvents()
        try:
            for name, inputs in self._last_fit_inputs.items():
                self._bootstrap_results[name] = bootstrap_ci(
                    inputs["z"], inputs["w"], inputs["wavelength"], inputs["sigma"],
                    inputs["lower"], inputs["upper"], n_boot=300,
                )
        finally:
            self.finalizeButton.setEnabled(True)
            self.finalizeButton.setText("Finalize Analysis (bootstrap 95% CI)")
        self._render_results()

    # -- save / load / export --------------------------------------------------

    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save propagation session", "propagation_session.json", "JSON (*.json)")
        if not path:
            return
        payload = {
            "schema_version": 2,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "wavelength_nm": self.wavelengthSpin.value(),
            "method": self.methodCombo.currentText(),
            "direction": self.directionCombo.currentText(),
            "lens_focal_length_mm": self.lensEdit.value() or None,
            "scope_note": PROPAGATION_SCOPE_NOTE,
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
        self.calibrationWarningLabel.setVisible(False)
        self._bootstrap_results = {}
        self.refit()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export raw propagation CSV", "propagation_points_raw.csv", "CSV (*.csv)")
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

    def export_aggregated_csv(self):
        points = aggregate_replicates(self._checked_samples())
        if not points:
            QMessageBox.information(self, "Nothing to export", "No aggregated points available yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export aggregated propagation CSV", "propagation_points_aggregated.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            fieldnames = list(asdict(points[0]).keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for p in points:
                    row = asdict(p)
                    row["source_sample_ids"] = ";".join(row["source_sample_ids"])
                    writer.writerow(row)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


class CameraCommandBridge(QObject):
    connect_requested = pyqtSignal(object)
    disconnect_requested = pyqtSignal()
    exposure_requested = pyqtSignal(float)
    dark_requested = pyqtSignal(int)
    stop_requested = pyqtSignal()
    demo_diameter_requested = pyqtSignal(float)


class UploadImagePanel(QWidget):
    """Dedicated control surface for the Uploaded Image backend: browse a
    file, see what was detected about it (sidecar metadata, channel/format
    warnings), set the image scale, and trigger load/reprocess. Kept
    separate from ControlPanel/AdvancedSettingsPanel because an uploaded
    image is not a live acquisition source -- it has its own concerns
    (provenance, scale trust, static-source status) that don't belong
    mixed into the camera-selection or analysis-tuning panels."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        def section_label(text):
            lbl = QLabel(text)
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
            return lbl

        layout.addWidget(section_label("Upload Image"))

        self.browseButton = QPushButton("Browse Image\u2026")
        layout.addWidget(self.browseButton)

        self.selectedFileLabel = QLabel("Selected: No image selected")
        self.selectedFileLabel.setWordWrap(True)
        layout.addWidget(self.selectedFileLabel)

        self.metadataLabel = QLabel("")
        self.metadataLabel.setWordWrap(True)
        layout.addWidget(self.metadataLabel)

        self.warningsLabel = QLabel("")
        self.warningsLabel.setWordWrap(True)
        self.warningsLabel.setStyleSheet("color: #b36b00;")
        layout.addWidget(self.warningsLabel)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Channel (if multi-channel):"))
        self.channelCombo = QComboBox()
        self.channelCombo.addItems(["Luminance (qualitative)", "Red", "Green", "Blue"])
        channel_row.addWidget(self.channelCombo)
        layout.addLayout(channel_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Image Scale (\u00b5m/pixel):"))
        self.pixelScaleSpinBox = QDoubleSpinBox()
        self.pixelScaleSpinBox.setDecimals(4)
        self.pixelScaleSpinBox.setRange(0.0, 1000.0)
        self.pixelScaleSpinBox.setSingleStep(0.01)
        self.pixelScaleSpinBox.setValue(0.0)
        self.pixelScaleSpinBox.setToolTip(
            "The physical scale of THIS image, not necessarily any camera's sensor pixel pitch -- "
            "e.g. a resized, cropped, or magnified image has a different effective scale. "
            "Auto-filled only from a trusted VOILALab .json sidecar; otherwise enter manually."
        )
        scale_row.addWidget(self.pixelScaleSpinBox)
        layout.addLayout(scale_row)

        self.loadButton = QPushButton("Load / Analyze Image")
        self.loadButton.setEnabled(False)
        layout.addWidget(self.loadButton)

        self.reprocessButton = QPushButton("Reprocess (apply current settings)")
        self.reprocessButton.setEnabled(False)
        layout.addWidget(self.reprocessButton)

        layout.addStretch(1)

    def get_selected_channel(self):
        text = self.channelCombo.currentText()
        return {"Red": "red", "Green": "green", "Blue": "blue"}.get(text, "luminance")

    def show_detection(self, path, result):
        self.selectedFileLabel.setText(f"Selected: {os.path.basename(path)}")
        h, w = result["array"].shape
        self.metadataLabel.setText(
            f"{w}x{h}, dtype {result['array'].dtype}\n"
            f"Image scale: {result['pixel_pitch_um']:.4f} \u00b5m/px (from sidecar)"
            if result["pixel_pitch_um"] is not None else
            f"{w}x{h}, dtype {result['array'].dtype}\nImage scale: not available -- enter manually"
        )
        self.warningsLabel.setText("\n".join(f"\u26A0 {msg}" for msg in result["warnings"]))
        self.channelCombo.setVisible(bool(result["is_rgb_source"]))
        if result["pixel_pitch_um"] is not None:
            self.pixelScaleSpinBox.setValue(float(result["pixel_pitch_um"]))
        else:
            self.pixelScaleSpinBox.setValue(0.0)
        self._update_load_enabled()

    def _update_load_enabled(self):
        self.loadButton.setEnabled(self.pixelScaleSpinBox.value() > 0)

    def clear_selection(self):
        self.selectedFileLabel.setText("Selected: No image selected")
        self.metadataLabel.setText("")
        self.warningsLabel.setText("")
        self.pixelScaleSpinBox.setValue(0.0)
        self.loadButton.setEnabled(False)
        self.reprocessButton.setEnabled(False)


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
        self.upload_panel = UploadImagePanel()

        self.camera_panel.snapButton.clicked.connect(self.snap_to_beam)

        self.graph_panel_x = GraphPanel()
        self.graph_panel_x.ax.set_title("Live Beam Profile (X-Axis)", fontsize=10)

        self.graph_panel_y = GraphPanel()
        self.graph_panel_y.ax.set_title("Live Beam Profile (Y-Axis)", fontsize=10)

        self.sub_camera = self.add_panel(self.camera_panel, "Camera Panel", 20, 20, 500, 450)
        self.sub_graph_x = self.add_panel(self.graph_panel_x, "Beam Profile (X)", 20, 480, 340, 210)
        self.sub_graph_y = self.add_panel(self.graph_panel_y, "Beam Profile (Y)", 370, 480, 340, 210)
        self.sub_controls = self.add_panel(self.control_panel, "Controls", 530, 20, 390, 500)
        self.sub_waist = self.add_panel(self.analyze_panel, "Calculations", 930, 20, 400, 600)
        self.sub_advanced = self.add_panel(self.advanced_panel, "Advanced Settings", 950, 20, 378, 690, visible=False)
        self.sub_stability = self.add_panel(self.stability_panel, "Beam Stability", 970, 60, 320, 260, visible=False)
        self.sub_propagation = self.add_panel(self.propagation_panel, "Propagation / M² Analysis", 60, 60, 1165, 451, visible=False)
        self.sub_upload = self.add_panel(self.upload_panel, "Upload Image", 950, 60, 381, 324, visible=False)

        self.control_panel.connectCameraButton.clicked.connect(self.connect_selected_camera)
        self.control_panel.scanCamerasButton.clicked.connect(self.scan_for_cameras)
        self.control_panel.captureButton.clicked.connect(self.capture_image)
        self.control_panel.exposureSpinBox.valueChanged.connect(self.update_camera_settings)
        self.control_panel.demoDiameterSpinBox.valueChanged.connect(self.update_demo_beam_diameter)
        self.control_panel.cameraBackendCombo.currentTextChanged.connect(self.on_camera_backend_changed)
        self.upload_panel.browseButton.clicked.connect(self.browse_upload_image)
        self.upload_panel.loadButton.clicked.connect(self.load_uploaded_image)
        self.upload_panel.reprocessButton.clicked.connect(self.reprocess_current_frame)
        self.upload_panel.pixelScaleSpinBox.valueChanged.connect(self._on_upload_setting_changed)
        self.upload_panel.channelCombo.currentTextChanged.connect(self._on_upload_channel_changed)
        self.advanced_panel.calcAreaCheckbox.toggled.connect(self._on_static_source_setting_changed)
        self.advanced_panel.calcAreaSpinBox.valueChanged.connect(self._on_static_source_setting_changed)
        # Overlay-visibility checkboxes: for a static source (e.g. an
        # uploaded image), no new frame will arrive on its own to pick up
        # the changed setting, so an explicit reprocess is needed -- same
        # reasoning as calcAreaCheckbox above. Bundled together since all
        # five are the same kind of "how the current frame is rendered"
        # toggle.
        self.camera_panel.showGridCheckbox.toggled.connect(self._on_static_source_setting_changed)
        self.camera_panel.showGaussianFitCheckbox.toggled.connect(self._on_static_source_setting_changed)
        self.camera_panel.centroidCrosshairCheckbox.toggled.connect(self._on_static_source_setting_changed)
        self.camera_panel.peakCrosshairCheckbox.toggled.connect(self._on_static_source_setting_changed)
        self.advanced_panel.showIntegrationAreaCheckbox.toggled.connect(self._on_static_source_setting_changed)
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
        self._camera_bridge.demo_diameter_requested.connect(self._camera_worker.set_demo_beam_diameter_um)
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
        self.sensor_saturation_known = True
        self.current_backend_capabilities = None
        self._upload_source_filename = None
        self._pending_upload = None
        self.sensor_display_divisor = SENSOR_DISPLAY_DIVISOR
        self.min_signal_above_background = max(1, int(0.2 * self.sensor_max_raw))
        self._last_stats_ui_update = 0.0
        self._stats_ui_interval = 1.0  # seconds

        self._last_frame_ts = None
        self._fps_smoothed = None
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
        self._latest_raw_packet = None
        self._latest_quality = None
        self._latest_measurement = None
        self._dark_frame = None
        self._dark_metadata = None

        self._last_advanced_metrics_ts = None
        self._advanced_metrics_age_ms = None
        self._cached_background_level = 0.0
        self._cached_background_noise = 0.0
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

        self.action_upload = QAction("Upload Image", self)
        self.action_upload.setCheckable(True)
        self.action_upload.setChecked(False)
        self.action_upload.toggled.connect(self.toggle_upload)

        panel_actions = (self.action_video, self.action_x, self.action_y, self.action_controls,
                          self.action_waist, self.action_advanced, self.action_stability,
                          self.action_propagation, self.action_upload)
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
        reset_settings_action = QAction("Reset All Settings...", self)
        reset_settings_action.triggered.connect(self.factory_reset_settings)
        file_menu.addAction(reset_settings_action)

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
        self.sub_upload.toggle_action = self.action_upload


    def get_stable_propagation_sample(self, z_mm):
        now = time.time()
        recent = [s for ts, s in self._propagation_history if now - ts <= 2.0 and s.valid]
        if len(recent) < 5:
            return None, "Need at least five valid live frames from the last two seconds. Wait for a stable beam and try again."
        fields = ("d4sigma_x_um", "d4sigma_y_um", "d4sigma_major_um", "d4sigma_minor_um",
                  "orientation_deg", "gaussian_x_um", "gaussian_y_um", "r2_x", "r2_y")
        values = {}
        for field_name in fields:
            arr = np.asarray([getattr(s, field_name) for s in recent if getattr(s, field_name) is not None], dtype=float)
            values[field_name] = float(np.median(arr)) if arr.size else None

        # Per-axis repeatability (frame-to-frame SD), NOT collapsed into a
        # single max(x,y) number -- needed separately for X/Y weighted
        # fitting. This SD describes short-term frame-to-frame spread; it is
        # explicitly not "the uncertainty of the stored median" (see
        # PropagationSample docstring).
        def axis_sd(field_name):
            arr = np.asarray([getattr(s, field_name) for s in recent if getattr(s, field_name) is not None], dtype=float)
            if arr.size < 2 or np.mean(arr) <= 0:
                return None, None
            sd = float(np.std(arr, ddof=1))
            cv = 100.0 * sd / float(np.mean(arr)) if arr.size >= 5 else None
            return sd, cv

        d4x_sd, d4x_cv = axis_sd("d4sigma_x_um")
        d4y_sd, d4y_cv = axis_sd("d4sigma_y_um")
        gx_sd, gx_cv = axis_sd("gaussian_x_um")
        gy_sd, gy_cv = axis_sd("gaussian_y_um")
        # Kept for backward compatibility with the existing "beam is
        # fluctuating?" trip-wire dialog below, which only needs a single
        # worst-of-axes number, not per-axis weighting.
        cv = max([c for c in (d4x_cv, d4y_cv) if c is not None], default=None)
        exposure_values = [s.exposure_ms for s in recent if s.exposure_ms is not None]
        sample = PropagationSample(
            timestamp=datetime.now().isoformat(timespec="milliseconds"), z_mm=float(z_mm),
            saturation_fraction=max(s.saturation_fraction for s in recent),
            clipped=any(s.clipped for s in recent), valid=True,
            d4sigma_x_repeatability_um=d4x_sd, d4sigma_y_repeatability_um=d4y_sd,
            gaussian_x_repeatability_um=gx_sd, gaussian_y_repeatability_um=gy_sd,
            quality_flags=sorted({flag for s in recent for flag in s.quality_flags}),
            exposure_ms=float(np.median(exposure_values)) if exposure_values else None,
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

    def toggle_upload(self, checked):
        self.sub_upload.setVisible(checked)

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
        elif backend_name == "Demo / Simulated Camera":
            devices = DemoCameraBackend.list_available_devices()
        else:
            devices = []

        if not devices:
            self.control_panel.deviceCombo.addItem("No devices found (will use default)", None)
        else:
            for display_name, identifier in devices:
                self.control_panel.deviceCombo.addItem(display_name, identifier)

    def connect_selected_camera(self, show_errors=True):
        backend_name = self.control_panel.cameraBackendCombo.currentText()
        if backend_name == "Uploaded Image":
            # No hardware "device" concept applies here; loading is
            # triggered explicitly from the Upload Image panel's
            # Load/Analyze button (see load_uploaded_image), never from
            # this method -- including at startup, where this is called
            # unconditionally with whatever backend was last persisted.
            return
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
            "demo_diameter_um": self.control_panel.demoDiameterSpinBox.value(),
        })

    @pyqtSlot(object)
    def _on_camera_connected(self, info):
        self._camera_connected = True
        self.current_camera_backend_name = info["backend_name"]
        self.sensor_type = info.get("sensor_type")
        self.sensor_bit_depth = int(info["sensor_bit_depth"])
        self.sensor_max_raw = int(info["sensor_max_raw"])
        self.sensor_saturation_known = bool(info.get("sensor_saturation_known", True))
        self.sensor_display_divisor = (self.sensor_max_raw + 1) / 256.0
        self.min_signal_above_background = max(1, int(0.2 * self.sensor_max_raw))
        self._camera_info_text = info.get("info_text", "")
        self._camera_detector_size = info.get("detector_size")
        self._upload_source_filename = info.get("source_filename")
        if info.get("pixel_pitch_um") is not None:
            self.analyze_panel.pixel_pitch_input.setValue(float(info["pixel_pitch_um"]))
        self.graph_panel_x.set_intensity_mode(self.control_panel.percentModeCheckbox.isChecked(), self.sensor_max_raw)
        self.graph_panel_y.set_intensity_mode(self.control_panel.percentModeCheckbox.isChecked(), self.sensor_max_raw)
        self._reset_camera_session_state()
        self.current_backend_capabilities = {
            "supports_exposure": bool(info.get("supports_exposure", True)),
            "supports_dark_capture": bool(info.get("supports_dark_capture", True)),
            "supports_temporal_analysis": bool(info.get("supports_temporal_analysis", True)),
            "supports_frame_averaging": bool(info.get("supports_frame_averaging", True)),
            "supports_peak_hold": bool(info.get("supports_peak_hold", True)),
            "is_static_source": bool(info.get("is_static_source", False)),
        }
        self._apply_backend_capabilities()
        if self.current_backend_capabilities["is_static_source"]:
            # A static source (e.g. an uploaded image) has no acquisition
            # exposure at all -- record that plainly rather than letting
            # whatever the exposure control last showed leak into capture
            # metadata and propagation samples as if it were real.
            self.current_exposure_ms = None
        self.camera_panel.videoLabel.setText(f"Connected: {self.current_camera_backend_name}")
        self.status_camera_label.setText(f"Camera: Connected ({self.current_camera_backend_name})")
        if hasattr(self, "propagation_panel"):
            self.propagation_panel.note_environment_change(
                pixel_pitch_um=info.get("pixel_pitch_um"),
                backend_name=self.current_camera_backend_name,
            )

    def _apply_backend_capabilities(self):
        caps = self.current_backend_capabilities or {
            "supports_exposure": True, "supports_dark_capture": True,
            "supports_temporal_analysis": True, "supports_frame_averaging": True,
            "supports_peak_hold": True, "is_static_source": False,
        }

        self.control_panel.exposureSlider.setEnabled(caps["supports_exposure"])
        self.control_panel.exposureSpinBox.setEnabled(caps["supports_exposure"])
        exposure_tip = "" if caps["supports_exposure"] else "Not applicable to this source."
        self.control_panel.exposureSpinBox.setToolTip(exposure_tip)
        self.control_panel.exposureSlider.setToolTip(exposure_tip)

        self.advanced_panel.captureDarkButton.setEnabled(caps["supports_dark_capture"])
        self.advanced_panel.darkEnableCheckbox.setEnabled(caps["supports_dark_capture"])
        dark_tip = "" if caps["supports_dark_capture"] else "Dark-frame acquisition/subtraction unavailable for this source."
        self.advanced_panel.captureDarkButton.setToolTip(dark_tip)
        self.advanced_panel.darkEnableCheckbox.setToolTip(dark_tip)
        if not caps["supports_dark_capture"]:
            self.advanced_panel.darkEnableCheckbox.setChecked(False)

        self.advanced_panel.averagingSpinBox.setEnabled(caps["supports_frame_averaging"])
        avg_tip = "" if caps["supports_frame_averaging"] else "Not applicable to a single static source."
        self.advanced_panel.averagingSpinBox.setToolTip(avg_tip)
        if not caps["supports_frame_averaging"]:
            self.advanced_panel.averagingSpinBox.setValue(1)

        self.advanced_panel.holdMaxCheckbox.setEnabled(caps["supports_peak_hold"])
        self.advanced_panel.resetHoldButton.setEnabled(caps["supports_peak_hold"])
        hold_tip = "" if caps["supports_peak_hold"] else "Not applicable to a single static source."
        self.advanced_panel.holdMaxCheckbox.setToolTip(hold_tip)
        if not caps["supports_peak_hold"]:
            self.advanced_panel.holdMaxCheckbox.setChecked(False)

        self.advanced_panel.maxFpsSpinBox.setEnabled(not caps["is_static_source"])
        fps_tip = "No live acquisition rate to limit for a static source." if caps["is_static_source"] else ""
        self.advanced_panel.maxFpsSpinBox.setToolTip(fps_tip)

        self.action_stability.setEnabled(caps["supports_temporal_analysis"])
        if not caps["supports_temporal_analysis"] and self.sub_stability.isVisible():
            self.sub_stability.setVisible(False)
            self.action_stability.setChecked(False)

        if caps["is_static_source"]:
            self.control_panel.captureButton.setEnabled(self._latest_raw_packet is not None)
        else:
            self.control_panel.captureButton.setEnabled(True)

    def _reset_camera_session_state(self):
        self._last_frame_ts = None; self._fps_smoothed = None
        self._display_fps_smoothed = None; self._last_display_ts = None; self._analysis_ms_smoothed = None
        self._hold_max_frame = None; self._latest_raw_frame = None; self._latest_raw_packet = None
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
        self.current_backend_capabilities = None
        self._upload_source_filename = None
        self._apply_backend_capabilities()
        self.upload_panel.reprocessButton.setEnabled(False)
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
            "demo_diameter_um": self.control_panel.demoDiameterSpinBox.value(),
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
            "show_integration_area": self.advanced_panel.showIntegrationAreaCheckbox.isChecked(),
            "show_grid": self.camera_panel.showGridCheckbox.isChecked(),
            "show_gaussian_fit": self.camera_panel.showGaussianFitCheckbox.isChecked(),
            "show_centroid_crosshair": self.camera_panel.centroidCrosshairCheckbox.isChecked(),
            "show_peak_crosshair": self.camera_panel.peakCrosshairCheckbox.isChecked(),
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
        self.control_panel.demoDiameterSpinBox.setValue(float(d.get("demo_diameter_um", 500.0)))
        self.camera_panel.gridLabelSizeSpinBox.setValue(int(d.get("grid_label_size", 50)))
        self.analyze_panel.pixel_pitch_input.setValue(float(d.get("pixel_pitch", 3.45)))

        cmap = d.get("colormap", "Inferno")
        idx = self.control_panel.cmapCombo.findText(cmap)
        if idx >= 0:
            self.control_panel.cmapCombo.setCurrentIndex(idx)

        self.control_panel.opacitySlider.setValue(int(d.get("opacity", DEFAULT_RAW_DATA_OPACITY_PERCENT)))
        self.control_panel.zoomSlider.setValue(int(d.get("zoom", 60)))
        self.control_panel.overlayHeightSlider.setValue(int(d.get("overlay_height", OVERLAY_CURVE_HEIGHT)))
        self.control_panel.percentModeCheckbox.setChecked(bool(d.get("percent_mode", False)))

        self.advanced_panel.maxFpsSpinBox.setValue(int(d.get("max_display_fps", DEFAULT_MAX_DISPLAY_FPS)))
        self.advanced_panel.gaussianR2SpinBox.setValue(float(d.get("gaussian_r2_min", GAUSSIAN_R2_MIN)))
        self.advanced_panel.averagingSpinBox.setValue(int(d.get("averaging", 1)))
        # Overlay-visibility checkboxes are wired to _on_static_source_setting_changed
        # (see __init__), which can trigger reprocess_current_frame(). Block
        # signals while restoring so loading settings doesn't fire several
        # redundant reprocess calls back-to-back.
        for checkbox, key, default in (
            (self.advanced_panel.showIntegrationAreaCheckbox, "show_integration_area", False),
            (self.camera_panel.showGridCheckbox, "show_grid", True),
            (self.camera_panel.showGaussianFitCheckbox, "show_gaussian_fit", True),
            (self.camera_panel.centroidCrosshairCheckbox, "show_centroid_crosshair", True),
            (self.camera_panel.peakCrosshairCheckbox, "show_peak_crosshair", True),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(d.get(key, default)))
            checkbox.blockSignals(False)
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
        d = {key: s.value(key, default, type=type(default)) for key, default in SETTINGS_DEFAULTS.items()}
        self._apply_settings_dict(d)

    def _save_settings(self):
        s = self.settings
        for key, value in self._collect_settings_dict().items():
            s.setValue(key, value)

    def factory_reset_settings(self):
        reply = QMessageBox.question(
            self, "Reset All Settings",
            "Reset all VOILALab Beam settings to their defaults?\n\n"
            "This resets persisted preferences (camera backend, exposure, "
            "display, analysis, and Demo settings). Saved presets, logged "
            "CSV files, and captured images are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Disconnect BEFORE applying defaults: the camera is going away
        # regardless, so there's no reason to send it a pointless exposure
        # command in the moment before it closes. Set the flag synchronously
        # here rather than waiting on the queued _on_camera_disconnected
        # slot, since _apply_settings_dict below runs immediately afterward
        # in this same call and checks this flag.
        if self._camera_connected:
            self._camera_bridge.disconnect_requested.emit()
            self._camera_connected = False

        self.settings.clear()

        # Apply defaults to every widget right now (not just on next
        # restart). Block the backend combo's own change signal for this
        # one call so setting it doesn't trigger its own auto-rescan on top
        # of the explicit, guaranteed single scan a few lines down.
        self.control_panel.cameraBackendCombo.blockSignals(True)
        self._apply_settings_dict(SETTINGS_DEFAULTS)
        self.control_panel.cameraBackendCombo.blockSignals(False)
        self.scan_for_cameras()

        # Configuration-adjacent session state that should go with a
        # factory reset, even though it isn't itself a QSettings key.
        self._dark_frame = None
        self._dark_metadata = None
        self.advanced_panel.darkStatusLabel.setText("Dark frame: not captured")
        self.reset_hold_maximum()

        # Persist the now-default state immediately, so nothing depends on
        # how (or whether) the window gets closed afterward.
        self._save_settings()
        self.settings.sync()

        QMessageBox.information(
            self, "Settings Reset",
            "All settings have been reset to their defaults."
        )


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
        self.advanced_panel.showIntegrationAreaCheckbox.setChecked(False)
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
            caps = self.current_backend_capabilities
            already_delivered = bool(caps and caps.get("is_static_source") and self._latest_raw_packet is not None)
            if already_delivered:
                # Expected steady state for a static source (e.g. an
                # uploaded image) once it has successfully delivered its
                # one frame: there is nothing more to acquire, ever. This
                # is not staleness/failure, so none of the live-camera
                # watchdog bookkeeping applies -- an empty mailbox and a
                # worker failure are different things, and this branch
                # only covers the former. A genuine failure during the
                # static source's initial load still falls through to the
                # normal handling below, since _latest_raw_packet would
                # still be None at that point.
                return None
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

    @staticmethod
    def _set_cell(label, text, tooltip=""):
        """Set a Calculations-panel value plus its hover tooltip (px value,
        warnings), only touching the widget when something changed."""
        if label.text() != text:
            label.setText(text)
        if label.toolTip() != tooltip:
            label.setToolTip(tooltip)

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
        if self.sensor_saturation_known and raw_peak >= self.sensor_max_raw:
            raw_sat=gray_raw>=self.sensor_max_raw; sat_count=int(np.count_nonzero(raw_sat)); sat_fraction=sat_count/float(gray_raw.size)
            if sat_count:
                sy,sx=np.nonzero(raw_sat)
                spread=max(float(np.ptp(sx)) if len(sx) else 0.0,float(np.ptp(sy)) if len(sy) else 0.0)
                isolated=sat_count<=4 and spread<=2.0; central=not isolated
        low=snr_peak<self.advanced_panel.minSnrSpinBox.value()
        _,roi_max,_,maxloc=cv2.minMaxLoc(calc_img)
        if roi_max<=0: return None
        peak_x=maxloc[0]+offset_x; peak_y=maxloc[1]+offset_y

        # Centroid and D4-sigma both use the same robust (border-median +
        # MAD) background estimate already computed above for SNR/hot-pixel
        # purposes, rather than a separate whole-image low-percentile
        # estimate. The low-percentile estimate is systematically biased
        # low whenever the background has noise, and that bias -- spread
        # over every background pixel after clip-to-zero -- was the main
        # driver of D4-sigma background sensitivity (see second_moment_2d).
        cent = intensity_weighted_centroid(
            calc_img,
            background_level,
            0.0
        )
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
            moment = second_moment_2d(
                calc_img,
                background=background_level,
                background_noise=background_noise
            )
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
        raw_frame=self._acquire_frame_stage()
        if raw_frame is None:
            if self._should_refresh_stats_ui():
                self._update_performance_instrumentation()
            return
        self._latest_raw_packet = raw_frame
        self._analyze_and_render_frame(raw_frame, is_new_acquisition=True)

    def reprocess_current_frame(self):
        """Re-run analysis/rendering on the already-loaded static frame
        (e.g. after changing image scale, calc area, or another analysis
        setting for an Uploaded Image source) WITHOUT treating it as a new
        acquisition: no new propagation-history sample, no CSV row, no
        stability sample, no frame/FPS counters advanced. Used for
        is_static_source backends, which only ever deliver one real frame."""
        if self._latest_raw_packet is None:
            return
        self._analyze_and_render_frame(self._latest_raw_packet, is_new_acquisition=False)

    def _analyze_and_render_frame(self, raw_frame, is_new_acquisition=True):
        t_frame_start = time.perf_counter()
        sensor_pixel_size=self.analyze_panel.pixel_pitch_input.value()
        gray_raw=raw_frame.data
        processed=self._process_frame_stage(raw_frame)
        self._latest_processed_frame = processed
        measure_img=processed.analysis; display_img=processed.display_u8; calc_img=processed.roi
        offset_x,offset_y=processed.offset_x,processed.offset_y
        h_img,w_img=measure_img.shape
        measurement=self._measure_frame_stage(processed, force_advanced=not is_new_acquisition)
        analysis_ms = (time.perf_counter() - t_frame_start) * 1000.0
        self._analysis_ms_smoothed = analysis_ms if self._analysis_ms_smoothed is None else 0.9*self._analysis_ms_smoothed + 0.1*analysis_ms
        if measurement is None:
            # A valid frame was successfully acquired and processed above
            # (processed.display_u8 exists) -- _measure_frame_stage()
            # returning None currently means only one thing: no positive
            # signal in the selected Calculation Area. That's a
            # measurement-validity problem, not an image-validity one, so
            # the image stays on screen; only the measurement outputs are
            # cleared.
            self._show_frame_no_measurement(processed, "No measurable signal in the current Calculation Area.")
            return
        if (self.current_backend_capabilities and self.current_backend_capabilities.get("is_static_source")
                and not self.control_panel.captureButton.isEnabled()):
            self.control_panel.captureButton.setEnabled(True)
            self.upload_panel.reprocessButton.setEnabled(True)
        if not self._warned_possible_clipping and self.sensor_saturation_known and measurement.quality.raw_peak >= self.sensor_max_raw:
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
        moment_converged = True
        moment_iterations = 0
        if moment is not None:
            d4sigma_x_um = moment["d4sigma_x_px"] * pitch
            d4sigma_y_um = moment["d4sigma_y_px"] * pitch
            d4sigma_major_um = moment["d4sigma_major_px"] * pitch
            d4sigma_minor_um = moment["d4sigma_minor_px"] * pitch
            orientation_deg = moment["orientation_deg"]
            moment_ellipticity = moment["ellipticity"]
            moment_truncated = moment["truncated"]
            moment_converged = moment.get("converged", True)
            moment_iterations = moment.get("iterations", 0)
            warn_bits = []
            if moment_truncated:
                warn_bits.append("beam extends past analysis window")
            if not moment_converged:
                warn_bits.append(f"integration did not converge ({moment_iterations} iterations)")
            edge_warn = ("  \u26A0 " + "; ".join(warn_bits) + " (D4σ unreliable)") if warn_bits else ""
            d4_mark = "  \u26A0" if warn_bits else ""
            d4_note = ("\n\u26A0 " + "; ".join(warn_bits) + " (D4σ unreliable)") if warn_bits else ""
            self._set_cell(self.analyze_panel.d4sigma_x_label, f"{d4sigma_x_um:.1f} µm{d4_mark}",
                           f"{moment['d4sigma_x_px']:.1f} px{d4_note}")
            self._set_cell(self.analyze_panel.d4sigma_y_label, f"{d4sigma_y_um:.1f} µm{d4_mark}",
                           f"{moment['d4sigma_y_px']:.1f} px{d4_note}")
            self._set_text_if_changed(self.analyze_panel.d4sigma_principal_label, 
                f"Principal D4σ: {d4sigma_major_um:.1f} / {d4sigma_minor_um:.1f} µm; angle {orientation_deg:.1f}°{edge_warn}")
            if moment_ellipticity is not None:
                self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, 
                    f"Ellipticity (minor/major D4σ): {moment_ellipticity:.3f}{edge_warn}")
            else:
                self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, "Ellipticity (minor/major D4σ): N/A")
        else:
            self._set_cell(self.analyze_panel.d4sigma_x_label, "N/A")
            self._set_cell(self.analyze_panel.d4sigma_y_label, "N/A")
            self._set_text_if_changed(self.analyze_panel.d4sigma_principal_label, "Principal D4σ (major/minor, angle): N/A")
            self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, "Ellipticity (minor/major D4σ): N/A")

        if h_info is not None:
            fwhm_x = h_info.get("fwhm_width")
            if fwhm_x is not None:
                fwhm_x_um = correct_for_instrument_psf(fwhm_x * pitch, psf_um)
                trunc = h_info.get("fwhm_touches_edge")
                self._set_cell(self.analyze_panel.fwhm_x_label,
                               f"{fwhm_x_um:.1f} µm" + ("  \u26A0" if trunc else ""),
                               f"{fwhm_x:.1f} px" + ("\n\u26A0 May be truncated: the profile touches the image edge." if trunc else ""))
            if "e2_width" in h_info:
                e2_x = h_info["e2_width"]
                e2_x_um = correct_for_instrument_psf(e2_x * pitch, psf_um)
                trunc = h_info.get("e2_touches_edge")
                self._set_cell(self.analyze_panel.e2_x_label,
                               f"{e2_x_um:.1f} µm" + ("  \u26A0" if trunc else ""),
                               f"{e2_x:.1f} px" + ("\n\u26A0 May be truncated: the profile touches the image edge." if trunc else ""))
        else:
            self._set_cell(self.analyze_panel.fwhm_x_label, "N/A")
            self._set_cell(self.analyze_panel.e2_x_label, "N/A")

        if v_info is not None:
            fwhm_y = v_info.get("fwhm_width")
            if fwhm_y is not None:
                fwhm_y_um = correct_for_instrument_psf(fwhm_y * pitch, psf_um)
                trunc = v_info.get("fwhm_touches_edge")
                self._set_cell(self.analyze_panel.fwhm_y_label,
                               f"{fwhm_y_um:.1f} µm" + ("  \u26A0" if trunc else ""),
                               f"{fwhm_y:.1f} px" + ("\n\u26A0 May be truncated: the profile touches the image edge." if trunc else ""))
            if "e2_width" in v_info:
                e2_y = v_info["e2_width"]
                e2_y_um = correct_for_instrument_psf(e2_y * pitch, psf_um)
                trunc = v_info.get("e2_touches_edge")
                self._set_cell(self.analyze_panel.e2_y_label,
                               f"{e2_y_um:.1f} µm" + ("  \u26A0" if trunc else ""),
                               f"{e2_y:.1f} px" + ("\n\u26A0 May be truncated: the profile touches the image edge." if trunc else ""))
        else:
            self._set_cell(self.analyze_panel.fwhm_y_label, "N/A")
            self._set_cell(self.analyze_panel.e2_y_label, "N/A")

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
        if need_gaussian and (not is_new_acquisition or self.frame_counter % GAUSSIAN_FIT_INTERVAL == 0):
            self._last_gaussian_popt_x, self._last_gaussian_r2_x = self._fit_gaussian(
                numbers_x, x_values, h_info, center_x)
            self._last_gaussian_popt_y, self._last_gaussian_r2_y = self._fit_gaussian(
                numbers_y, y_values, v_info, center_y)

        r2_threshold = self.advanced_panel.gaussianR2SpinBox.value()
        good_fit_x = self._last_gaussian_popt_x is not None and (self._last_gaussian_r2_x or 0) >= r2_threshold
        good_fit_y = self._last_gaussian_popt_y is not None and (self._last_gaussian_r2_y or 0) >= r2_threshold

        wavelength_nm = self.advanced_panel.wavelengthSpinBox.value()
        treat_as_waist = self.analyze_panel.treatAsWaistCheckbox.isChecked()
        diam_x_um = None
        diam_y_um = None

        if good_fit_x:
            diam_x_px = gaussian_1e2_diameter(self._last_gaussian_popt_x)
            diam_x_um = correct_for_instrument_psf(diam_x_px * pitch, psf_um)
            self._set_cell(self.analyze_panel.gauss_diam_x_label,
                f"{diam_x_um:.1f} µm  (R² {self._last_gaussian_r2_x:.3f})",
                f"{diam_x_px:.1f} px\nFit shape: Gaussian (R² ≥ {r2_threshold:.3f})")
            if treat_as_waist:
                div_x_mrad = divergence_half_angle_mrad(diam_x_um / 2.0, wavelength_nm)
                self._set_text_if_changed(self.analyze_panel.divergence_x_label, 
                    f"X Min. Divergence (IF waist, M²=1): {div_x_mrad:.3f} mrad" if div_x_mrad is not None
                    else "X Divergence: N/A")
            else:
                self._set_text_if_changed(self.analyze_panel.divergence_x_label, "X Divergence: enable checkbox")
        else:
            if self._last_gaussian_r2_x is None:
                self._set_cell(self.analyze_panel.gauss_diam_x_label, "N/A")
            else:
                self._set_cell(self.analyze_panel.gauss_diam_x_label,
                    f"Not Gaussian  (R² {self._last_gaussian_r2_x:.3f})",
                    f"R² is below the fit threshold ({r2_threshold:.3f}), so the Gaussian "
                    "diameter is not reported. Use D4σ for non-Gaussian beams.")
            self._set_text_if_changed(self.analyze_panel.divergence_x_label, "X Divergence: enable checkbox")

        if good_fit_y:
            diam_y_px = gaussian_1e2_diameter(self._last_gaussian_popt_y)
            diam_y_um = correct_for_instrument_psf(diam_y_px * pitch, psf_um)
            self._set_cell(self.analyze_panel.gauss_diam_y_label,
                f"{diam_y_um:.1f} µm  (R² {self._last_gaussian_r2_y:.3f})",
                f"{diam_y_px:.1f} px\nFit shape: Gaussian (R² ≥ {r2_threshold:.3f})")
            if treat_as_waist:
                div_y_mrad = divergence_half_angle_mrad(diam_y_um / 2.0, wavelength_nm)
                self._set_text_if_changed(self.analyze_panel.divergence_y_label, 
                    f"Y Min. Divergence (IF waist, M²=1): {div_y_mrad:.3f} mrad" if div_y_mrad is not None
                    else "Y Divergence: N/A")
            else:
                self._set_text_if_changed(self.analyze_panel.divergence_y_label, "Y Divergence: enable checkbox")
        else:
            if self._last_gaussian_r2_y is None:
                self._set_cell(self.analyze_panel.gauss_diam_y_label, "N/A")
            else:
                self._set_cell(self.analyze_panel.gauss_diam_y_label,
                    f"Not Gaussian  (R² {self._last_gaussian_r2_y:.3f})",
                    f"R² is below the fit threshold ({r2_threshold:.3f}), so the Gaussian "
                    "diameter is not reported. Use D4σ for non-Gaussian beams.")
            self._set_text_if_changed(self.analyze_panel.divergence_y_label, "Y Divergence: enable checkbox")

        self._set_text_if_changed(self.analyze_panel.peak_position_label, 
            f"Peak: ({peak_x * pitch:.1f}, {peak_y * pitch:.1f}) µm")
        self._set_text_if_changed(self.analyze_panel.centroid_position_label, 
            f"Centroid: ({center_x_precise * pitch:.1f}, {center_y_precise * pitch:.1f}) µm")

        if self.sensor_saturation_known:
            self._set_text_if_changed(self.analyze_panel.saturation_label,
                f"A/D Saturation: {saturation_pct:.1f}%")
            self.analyze_panel.saturation_label.setStyleSheet("color: red;" if saturation_pct >= 95 else "")
            # Camera panel readout, color-coded for a quick glance while adjusting exposure.
            if saturation_pct >= 95:
                sat_color = "#e04040"      # red: saturating
            elif saturation_pct > 90 or saturation_pct < 20:
                sat_color = "#e0a030"      # yellow: near saturation, or weak signal
            else:
                sat_color = "#3cb371"      # green: good exposure
            self._set_text_if_changed(self.camera_panel.saturationLabel, f"Saturation: {saturation_pct:.1f}%")
            sat_style = f"color: {sat_color}; font-weight: bold;"
        else:
            self._set_text_if_changed(self.analyze_panel.saturation_label,
                f"Peak Level: {saturation_pct:.1f}% of file range \u2014 sensor saturation threshold unknown")
            self.analyze_panel.saturation_label.setStyleSheet("")
            self._set_text_if_changed(self.camera_panel.saturationLabel, f"Peak: {saturation_pct:.1f}% of file range")
            sat_style = ""
        if self.camera_panel.saturationLabel.styleSheet() != sat_style:
            self.camera_panel.saturationLabel.setStyleSheet(sat_style)

        displayed_peak = measurement.quality.displayed_peak
        averaged_peak = measurement.quality.averaged_peak
        raw_peak = measurement.quality.raw_peak
        if self.advanced_panel.enableAdvancedMetricsCheckbox.isChecked():
            self._set_cell(self.analyze_panel.background_label,
                f"Background: {background_level:.2f} counts (noise σ {background_noise:.2f})",
                f"Median background {background_level:.2f}, MAD noise σ {background_noise:.2f}\n"
                f"Background-subtracted total counts: {measurement.background_subtracted_counts:.1f}")
        else:
            self._set_cell(self.analyze_panel.background_label,
                "Background: disabled (Advanced Settings > Enable D4σ/background/hot-pixel analysis)")
        flags = []
        if low_snr: flags.append("LOW SNR")
        if central_saturation: flags.append("CENTRAL SATURATION")
        if isolated_hot_saturation: flags.append("ISOLATED HOT-PIXEL SATURATION")
        if sensor_edge_truncated: flags.append("SENSOR-EDGE TRUNCATION")
        if roi_edge_truncated: flags.append("ROI-EDGE TRUNCATION")
        if moment is not None and not moment_converged: flags.append(f"D4SIGMA NON-CONVERGED ({moment_iterations} iterations)")
        if self.advanced_panel.holdMaxCheckbox.isChecked(): flags.append("PEAK HOLD ACTIVE: METRICS ARE HISTORICAL ENVELOPE")
        peaks_tip = (f"Peaks (raw / corrected / averaged / display): {raw_peak:.1f} / "
                     f"{corrected_peak:.1f} / {averaged_peak:.1f} / {displayed_peak:.1f}")
        status_text = ", ".join(flags) if flags else "OK"
        if self.advanced_panel.enableAdvancedMetricsCheckbox.isChecked():
            if np.isinf(snr_peak):
                # Zero measured background noise gives an infinite ratio. That is
                # correct, but "inf" looks like a bug, so explain where it comes from.
                if self.current_camera_backend_name == DemoCameraBackend.name:
                    snr_text = "\u221e (noise-free simulation)"
                else:
                    snr_text = "\u221e (no measurable background noise)"
            else:
                snr_text = f"{snr_peak:.1f}"
        else:
            snr_text = "N/A (background analysis disabled)"
        self._set_cell(self.analyze_panel.quality_label,
                       f"SNR: {snr_text}   \u2022   Status: {status_text}", peaks_tip)
        quality_style = "color: #e0a030;" if flags else ""
        if self.analyze_panel.quality_label.styleSheet() != quality_style:
            self.analyze_panel.quality_label.setStyleSheet(quality_style)
        self._latest_quality = measurement.quality

        clipped = bool((h_info and (h_info.get("fwhm_touches_edge") or h_info.get("e2_touches_edge"))) or
                       (v_info and (v_info.get("fwhm_touches_edge") or v_info.get("e2_touches_edge"))) or
                       moment_truncated)
        sample_quality_flags = []
        if low_snr: sample_quality_flags.append("LOW_SNR")
        if central_saturation: sample_quality_flags.append("CENTRAL_SATURATION")
        if isolated_hot_saturation: sample_quality_flags.append("ISOLATED_HOT_PIXEL_SATURATION")
        if sensor_edge_truncated: sample_quality_flags.append("SENSOR_EDGE_TRUNCATED")
        if roi_edge_truncated: sample_quality_flags.append("ROI_EDGE_TRUNCATED")
        if moment is not None and not moment_converged: sample_quality_flags.append("D4SIGMA_NON_CONVERGED")
        current_sample = PropagationSample(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            quality_flags=sample_quality_flags,
            d4sigma_x_um=d4sigma_x_um, d4sigma_y_um=d4sigma_y_um,
            d4sigma_major_um=d4sigma_major_um, d4sigma_minor_um=d4sigma_minor_um,
            orientation_deg=orientation_deg, ellipticity=moment_ellipticity,
            d4sigma_truncated=moment_truncated,
            gaussian_x_um=diam_x_um, gaussian_y_um=diam_y_um,
            r2_x=self._last_gaussian_r2_x, r2_y=self._last_gaussian_r2_y,
            saturation_fraction=saturated_fraction, clipped=clipped,
            valid=(moment is not None and moment_converged and not clipped and not quality_invalid
                   and h_info is not None and v_info is not None
                   and not self.advanced_panel.holdMaxCheckbox.isChecked()),
            exposure_ms=self.current_exposure_ms, pixel_pitch_um=pitch, wavelength_nm=wavelength_nm,
        )
        self._latest_propagation_sample = current_sample
        if is_new_acquisition:
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

        if is_new_acquisition and self._log_writer is not None and self.frame_counter % LOG_EVERY_N_FRAMES == 0:
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

        if is_new_acquisition and self.render_stability and self.frame_counter % LOG_EVERY_N_FRAMES == 0:
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
                show_grid = self.camera_panel.showGridCheckbox.isChecked()
                show_gaussian_fit = self.camera_panel.showGaussianFitCheckbox.isChecked()
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
                        show_centroid, show_peak,
                        show_grid=show_grid, show_gaussian_fit=show_gaussian_fit,
                    )
                else:
                    if show_centroid:
                        cv2.drawMarker(rgb_image, (disp_cx, disp_cy), (0, 255, 0), cv2.MARKER_CROSS, 34, 3)
                    if show_peak:
                        cv2.drawMarker(rgb_image, (disp_px, disp_py), (255, 0, 0), cv2.MARKER_CROSS, 34, 3)

                if self.advanced_panel.showIntegrationAreaCheckbox.isChecked() and moment is not None:
                    bounds = moment.get("integration_bounds")
                    if bounds is not None:
                        rx0 = int(round(bounds["x0"] + offset_x))
                        rx1 = int(round(bounds["x1"] + offset_x)) - 1
                        ry0 = int(round(bounds["y0"] + offset_y))
                        ry1 = int(round(bounds["y1"] + offset_y)) - 1
                        # Orange if the ROI never converged (or a degenerate
                        # iteration forced an early fallback) -- a strong
                        # visual cue that the number shouldn't be trusted
                        # as-is, separate from whether it happens to touch
                        # the frame edge.
                        rect_color = (0, 165, 255) if not moment.get("converged", True) else (0, 220, 220)
                        cv2.rectangle(rgb_image, (rx0, ry0), (max(rx0, rx1), max(ry0, ry1)), rect_color, 1)

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

        if self._should_refresh_stats_ui():
            acq = self._frame_mailbox.snapshot()["acquisition_fps"]
            if acq > 0:
                self.camera_panel.fpsLabel.setText(
                    f"Measured: {acq:.1f} FPS | {1000.0 / acq:.1f} ms/frame")
            self._update_performance_instrumentation()

    def _should_refresh_stats_ui(self):
        now = time.perf_counter()
        if now - self._last_stats_ui_update >= self._stats_ui_interval:
            self._last_stats_ui_update = now
            return True
        return False

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

    def _clear_measurement_outputs(self):
        """Reset every label/plot/cache that depends on a successful beam
        measurement. Shared by _show_no_signal() (no image at all) and
        _show_frame_no_measurement() (a valid image exists, but analysis
        could not characterize it) -- these are different conditions, but
        both need the same measurement-side cleanup."""
        for x_label, y_label in self.analyze_panel.size_labels.values():
            self._set_cell(x_label, "N/A")
            self._set_cell(y_label, "N/A")
        self._set_text_if_changed(self.analyze_panel.d4sigma_principal_label, "Principal D4σ (major/minor, angle): N/A")
        self._set_text_if_changed(self.analyze_panel.d4sigma_ellipticity_label, "Ellipticity (minor/major D4σ): N/A")
        self._set_text_if_changed(self.analyze_panel.divergence_x_label, "X Divergence: enable checkbox")
        self._set_text_if_changed(self.analyze_panel.divergence_y_label, "Y Divergence: enable checkbox")

        self._set_text_if_changed(self.analyze_panel.peak_position_label, "Peak: N/A")
        self._set_text_if_changed(self.analyze_panel.centroid_position_label, "Centroid: N/A")
        self._set_text_if_changed(self.analyze_panel.saturation_label, "A/D Saturation: N/A")
        self.analyze_panel.saturation_label.setStyleSheet("")
        self._set_text_if_changed(self.camera_panel.saturationLabel, "Saturation: --")
        self.camera_panel.saturationLabel.setStyleSheet("")

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

    def _show_no_signal(self):
        # True absence of image data/source (disconnected, no packet has
        # ever arrived, or a live camera has genuinely gone stale) --
        # distinct from _show_frame_no_measurement(), which keeps a valid
        # image on screen when analysis simply couldn't characterize it.
        label = self.camera_panel.videoLabel
        label.setPixmap(QPixmap())
        viewport_size = self.camera_panel.scrollArea.viewport().size()
        label.setFixedSize(max(320, viewport_size.width()),
                           max(240, viewport_size.height()))
        label.setWordWrap(True)
        label.setText("No live frame from camera")
        self._clear_measurement_outputs()
        self.camera_panel.fpsLabel.setText("Measured: -- FPS | -- ms/frame")

    def _show_frame_no_measurement(self, processed, reason_text):
        """A valid frame exists, but _measure_frame_stage() could not
        extract a measurement from it (currently: no positive signal in
        the selected Calculation Area). Keep the actual image visible --
        this is a measurement-validity problem, not an image-validity
        problem -- and only clear/N-A the measurement-dependent outputs."""
        display_img = processed.display_u8
        cmap_choice = self.control_panel.cmapCombo.currentText()
        cv_cmap = {
            "Viridis": cv2.COLORMAP_VIRIDIS, "Jet": cv2.COLORMAP_JET,
            "Hot": cv2.COLORMAP_HOT, "Magma": cv2.COLORMAP_MAGMA,
        }.get(cmap_choice, cv2.COLORMAP_INFERNO)
        color_lut = self._get_colormap_lut(cv_cmap, cmap_choice)
        rgb_image = cv2.LUT(cv2.merge([display_img, display_img, display_img]), color_lut)

        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        zoom_factor = self.control_panel.zoomSlider.value() / 100.0
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        scaled_pixmap = pixmap if (new_w, new_h) == (w, h) else pixmap.scaled(
            new_w, new_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        if (new_w, new_h) != self._last_video_label_size:
            self.camera_panel.videoLabel.setFixedSize(new_w, new_h)
            self._last_video_label_size = (new_w, new_h)
        self.camera_panel.videoLabel.setPixmap(scaled_pixmap)

        self._clear_measurement_outputs()
        self.status_camera_label.setText(reason_text)

    def _draw_live_overlay(self, image, scale, centers, numbers_x, numbers_y, popt_x, popt_y, pixel_size,
                           zoom_factor, show_centroid=True, show_peak=True,
                           show_grid=True, show_gaussian_fit=True):
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
        crosshair_centroid_thick = max(1, int(round(OVERLAY_CROSSHAIR_CENTROID_THICK_BASE_PX * ui_scale)))

        sensor_cx = w // 2
        sensor_cy = h // 2

        # Show Grid and Show Gaussian Fit are independent controls: each
        # gates only the computation/drawing it actually needs. (An earlier
        # draft of this function computed the amp-sign/curve math whenever
        # the grid was visible; that coupling no longer exists now that the
        # two checkboxes are separate.)
        if show_gaussian_fit:
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

        if show_grid:
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
            cv2.drawMarker(overlay, (px, py), (255, 0, 0), cv2.MARKER_CROSS, crosshair_big, crosshair_centroid_thick)

        # Deliberately not early-returning `image` unchanged when every flag
        # above is False: the alpha blend is a no-op on undrawn pixels
        # already (overlay == image where nothing was drawn), so skipping it
        # would only save one cheap addWeighted call. Not worth the added
        # branch for that; revisit only if profiling ever shows this call
        # matters.
        cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0, dst=image)

    def capture_image(self):
        if not self._camera_connected:
            return

        if self.advanced_panel.holdMaxCheckbox.isChecked() and self._hold_max_frame is not None:
            dtype = np.uint8 if self.sensor_bit_depth <= 8 else np.uint16
            gray_raw = np.clip(np.round(self._hold_max_frame), 0, self.sensor_max_raw).astype(dtype)
            capture_source = "hold_maximum_envelope"
        elif self._latest_raw_frame is not None:
            gray_raw = self._latest_raw_frame.copy()
            is_static = bool(self.current_backend_capabilities and self.current_backend_capabilities.get("is_static_source"))
            capture_source = "uploaded_image" if is_static else "live_frame"
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
        fresh_advanced_age_ms = self._advanced_metrics_age_ms

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
            "source_type": "uploaded_image" if capture_source == "uploaded_image" else "live_frame",
            "source_filename": self._upload_source_filename if capture_source == "uploaded_image" else None,
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
            # Age (ms) of the cached quality/second_moment values above at
            # the moment of capture, bounded by ADVANCED_METRICS_INTERVAL_S
            # (200 ms) -- recorded rather than assumed, so a saved capture
            # is auditable instead of just trusting "probably fresh".
            "advanced_measurement_age_ms": fresh_advanced_age_ms,
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

    def update_demo_beam_diameter(self, diameter_um):
        if self._camera_connected:
            self._camera_bridge.demo_diameter_requested.emit(float(diameter_um))

    def _on_upload_setting_changed(self, _value=None):
        self.upload_panel._update_load_enabled()
        caps = self.current_backend_capabilities
        if caps and caps.get("is_static_source") and self._latest_raw_packet is not None:
            pitch = self.upload_panel.pixelScaleSpinBox.value()
            if pitch > 0:
                self.analyze_panel.pixel_pitch_input.setValue(pitch)
            self.reprocess_current_frame()

    def _on_static_source_setting_changed(self, *_args):
        caps = self.current_backend_capabilities
        if caps and caps.get("is_static_source") and self._latest_raw_packet is not None:
            self.reprocess_current_frame()

    def on_camera_backend_changed(self, backend_name):
        is_upload = backend_name == "Uploaded Image"
        is_demo = backend_name == "Demo / Simulated Camera"

        self.control_panel.scanCamerasButton.setVisible(not is_upload)
        self.control_panel.deviceLabel.setVisible(not is_upload)
        self.control_panel.deviceCombo.setVisible(not is_upload)
        self.control_panel.connectCameraButton.setVisible(not is_upload)
        self.control_panel.demoDiameterLabel.setVisible(is_demo)
        self.control_panel.demoDiameterSpinBox.setVisible(is_demo)

        if is_upload:
            # Never open a file dialog just because the dropdown changed
            # (including at startup restoration) -- browsing only happens
            # from an explicit Browse-Image click in the Upload panel.
            if not self.sub_upload.isVisible():
                self.sub_upload.setVisible(True)
                self.action_upload.setChecked(True)
        else:
            self.scan_for_cameras()

    def browse_upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Browse for beam image", "",
            "Beam Images (*.npy *.tif *.tiff *.png *.bmp *.jpg *.jpeg);;"
            "NumPy Arrays (*.npy);;TIFF Images (*.tif *.tiff);;PNG Images (*.png);;"
            "All Files (*)"
        )
        if not path:
            return
        self._detect_and_show_upload(path, self.upload_panel.get_selected_channel())

    def _on_upload_channel_changed(self, _text=None):
        upload = getattr(self, "_pending_upload", None)
        if upload is None:
            self.upload_panel._update_load_enabled()
            return
        # The channel selector changes which array detect_upload_metadata
        # produces (luminance vs. a single R/G/B channel) -- re-decode
        # rather than just letting the dropdown lie about what's loaded.
        self._detect_and_show_upload(upload["path"], self.upload_panel.get_selected_channel())

    def _detect_and_show_upload(self, path, channel):
        try:
            result = detect_upload_metadata(path, channel=channel)
        except Exception as exc:
            QMessageBox.critical(self, "Could Not Load Image", str(exc))
            return
        self._pending_upload = {"path": path, **result}
        self.upload_panel.show_detection(path, result)

    def load_uploaded_image(self):
        upload = getattr(self, "_pending_upload", None)
        if upload is None:
            return
        pitch = self.upload_panel.pixelScaleSpinBox.value()
        if pitch <= 0:
            QMessageBox.warning(self, "Image Scale Required",
                "Enter a valid image scale (\u00b5m/pixel) before loading -- it could not be "
                "determined automatically for this file.")
            return

        idx = self.control_panel.cameraBackendCombo.findText("Uploaded Image")
        if idx >= 0 and self.control_panel.cameraBackendCombo.currentIndex() != idx:
            self.control_panel.cameraBackendCombo.setCurrentIndex(idx)

        self._camera_connected = False
        self.current_camera_backend_name = "Uploaded Image"
        self.status_camera_label.setText("Camera: Connecting (Uploaded Image)")
        self.camera_panel.videoLabel.setText(f"Loading: {os.path.basename(upload['path'])}")
        self._frame_mailbox.clear()
        self._camera_bridge.connect_requested.emit({
            "backend_name": "Uploaded Image",
            "device_id": None, "device_text": os.path.basename(upload["path"]),
            # A fixed internal placeholder, never displayed as a real
            # exposure -- an uploaded image has no acquisition exposure at
            # all. self.current_exposure_ms (what capture metadata and
            # PropagationSample actually read) is set to None on connect
            # below, independent of this pipeline-internal value.
            "exposure_ms": 0.0,
            "show_errors": True,
            "upload_data": {
                "path": upload["path"], "array": upload["array"],
                "pixel_pitch_um": pitch,
                "sensor_bit_depth": upload["sensor_bit_depth"],
                "sensor_max_raw": upload["sensor_max_raw"],
                "sensor_saturation_known": upload["sensor_saturation_known"],
                "warnings": upload["warnings"],
            },
        })

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
