"""Generate synthetic Gaussian beam images with KNOWN parameters for
validating VOILALab Beam (load them with the Upload Image panel).

Each test case is saved as:
  <name>.npy   16-bit image array (counts), what the app analyzes
  <name>.json  VOILALab sidecar: fills in the image scale automatically
  <name>.png   8-bit preview for viewing only (do NOT analyze the .png)
plus truth.csv listing the true values for every case.

Sensor model: 1440 x 1080 pixels at 3.45 um (same format as the
Thorlabs CS165MU), 12-bit (0 to 4095 counts), beam peak 3000 counts
(about 73% of full scale, so nothing saturates).

Beam model (ISO 11146 convention):
  I(x, y) = peak * exp(-2 * (u^2 / wu^2 + v^2 / wv^2))
where wu, wv are the 1/e^2 RADII along the beam's own axes u, v (rotated
by angle_deg from the image x axis). For a Gaussian beam the expected
diameters are:
  D4sigma = 1/e^2 diameter = 2 w
  FWHM    = sqrt(2 ln 2) * w = 0.5887 * (2 w)

The beam center is placed at a non-integer pixel position on purpose, so
the test does not rely on perfect pixel symmetry. Noise uses a fixed
random seed, so running this script again gives identical files.

Run:  python generate_synthetic_beams.py
"""
import csv
import json
import os

import numpy as np

PIXEL_UM = 3.45
WIDTH, HEIGHT = 1440, 1080
BIT_DEPTH = 12
MAX_RAW = 2 ** BIT_DEPTH - 1          # 4095
PEAK = 3000.0
FWHM_PER_E2_DIAMETER = np.sqrt(2.0 * np.log(2.0)) / 2.0   # 0.5887
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Center slightly off the exact sensor center (in pixels).
CX, CY = WIDTH / 2 + 0.37, HEIGHT / 2 - 0.21

# name, major diameter (um), minor diameter (um), angle (deg),
# background offset (counts), read noise sigma (counts), shot noise,
# center x (px), center y (px), purpose
CASES = [
    ("01_round_100um", 100, 100, 0, 0, 0, False, CX, CY,
     "Small beam (about 29 px across)"),
    ("02_round_500um", 500, 500, 0, 0, 0, False, CX, CY,
     "Standard beam"),
    ("03_round_1000um", 1000, 1000, 0, 0, 0, False, CX, CY,
     "Large beam"),
    ("04_ellipse_500x300um", 500, 300, 0, 0, 0, False, CX, CY,
     "Elliptical beam, axes along x and y"),
    ("05_ellipse_500x300um_rot30", 500, 300, 30, 0, 0, False, CX, CY,
     "Elliptical beam rotated 30 deg (check principal D4sigma and angle)"),
    ("06_round_500um_noise", 500, 500, 0, 50, 10, True, CX, CY,
     "Shot noise + read noise (sigma 10 counts) on a 50-count black level"),
    ("07_round_500um_background", 500, 500, 0, 300, 0, False, CX, CY,
     "Uniform 300-count background offset, no noise"),
    ("08_round_500um_noise_background", 500, 500, 0, 150, 10, True, CX, CY,
     "Realistic: 150-count background, shot noise and read noise"),
    ("09_edge_truncated_500um", 500, 500, 0, 0, 0, False, 150 / PIXEL_UM, CY,
     "Beam center only 150 um from the left edge: truncation warning expected, "
     "widths NOT expected to be accurate"),
]


def make_beam(d_major_um, d_minor_um, angle_deg, cx, cy):
    """Noise-free Gaussian (float counts) with the given 1/e^2 diameters."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    dx, dy = x - cx, y - cy
    a = np.deg2rad(angle_deg)
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    wu = d_major_um / 2.0 / PIXEL_UM
    wv = d_minor_um / 2.0 / PIXEL_UM
    return PEAK * np.exp(-2.0 * (u ** 2 / wu ** 2 + v ** 2 / wv ** 2))


def expected_axis_diameters(d_major_um, d_minor_um, angle_deg):
    """1/e^2 (= D4sigma) diameters measured along the IMAGE x and y axes.
    For a rotated ellipse the second moments mix:
      sigma_x^2 = sigma_u^2 cos^2 + sigma_v^2 sin^2  (and similarly for y)
    and D4sigma = 4 sigma."""
    a = np.deg2rad(angle_deg)
    su2 = (d_major_um / 4.0) ** 2
    sv2 = (d_minor_um / 4.0) ** 2
    sx2 = su2 * np.cos(a) ** 2 + sv2 * np.sin(a) ** 2
    sy2 = su2 * np.sin(a) ** 2 + sv2 * np.cos(a) ** 2
    return 4.0 * np.sqrt(sx2), 4.0 * np.sqrt(sy2)


def main():
    rng = np.random.default_rng(20261028)
    rows = []
    for (name, d_maj, d_min, angle, offset, read_sigma, shot, cx, cy, purpose) in CASES:
        img = make_beam(d_maj, d_min, angle, cx, cy) + offset
        if shot:
            img = rng.poisson(img).astype(np.float64)
        if read_sigma > 0:
            img = img + rng.normal(0.0, read_sigma, img.shape)
        img = np.clip(np.rint(img), 0, MAX_RAW).astype(np.uint16)

        np.save(os.path.join(OUT_DIR, f"{name}.npy"), img)
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump({"pixel_pitch_um": PIXEL_UM, "sensor_bit_depth": BIT_DEPTH,
                       "sensor_max_raw": MAX_RAW,
                       "synthetic": True, "note": purpose}, f, indent=2)
        preview = (np.clip(img / MAX_RAW, 0, 1) * 255).astype(np.uint8)
        try:
            import cv2
            cv2.imwrite(os.path.join(OUT_DIR, f"{name}.png"), preview)
        except ImportError:
            pass

        dx_exp, dy_exp = expected_axis_diameters(d_maj, d_min, angle)
        rows.append({
            "file": f"{name}.npy",
            "purpose": purpose,
            "true_major_1e2_diameter_um": f"{d_maj:.1f}",
            "true_minor_1e2_diameter_um": f"{d_min:.1f}",
            "true_angle_deg": f"{angle:.1f}",
            "expected_D4sigma_x_um": f"{dx_exp:.1f}",
            "expected_D4sigma_y_um": f"{dy_exp:.1f}",
            "expected_FWHM_x_um": f"{dx_exp * FWHM_PER_E2_DIAMETER:.1f}" if angle == 0 else "n/a (rotated)",
            "expected_FWHM_y_um": f"{dy_exp * FWHM_PER_E2_DIAMETER:.1f}" if angle == 0 else "n/a (rotated)",
            "true_center_x_px": f"{cx:.2f}",
            "true_center_y_px": f"{cy:.2f}",
            "background_offset_counts": offset,
            "read_noise_sigma_counts": read_sigma,
            "shot_noise": "yes" if shot else "no",
        })
        print(f"wrote {name}")

    with open(os.path.join(OUT_DIR, "truth.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote truth.csv")


if __name__ == "__main__":
    main()
