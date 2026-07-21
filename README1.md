# VOILALab Beam Profiler

**VOILALab Beam** is a PyQt6 desktop application for live camera-based laser beam profiling. It displays the beam image and X/Y intensity profiles, calculates beam size and position metrics, monitors pointing stability, records measurements, and supports multi-position beam-propagation analysis.

> Current application version reported by the code: **v4.6**

## Key features

- Live camera display with selectable false-color maps
- X- and Y-axis beam intensity profiles
- Centroid and peak-position crosshairs
- FWHM and 1/e² threshold widths
- Gaussian-fit diameters and fit-quality reporting
- Two-dimensional second-moment (D4σ) measurements
- Beam major/minor axes, orientation, and ellipticity
- A/D saturation, background, noise, SNR, clipping, and hot-pixel diagnostics
- Optional dark-frame subtraction
- Frame averaging and peak-hold modes
- Live beam-pointing stability graph
- CSV measurement logging
- TIFF, NumPy, and JSON capture output
- Saved operating presets
- Multi-position propagation and provisional M² analysis
- OpenCV, Thorlabs, and UC480 camera backend support
- Light and dark interface themes

## Important measurement note

This application can perform **ISO-style D4σ and beam-caustic calculations**, but using the software alone does not guarantee ISO 11146 compliance. Reliable results require appropriate camera calibration, background correction, unsaturated and unclipped images, known pixel pitch and wavelength, and adequate measurements on both sides of the beam waist.

This software is not a substitute for laser-safety controls. Follow the safety procedures required by your institution and equipment.

---

## Quick start

1. Connect the camera and install any required manufacturer driver or SDK.
2. Launch the application.
3. Open **Controls** and select the appropriate camera backend.
4. Click **Scan for Cameras**, select the device, and click **Connect Camera**.
5. Enter or verify the camera's **pixel pitch** in the **Calculations** panel.
6. Adjust exposure until the beam is clearly visible without reaching 100% A/D saturation.
7. Center the full beam inside the image or calculation area.
8. Review the X/Y profiles and measurements.
9. Use **Capture Image** to save the current measurement, or **Start Logging** to record measurements over time.

For best quantitative measurements, capture a dark frame at the same camera settings, enable dark subtraction, avoid clipping at the image or ROI boundary, and keep the camera response within its linear unsaturated range.

---

## Installation from source

### Required files

Keep these files together in the application directory:

```text
project-folder/
├── v47_packaging_ready.py
├── GUI1.ui
├── app_icon.ico              # optional at runtime
└── README.md
```

`GUI1.ui` is required because the main window is loaded from this Qt Designer file. The icon is optional; the program falls back to the default Qt icon when it is missing.

### Python requirements

The application imports the following packages:

```text
PyQt6
numpy
opencv-python
scipy
matplotlib
pylablib
tifffile
```

Install them in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install PyQt6 numpy opencv-python scipy matplotlib pylablib tifffile
```

Run the application:

```bash
python v47_packaging_ready.py
```

The optional Thorlabs and UC480 backends also require compatible camera drivers and SDK support. Installing the Python package alone may not be enough to communicate with the hardware.

---

## Interface overview

The main window uses movable and resizable subwindows. Panels can be shown or hidden from either the toolbar or the **View** menu. Closing a panel hides it rather than destroying it, so it can be reopened safely.

### Menu bar

#### File

- **Capture Image** (`Ctrl+S`) — saves the latest displayed frame and its measurement metadata.
- **Start/Stop Logging** — begins or ends CSV measurement logging.
- **Save Preset As...** — saves the current application settings under a chosen name.
- **Load Preset** — applies a previously saved preset.
- **Delete Preset...** — removes a saved preset.
- **Exit** (`Ctrl+Q`) — closes the application and safely stops the camera worker.

#### View

Shows or hides the following panels:

- Live Video
- X Profile
- Y Profile
- Controls
- Calculations
- Advanced Settings
- Beam Stability
- Propagation / M² Analysis

The **Dark Mode** option switches the interface and plots between light and dark themes.

#### Tools

- **Propagation / M² Analysis** — opens the multi-position beam-caustic analysis panel.

#### Help

- **About** — displays application information.

### Toolbar

The toolbar provides quick show/hide buttons for all major panels:

- **Live Video**
- **X Profile**
- **Y Profile**
- **Controls**
- **Calculations**
- **Advanced Settings**
- **Beam Stability**
- **Propagation / M² Analysis**

A highlighted button means that panel is visible.

### Status bar

The bottom status bar reports:

- camera connection status
- acquisition and display performance
- analysis time
- dropped frames and frame age
- worker errors
- the most recent capture

---

## Panels

## 1. Camera Panel

The **Camera Panel** displays the live beam image.

### Snap to Beam

Centers the scrollable image view on the latest measured beam centroid. This is particularly useful when the image is zoomed beyond 100%.

### Zoom

Changes only the displayed size of the camera image. It does not change the camera resolution, the recorded raw data, or the calculated beam dimensions.

### Grid Label Size

Adjusts the size of the sensor-axis gridlines and micrometer labels drawn over the live image.

### Centroid Crosshair

Shows or hides the centroid crosshair. The centroid is the intensity-weighted center of the analyzed beam.

### Peak Crosshair

Shows or hides the crosshair marking the brightest measured pixel.

### Measured FPS

Reports the achieved frame rate and approximate time per frame. This may be limited by exposure time, camera transfer rate, analysis load, or the maximum display-rate setting.

---

## 2. X Profile and Y Profile Panels

These panels show one-dimensional intensity slices through the beam centroid along the camera's horizontal and vertical axes.

Each graph can display:

- **Raw Laser Data** — the measured profile samples
- **FWHM** — the region above 50% of the profile peak above background
- **1/e² width** — the region above approximately 13.5% of the profile peak above background
- **Ideal Gaussian** — the fitted Gaussian curve when the fit passes the configured R² quality threshold

The graph scale can be shown as raw camera counts or as a percentage of the sensor's measured full-scale value.

A Gaussian-fit diameter is meaningful only when the measured profile is sufficiently Gaussian. For non-Gaussian beams, the threshold width and D4σ width may differ substantially from the Gaussian-fit result.

---

## 3. Controls Panel

The **Controls** panel contains frequently used camera and display settings.

### Show Graphs as % of Full Scale

Switches the profile graphs between raw digital counts and percentage of the camera's measured full-scale range.

### Capture Image

Saves the exact latest frame used for the on-screen measurement. When **Hold Maximum** is active, it saves the accumulated peak-hold image instead.

### Camera Exposure

The slider provides fast adjustment, while the spin box allows precise entry in milliseconds. Increase exposure for a weak signal and decrease it when the image approaches saturation.

### Color Map

Selects the false-color display map:

- Inferno
- Jet
- Hot
- Magma

The color map affects visualization only. Raw saved intensity values are not colorized.

### Zoom Level

Changes display magnification from 10% to 500%. It mirrors the zoom setting in the Camera Panel.

### Overlay Curve Height

Adjusts the height of the X/Y intensity curves drawn directly over the live video.

### Raw Data Opacity

Changes the visibility of raw sample points in the X/Y profile graphs.

### Camera Selection

1. Select a backend.
2. Click **Scan for Cameras**.
3. Select the desired device.
4. Click **Connect Camera**.

Available backend choices are defined by the application and may include OpenCV-compatible cameras, Thorlabs scientific cameras, and UC480 cameras. Backend availability depends on installed packages, drivers, SDKs, and connected hardware.

---

## 4. Calculations Panel

The **Calculations** panel contains the main quantitative beam measurements.

### Pixel Pitch

The physical size of one camera pixel in micrometers. The application attempts to obtain this automatically from supported camera backends, but it should be verified against the camera specification.

All values reported in micrometers depend directly on this setting.

### A/D Saturation

Shows the brightest pixel as a percentage of the analog-to-digital converter's full-scale value. At 100%, at least one pixel is clipped and its true intensity is unknown.

### Threshold 1/e² widths

The X and Y widths measured where the background-subtracted profile crosses 13.5% of its peak. These are direct threshold measurements and do not require the beam to be Gaussian.

### Gaussian diameters

The fitted X and Y 1/e² diameters. These are shown only when the Gaussian fit meets the configured R² threshold.

### Fit shape / R²

Reports how closely each one-dimensional profile follows a Gaussian curve. A result near 1 indicates a close fit; a lower result indicates that a Gaussian diameter may not adequately describe the profile.

### Centroid and peak position

- **Centroid position** — intensity-weighted beam center
- **Peak position** — location of the brightest pixel

These positions are distinct and may separate when the beam is asymmetric, multimodal, noisy, or affected by a hot pixel.

### FWHM and related beam widths

The panel reports beam widths derived from the X and Y profiles, including threshold-based and fitted values. Read the labels and tooltips carefully because different width definitions are not interchangeable.

### D4σ second-moment measurements

When advanced metrics are enabled, the application calculates two-dimensional second-moment beam dimensions, including:

- D4σ X and Y diameters along the camera axes
- major- and minor-axis D4σ diameters
- beam orientation
- ellipticity
- truncation and validity status

D4σ is highly sensitive to low-level background, image boundaries, stray light, and distant hot pixels. Treat a truncated or invalid result as unreliable.

### Treat current plane as beam waist

Allows the current measured plane to be treated as the beam waist for a local divergence estimate. Do not enable this unless the camera is actually positioned at, or sufficiently close to, the waist.

### Power readout

The application can estimate total power only after a user-supplied calibration factor is entered in **Advanced Settings**. Wavelength entry alone does not calibrate camera responsivity.

### Start Logging

Creates a timestamped CSV file and records beam measurements periodically. Stop logging before closing or moving the file to ensure it is finalized properly.

---

## 5. Advanced Settings Panel

These settings are intended for less frequent adjustment.

### Reset to Defaults

Restores the advanced controls to their default values.

### Capture Dark (8 frames)

Captures and averages eight dark images. Block all light from reaching the sensor and keep camera settings unchanged during capture.

### Apply Dark Subtraction

Subtracts the stored dark frame before analysis. A dark frame may be rejected or marked invalid when its camera metadata no longer matches the current acquisition settings.

Capture a new dark frame after changing conditions that materially affect the detector baseline, such as exposure, gain, pixel format, resolution, or camera.

### Minimum Peak SNR

Sets the minimum acceptable peak signal-to-noise ratio used by the application's signal-quality checks.

### Mask isolated hot pixels

Excludes detected isolated hot pixels from analysis. This can prevent a defective pixel far from the beam from strongly distorting centroid or second-moment measurements.

### Enable D4σ / background / hot-pixel analysis

Enables the computationally heavier advanced diagnostics. Disabling it can improve performance while leaving centroid, FWHM, 1/e² width, peak position, and saturation tracking available.

### Gaussian Fit Threshold (R²)

Sets the minimum fit quality required before a Gaussian-fit diameter or related Gaussian result is trusted and displayed.

### Max Display Rate

Caps the interface refresh rate. A camera may acquire faster than the interface is redrawn.

### Averaged Frames

Averages 1–20 recent frames. Higher values reduce random noise but make the display less responsive to beam motion or rapid changes.

### Restrict Calculation Area

Limits analysis to a square region centered around the last known beam position. This can reject distant reflections or artifacts, but the selected area must remain large enough to contain the complete beam and its low-intensity tails.

### Clip Level

Sets the percentage-of-peak threshold used to construct the centroid mask.

### Hold Maximum

Maintains a running per-pixel maximum image. This can capture brief pulses or the envelope of a moving signal. Click **Reset Hold** before starting a new acquisition.

### Instrument PSF

Optionally corrects measured widths for a known instrument point-spread function using quadrature subtraction. Leave this at zero unless the imaging system's PSF has been independently characterized.

### Wavelength

Stores the wavelength for metadata and propagation calculations. It does not automatically correct the camera's wavelength-dependent responsivity.

### Power calibration factor

Converts integrated background-subtracted counts to power. Calibrate this value against a trusted power meter at the actual wavelength, camera, exposure, and optical configuration being used.

### Power unit

Displays calibrated power in µW, mW, or dBm.

---

## 6. Beam Stability Panel

The **Beam Stability** panel plots centroid movement over time as X and Y offsets from the sensor center in micrometers.

Use it to observe:

- pointing drift
- vibration
- warm-up behavior
- sudden beam movement
- long-term alignment stability

The graph is a live diagnostic. Use CSV logging when measurements must be retained for later analysis.

---

## 7. Propagation / M² Analysis Panel

This panel fits a beam caustic using measurements collected at several axial positions.

### Recommended workflow

1. Enter the beam wavelength.
2. Select the width method.
3. Move the camera, beam, or focusing optic to a known Z position.
4. Enter that Z coordinate.
5. Wait until the beam is stable and unsaturated.
6. Click **Log Stable Current Measurement**.
7. Repeat at multiple positions on both sides of the waist.
8. Inspect the fitted curves, residuals, warnings, and reported parameters.

The panel requires recent consistent frames and uses a robust summary rather than a single GUI frame. At least four distinct Z positions are required for a provisional fit; ten or more well-distributed positions are preferable for meaningful coverage.

### Width methods

- **D4σ camera axes** — preferred ISO-style second-moment method
- **Gaussian-fit 1/e² diameters** — non-ISO comparison method suitable only for sufficiently Gaussian profiles

### Table controls

- **Use** — include or exclude a point from the fit
- **Delete Selected** — removes selected rows
- **Clear All** — removes all collected positions
- **Save Session JSON** — saves the full propagation session
- **Load Session JSON** — restores a saved session
- **Export CSV** — exports collected propagation points

Avoid fitting points that are saturated, clipped, unstable, or based on a truncated D4σ calculation.

---

## Saved capture files

Each capture uses a timestamped base name such as:

```text
laser_profile_20260720_101530_123_0000
```

The application may create:

```text
laser_profile_... .tif
laser_profile_... .npy
laser_profile_... .json
laser_profile_..._display16.tif
```

### Raw TIFF

Stores the raw grayscale camera values while preserving the available bit depth.

### NumPy array

Stores the exact image array and data type for later Python analysis.

Load it with:

```python
import numpy as np

image = np.load("laser_profile_....npy")
print(image.shape, image.dtype)
```

### JSON metadata

Stores settings and measurements associated with the capture, including camera backend, exposure, pixel pitch, sensor bit depth, centroid, Gaussian fit information, signal-quality results, dark correction, profiles, and second-moment measurements.

### Display-scaled 16-bit TIFF

When `tifffile` is installed, the application also saves a display-friendly TIFF whose values are shifted to use the 16-bit display range. Use the raw TIFF or NumPy file for quantitative analysis.

By default, captures and logs are written to the application's current working directory. Confirm that this directory is writable and know where the program was launched from.

---

## Presets and persistent settings

Presets are stored in:

```text
beam_profiler_presets.json
```

The program also uses Qt settings to retain applicable interface and operating choices between sessions.

A preset should document a repeatable operating configuration, but it does not replace hardware calibration. Verify exposure, pixel pitch, dark frame, wavelength, and power calibration after changing cameras or optical arrangements.

---


## Packaging notes

When building with PyInstaller, include the UI file and icon as data resources. A typical Windows build command is:

```bash
pyinstaller --noconfirm --clean --windowed --name "VOILALab Beam" \
  --icon app_icon.ico \
  --add-data "GUI1.ui;." \
  --add-data "app_icon.ico;." \
  v47_packaging_ready.py
```

On macOS or Linux, use `:` rather than `;` inside `--add-data`:

```bash
--add-data "GUI1.ui:."
```

Camera SDK libraries may require additional hidden imports, binaries, or manufacturer drivers. Test the packaged application on a clean computer with the intended camera hardware.

---

## Documentation

See powerpoint tutorial in docs. [docs](docs)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full license terms.

## Citation and authorship

```text
Developed by Jason Liang | VOILA Lab, UC Berkeley
Contact: Jason Liang, jason.liang2@howardcc.edu
Project website: https://github.com/cande3ee/VOILALab-Beam
```
