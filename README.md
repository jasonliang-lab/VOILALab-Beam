# VOILALab Beam

**VOILALab Beam** is a free, open-source desktop app that turns a lab camera into a laser beam profiler. It shows the beam live, measures its size and position, tracks drift over time, and measures beam quality (M²) and divergence.

Current version: **v5.0**

## Features

- Live beam image with a micrometer grid and center markers
- X and Y brightness profiles, drawn over the image and in separate graphs
- Beam size by four methods: D4σ (ISO 11146 standard), 1/e², Gaussian fit, and FWHM
- Beam shape: major/minor axes, rotation angle, and ellipticity
- Warnings for overexposure, weak signal, hot pixels, and beams cut off at the image edge
- Dark-frame subtraction and frame averaging
- Beam drift (pointing stability) graph
- Measurement logging to a CSV file
- Image capture (TIFF, NumPy, and a JSON file with the settings and results)
- Analysis of saved images (Upload Image panel)
- Beam quality (M²) and divergence from measurements at several distances
- Saved presets, and light and dark themes

Supported cameras: Thorlabs scientific cameras (Zelux and others), IDS uEye / Thorlabs DCx (UC480) cameras, and standard webcams (OpenCV). A built-in simulated beam lets you try the app without a camera.

---

## Quick start

**See the PowerPoint tutorial in [docs](docs) for a step-by-step walkthrough with screenshots.**

1. Install the camera driver from the manufacturer (not needed for webcams or the simulated beam).
2. Launch the app.
3. In **Controls**, choose the camera type (**Backend**), click **Scan**, pick the camera, and click **Connect Camera**.
4. In **Calculations**, check that **Pixel Pitch** matches your camera's pixel size. All sizes depend on it.
5. Adjust **Exposure** so the beam is bright but below 100% saturation.
6. Keep the whole beam inside the image, including its faint edges.
7. Read the results in **Calculations**. Use **Capture Image** to save a frame or **Start Logging** to record over time.

For the most accurate results, block the beam, click **Capture Dark**, then check **Subtract**.

---

## Installation from source

Keep these files in the same folder:

```text
project-folder/
├── main.py           # the app
├── GUI1.ui           # required: main window layout
├── app_icon.ico      # optional
├── requirements.txt
└── README.md
```

Install Python packages (a virtual environment is recommended):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run:

```bash
python main.py
```

`pylablib-lightweight` is only needed for IDS uEye / UC480 cameras. Thorlabs cameras also need the Thorlabs camera SDK installed.

---

## Panels

Show or hide any panel from the toolbar or the **View** menu. Closing a panel only hides it. **View → Fit Panels to Contents** resizes panels to fit their contents.

### Camera Panel
The live beam image. **Zoom** changes only the display, not the measurement. **Snap to Beam** scrolls to the beam when zoomed in. The centroid crosshair marks the beam center; the peak crosshair marks the brightest pixel.

### Beam Profile (X) and (Y)
Brightness across the beam, through its center. Shaded bands show the FWHM and 1/e² widths, and a fitted Gaussian curve appears when the fit is good.

### Controls
- **Camera:** backend, device, connect. For the simulated beam, set its size and position here.
- **Acquisition:** exposure, averaged frames, dark frame (Capture Dark, Subtract), Capture Image.
- **Display:** color map, profile overlay, zoom, and graph units (counts or % of full scale).

### Calculations
Beam size from all four methods, beam shape, position, signal quality (saturation, noise, warnings), pixel pitch, and logging. Click **Copy Path** to copy the location of the log file.

**Which width to use?** D4σ is the standard method and works for any beam shape, but it is sensitive to noise and stray light. The Gaussian fit is only meaningful for Gaussian-shaped beams. If the methods disagree a lot, the beam is probably not Gaussian.

### Advanced Settings
Settings you rarely change: minimum signal-to-noise, Gaussian fit threshold (R²), hot-pixel masking, restricting the calculation area, maximum display rate, peak hold, wavelength, and power calibration. **Reset to Defaults** restores everything.

Power readout needs a calibration factor measured against a power meter. Entering the wavelength alone does not calibrate power.

### Beam Stability
Plots how far the beam center moves from the sensor center over time (X positive right, Y positive down). Use it to check drift, vibration, and warm-up. Use logging to keep the data.

### Upload Image
Analyzes saved beam images (`.npy`, `.tif`, `.png`, `.bmp`, `.jpg`) with the same calculations as the live camera. Enter the pixel size of the camera that took the image. If a `.json` file with the same name is next to the image (as saved by **Capture Image**), the pixel size is filled in automatically. Use `.npy` or `.tif` for measurements; `.png` and `.jpg` may have lost brightness information.

### Propagation / M² Analysis
Measures how the beam changes along its path. See below.

---

## Beam divergence and M²

Divergence is how fast the beam spreads as it travels. A single image cannot measure it, because one image only shows the beam size at one distance. VOILALab Beam measures divergence the standard way (ISO 11146): it measures the beam size at several distances and fits how the beam grows.

**How to measure:**

1. Open **Propagation / M² Analysis** and enter the laser **wavelength (λ)**.
2. Place the camera at a known distance along the beam and enter it as **Z position** (mm).
3. Wait until the beam is steady and not saturated, then click **Log Stable Current Measurement**.
4. Repeat at different distances: some close to the narrowest point (the waist) and some far from it, ideally on both sides. At least 4 positions are required; 10 or more is better.

**What you get** (for X and Y):

- **M²**: beam quality (1.0 = perfect Gaussian beam)
- **Waist Ø and z₀**: size and position of the narrowest point
- **Rayleigh range zR**: the distance over which the beam stays close to its narrowest size
- **Divergence (full angle, mrad)**: calculated as θ = M²λ / (π w₀) for the half angle, doubled for the full angle

The graph shades the region near the waist (green) and the far region (red, beyond 2 Rayleigh ranges). Divergence is only reliable when you have points in the far region. The status line tells you if more positions are needed.

Sessions can be saved and reloaded (JSON) and exported as CSV.

---

## Saved files

**Capture Image** saves, with a timestamped name:

| File | Use |
|---|---|
| `.tif` | Raw image, full bit depth |
| `.npy` | Raw image for Python (`np.load`) |
| `.json` | Settings and results for this capture |
| `_display16.tif` | Brightened copy for viewing only (not for measurements) |

Captures and logs are saved in the folder the app was started from.

---

## Validation

The `validation/` folder contains the scripts used to test accuracy on computer-generated beams of known size:

1. Run `python generate_synthetic_beams.py` to create the nine test images and `truth.csv`.
2. In the app, click **Start Logging**, load each `.npy` image in the **Upload Image** panel, then click **Stop Logging**.
3. Run `python validate_synthetic.py <your_log.csv> truth.csv` to get the percent error for each method (`validation_results.csv`).

On noise-free test beams, all four width methods were within 0.2% of the true size.

---

## Notes on accuracy

- Avoid saturation (100%) and beams cut off at the image edge. The app warns about both.
- Check the pixel pitch, and capture a new dark frame after changing exposure or camera.
- With a noisy image, D4σ can read a few percent low because pixels near the noise level are left out.
- Using this software does not by itself make a measurement ISO 11146 compliant.
- This software does not replace laser-safety controls. Follow your institution's laser safety rules.

---

## Building an app (PyInstaller)

Install PyInstaller (`pip install pyinstaller`), then run:

```bash
pyinstaller beam_profiler.spec
```

The app is created in `dist/VOILALabBeam/`. Test the built app on a clean computer with the camera you plan to use.

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

Developed by Jason Liang, VOILA Lab, UC Berkeley
Contact: jason.liang2@howardcc.edu
Project: https://github.com/jasonliang-lab/VOILALab-Beam