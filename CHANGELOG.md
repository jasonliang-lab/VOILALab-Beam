# Changelog

All notable changes to VOILALab Beam are listed here, newest first.

## [Unreleased] (planned v5.0)

### Added
- **Demo / Simulated Camera.** An ideal, noise-free Gaussian beam with an adjustable 1/e² diameter. You can try the app without a laser or camera, and it gives a beam of known size for checking the measurements.
- **Uploaded Image source.** Analyze saved images (TIFF, PNG, BMP, JPG, NPY), with bit depth detected automatically and a choice of color channel.
- **Saturation readout in the Camera panel,** color-coded: green 20 to 90%, yellow when close to saturating or weak, red at 95% or more.
- **Propagation / M² analysis upgrades:**
  - Replicate measurements at the same Z are merged, and their spread is used to weight the fit.
  - A quality gate keeps flagged measurements (low SNR, saturation, clipping) out of the fit.
  - Coverage check: reports whether points were taken both near the waist and far from it.
  - Identifiability diagnostics explain in plain language when the fit is poorly constrained.
  - "Finalize Analysis" computes bootstrap 95% confidence intervals for w0, z0, and M².
  - Choice of width method for the fit, separate raw and aggregated CSV export, and a "How to use / caveats" guide.
- Hover tooltips for pixel values, warnings, and explanations of each measurement method.

### Changed
- **D4σ now follows the ISO 11146 approach:** an iterative integration area centered on the beam, a robust border-based background estimate, and a noise-floor threshold. Previously it was a single pass with a fixed low-percentile background, which made it very sensitive to background noise.
- **Calculations panel redesigned:** a Beam Size table (D4σ, 1/e², Gaussian fit with R², FWHM; X and Y side by side), plus Shape, Position, Signal Quality, Divergence, and Setup sections.
- **Controls panel reorganized** in the order you use it: Camera, Acquisition, Display, Graphs.
- **Advanced Settings reorganized** into Analysis, Dark Frame, Calculation Area, Acquisition, and Calibration, with Reset to Defaults moved to the bottom.
- Matching blue section headers, label-left layout, and a uniform background across panels in light and dark mode.
- Controls that do not apply to the current camera source are disabled automatically (for example, exposure for the demo camera).
- Settings are stored in one place, so loading and Reset to Defaults always agree.
- Zoom is controlled only from the Camera panel, and the default zoom is now 60%.
- SNR shows "∞ (noise-free simulation)" instead of "inf" when there is no background noise.
- The toolbar font size is the same in light and dark mode.

### Fixed
- The Camera panel FPS readout never updated. It now shows the camera's acquisition rate.
- The 1/e² threshold used 0.135 instead of e⁻² (0.1353), which overstated 1/e² widths by about 0.06%.

### Removed
- The "Clip Level (% of peak)" setting. D4σ now uses the noise-floor threshold above instead of clipping.
