"""Turn a VOILALab Beam log of the synthetic test images into a validation
table (Table 1).

Usage:
    python validate_synthetic.py beam_log_YYYYMMDD_HHMMSS.csv truth.csv

The log comes from the app: Start Logging, load each synthetic .npy file
in the Upload Image panel, Stop Logging. Each load writes one row whose
"source_file" column names the image. truth.csv comes from
generate_synthetic_beams.py.

Output: a table printed to the screen and validation_results.csv next to
the log. Errors are (measured - expected) / expected x 100.
"""
import csv
import os
import sys

FWHM_PER_E2_DIAMETER = 0.5887   # sqrt(2 ln 2) / 2, for a Gaussian beam


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def err_pct(measured, expected):
    if measured is None or expected is None or expected == 0:
        return None
    return (measured - expected) / expected * 100.0


def fmt(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}"


def main(log_path, truth_path):
    with open(truth_path, newline="") as f:
        truth = {row["file"]: row for row in csv.DictReader(f)}
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "source_file" not in rows[0]:
        sys.exit("This log has no 'source_file' column: it was made with an older app version.")

    # Latest row per image (if an image was loaded more than once).
    latest, repeats = {}, {}
    for row in rows:
        name = row.get("source_file", "")
        if name:
            repeats[name] = repeats.get(name, 0) + 1
            latest[name] = row

    results = []
    for name, t in truth.items():
        row = latest.get(name)
        if row is None:
            results.append({"file": name, "result": "NOT LOGGED"})
            continue
        angle = num(t["true_angle_deg"]) or 0.0
        status = row.get("status", "")
        out = {"file": name, "status": status, "times_logged": repeats[name]}

        if "truncated" in name:
            refused = "TRUNCATION" in status and row["d4sigma_x_um"] == ""
            out["result"] = "PASS: measurement refused, truncation flagged" if refused else "FAIL: not refused"
            results.append(out)
            continue

        if angle != 0:
            # Rotated ellipse: compare the principal (major/minor) D4sigma and
            # the angle; line profiles along x/y are not the beam's own axes.
            pairs = {
                "D4sigma major": (num(row["d4sigma_major_um"]), num(t["true_major_1e2_diameter_um"])),
                "D4sigma minor": (num(row["d4sigma_minor_um"]), num(t["true_minor_1e2_diameter_um"])),
            }
            out["angle_measured_deg"] = num(row["orientation_deg"])
            out["angle_true_deg"] = angle
        else:
            ex, ey = num(t["expected_D4sigma_x_um"]), num(t["expected_D4sigma_y_um"])
            pairs = {
                "D4sigma X": (num(row["d4sigma_x_um"]), ex),
                "D4sigma Y": (num(row["d4sigma_y_um"]), ey),
                "1/e2 X": (num(row["e2_x_um"]), ex),
                "1/e2 Y": (num(row["e2_y_um"]), ey),
                "Gaussian X": (num(row["gaussian_x_um"]), ex),
                "Gaussian Y": (num(row["gaussian_y_um"]), ey),
                "FWHM X": (num(row["fwhm_x_um"]), ex * FWHM_PER_E2_DIAMETER),
                "FWHM Y": (num(row["fwhm_y_um"]), ey * FWHM_PER_E2_DIAMETER),
            }
        for label, (measured, expected) in pairs.items():
            out[f"{label} measured"] = measured
            out[f"{label} expected"] = expected
            out[f"{label} error %"] = err_pct(measured, expected)
        errors = [abs(v) for k, v in out.items() if k.endswith("error %") and v is not None]
        d4_errors = [abs(v) for k, v in out.items() if k.startswith("D4sigma") and k.endswith("error %") and v is not None]
        out["max D4sigma error %"] = max(d4_errors) if d4_errors else None
        out["max error, all methods %"] = max(errors) if errors else None
        missing = [k[:-8] for k, v in out.items() if k.endswith(" error %") and v is None and not k.startswith("max")]
        out["result"] = "OK" if not missing else "missing: " + ", ".join(missing)
        results.append(out)

    # Screen table
    print(f"{'Test image':38s} {'D4sigma (meas / true, um)':30s} {'max D4s err':>11s} {'max err, all':>12s}  status")
    for r in results:
        if "max D4sigma error %" not in r:
            print(f"{r['file']:38s} {r.get('result', ''):56s}  {r.get('status', '')}")
            continue
        if "D4sigma major measured" in r:
            d4 = (f"{fmt(r['D4sigma major measured'])} x {fmt(r['D4sigma minor measured'])} / "
                  f"{fmt(r['D4sigma major expected'])} x {fmt(r['D4sigma minor expected'])}"
                  f" @ {fmt(r['angle_measured_deg'])} deg")
        else:
            d4 = (f"{fmt(r['D4sigma X measured'])} x {fmt(r['D4sigma Y measured'])} / "
                  f"{fmt(r['D4sigma X expected'])} x {fmt(r['D4sigma Y expected'])}")
        print(f"{r['file']:38s} {d4:30s} {fmt(r['max D4sigma error %'], 2):>10s}% "
              f"{fmt(r['max error, all methods %'], 2):>11s}%  {r.get('status', '')}")
    if any(r.get("times_logged", 1) > 1 for r in results):
        print("\nNote: some images were logged more than once; the LAST row for each was used.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(log_path)), "validation_results.csv")
    fields = []
    for r in results:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python validate_synthetic.py <beam_log.csv> <truth.csv>")
    main(sys.argv[1], sys.argv[2])
