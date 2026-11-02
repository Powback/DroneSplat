#!/usr/bin/env python3
"""
measure_aperture.py — measure each lens's fisheye image circle from the raw frames.

The A1 encodes each lens as a 3840×3840 square; outside the ~196° image circle the corners
are black. The circle is the lens's real geometry: its centre is the principal point and its
radius fixes the focal (calibrate.py), and GOFish restricts training to it so the black
corners are never fit as scene content. Measured per lens (lens0_*, lens1_* frames) because
the two lenses sit differently on the sensor: median of a sample → threshold →
area-equivalent radius.

  python3 measure_aperture.py --in SCENE/images --out SCENE/apertures.json
Writes {"lens0": {"cx", "cy", "r", "r_train", "width", "height"}, "lens1": {...}} in native
pixels; r_train = r × margin (the vignetted rim is left out of training).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("✗ pip install opencv-python numpy")


def measure(frames: list[Path], samples: int, thresh: int) -> dict:
    step = max(1, len(frames) // samples)
    stack = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frames[::step][:samples]]
    # Median over frames: a dark scene region in one frame doesn't punch a hole in the circle.
    med = np.median(np.stack(stack), axis=0).astype(np.uint8)
    mask = (med > thresh).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        sys.exit("✗ no image circle found (all frames black?)")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    h, w = med.shape
    cx, cy = cents[big]
    return {"cx": float(cx), "cy": float(cy),
            "r": math.sqrt(stats[big, cv2.CC_STAT_AREA] / math.pi), "width": w, "height": h}


def main() -> None:
    ap = argparse.ArgumentParser(description="measure each lens's fisheye image circle")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="raw fisheye frames (lensN_*.jpg)")
    ap.add_argument("--out", type=Path, required=True, help="apertures JSON")
    ap.add_argument("--samples", type=int, default=24, help="frames sampled for the median")
    ap.add_argument("--thresh", type=int, default=12, help="luma above this = inside the circle")
    ap.add_argument("--margin", type=float, default=0.99, help="r_train = r × margin (vignetted rim)")
    args = ap.parse_args()

    frames = sorted(p for p in args.inp.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    lenses: dict[str, list[Path]] = {}
    for p in frames:
        lenses.setdefault(p.stem.split("_")[0], []).append(p)
    if not lenses:
        sys.exit(f"✗ no frames in {args.inp}")
    out = {}
    for lens, group in sorted(lenses.items()):
        a = measure(group, args.samples, args.thresh)
        a["r_train"] = a["r"] * args.margin
        out[lens] = a
        print(f"  {lens}: centre ({a['cx']:.1f}, {a['cy']:.1f})  radius {a['r']:.1f} px  "
              f"({a['width']}×{a['height']}, {len(group)} frames)")
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
