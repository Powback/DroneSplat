#!/usr/bin/env python3
"""
measure_aperture.py — measure the fisheye image circle (lens aperture) from the raw frames.

The A1 encodes each lens as a 3840×3840 square; outside the ~196° image circle the corners
are black. GOFish restricts training to a camera's calibrated domain — given a measured
aperture it never fits those black corners as scene content. We measure the circle from
the frames themselves (median of a sample → threshold → area-equivalent radius), so it
tracks the real optics instead of the nominal FoV.

  python3 measure_aperture.py --in SCENE/images --out SCENE/aperture.txt
Writes "CX CY RADIUS" in native pixels (the args for gofish.camera.make_domain --aperture).
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("✗ pip install opencv-python numpy")


def main() -> None:
    ap = argparse.ArgumentParser(description="measure the fisheye image circle")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="raw fisheye frames")
    ap.add_argument("--out", type=Path, required=True, help="writes 'CX CY RADIUS'")
    ap.add_argument("--samples", type=int, default=24, help="frames sampled for the median")
    ap.add_argument("--thresh", type=int, default=12, help="luma above this = inside the circle")
    ap.add_argument("--margin", type=float, default=0.99, help="shrink radius (vignetted rim)")
    args = ap.parse_args()

    frames = sorted(p for p in args.inp.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not frames:
        sys.exit(f"✗ no frames in {args.inp}")
    step = max(1, len(frames) // args.samples)
    stack = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frames[::step][:args.samples]]
    # Median over frames: a dark scene region in one frame doesn't punch a hole in the circle.
    med = np.median(np.stack(stack), axis=0).astype(np.uint8)
    mask = (med > args.thresh).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        sys.exit("✗ no image circle found (all frames black?)")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[big, cv2.CC_STAT_AREA]
    cx, cy = cents[big]
    r = math.sqrt(area / math.pi) * args.margin
    h, w = med.shape
    args.out.write_text(f"{cx:.3f} {cy:.3f} {r:.3f}\n")
    print(f"  aperture: centre ({cx:.1f}, {cy:.1f})  radius {r:.1f} px  ({w}×{h} raster)")


if __name__ == "__main__":
    main()
