#!/usr/bin/env python3
"""
reproject_sfm.py — fisheye → MULTIPLE perspective views per frame, for pose-solving only.

A single central perspective cone registers only ~1/3 of frames (those pointing at
texture). Emitting a small rig of views per frame (center + tilts) covers the whole
~196° fisheye with overlap, so COLMAP connects and registers far more. Each fisheye
pose is later recovered from ANY registered sub-view (transplant_poses.py).

Training still happens on the RAW fisheye — these proxies are throwaway scaffolding.
See docs/360-proper.md.

  python3 reproject_sfm.py --in FLIGHT_fish --out FLIGHT_proxy \
      --fisheye-focal 339 --fisheye-size 1024 --fov 90
Writes the proxy views + rig.json (per-view rotation, needed by the transplant).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("✗ pip install opencv-python numpy")

# Rig: (yaw°, pitch°) per view. Center + 4 tilts → full-FoV coverage with overlap.
DEFAULT_RIG = [(0, 0), (45, 0), (-45, 0), (0, 45), (0, -45)]


def rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return Ry @ Rx                                  # view-frame → fisheye-frame


def build_map(in_size, in_focal, out_size, fov_deg, R):
    c = in_size / 2
    fp = (out_size / 2) / math.tan(math.radians(fov_deg) / 2)
    j, i = np.meshgrid(np.arange(out_size), np.arange(out_size))
    x = (j - out_size / 2) / fp; y = (i - out_size / 2) / fp; z = np.ones_like(x)
    n = np.sqrt(x * x + y * y + z * z)
    d = np.stack([x / n, y / n, z / n], axis=-1)    # view-frame rays
    df = d @ R.T                                     # rotate into fisheye frame
    dx, dy, dz = df[..., 0], df[..., 1], df[..., 2]
    theta = np.arctan2(np.sqrt(dx * dx + dy * dy), dz)
    phi = np.arctan2(dy, dx)
    r = in_focal * theta                             # equidistant fisheye
    return (c + r * np.cos(phi)).astype(np.float32), (c + r * np.sin(phi)).astype(np.float32), fp


def main() -> None:
    ap = argparse.ArgumentParser(description="fisheye → multi-view perspective proxy for SfM")
    ap.add_argument("--in", dest="indir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fisheye-size", type=int, default=1024)
    ap.add_argument("--fisheye-focal", type=float, default=339.0)
    ap.add_argument("--fov", type=float, default=90.0, help="per-view FoV degrees")
    ap.add_argument("--out-size", type=int, default=1024)
    ap.add_argument("--single", action="store_true", help="center cone only (reliable CPU path)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rig = [(0, 0)] if args.single else DEFAULT_RIG
    maps = []
    fp = None
    for (yaw, pitch) in rig:
        mx, my, fp = build_map(args.fisheye_size, args.fisheye_focal, args.out_size, args.fov,
                               rot(yaw, pitch))
        maps.append((mx, my))

    frames = sorted(args.indir.glob("*.jpg")) + sorted(args.indir.glob("*.png"))
    if not frames:
        sys.exit(f"✗ no frames in {args.indir}")
    for f in frames:
        im = cv2.imread(str(f))
        if im is None:
            continue
        if im.shape[0] != args.fisheye_size:
            im = cv2.resize(im, (args.fisheye_size, args.fisheye_size))
        for k, (mx, my) in enumerate(maps):
            pv = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(args.out / f"{f.stem}__v{k}.jpg"), pv)

    (args.out / "rig.json").write_text(json.dumps({
        "views": [{"k": k, "yaw": y, "pitch": p, "R": rot(y, p).tolist()}
                  for k, (y, p) in enumerate(rig)],
        "pfocal": fp, "out_size": args.out_size,
    }, indent=2))
    print(f"✓ {len(frames)} frames × {len(rig)} views = {len(frames)*len(rig)} proxies → {args.out}")
    print(f"  COLMAP PINHOLE params: {fp:.2f},{fp:.2f},{args.out_size/2},{args.out_size/2}")


if __name__ == "__main__":
    main()
