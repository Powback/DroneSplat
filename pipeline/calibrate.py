#!/usr/bin/env python3
"""
calibrate.py — automatic fisheye intrinsics for the A1. NO checkerboard.

Two sources, best first:
  1. FACTORY calibration baked into the .insv trailer (per-lens floats). Parsed
     best-effort straight from the file — zero capture needed.
  2. NOMINAL init from the A1's known geometry (5248² per lens, ~190° FoV). COLMAP
     then self-calibrates (refines focal + distortion) during bundle adjustment.

Emits an OPENCV_FISHEYE camera init (fx, fy, cx, cy, k1..k4) for COLMAP.

  python3 calibrate.py --insv captures/raw/FLIGHT.insv --out captures/cam_init.json
  python3 calibrate.py --lens-size 5248 --fov 190 --out captures/cam_init.json
"""
import argparse, json, math, re, sys
from pathlib import Path


def nominal_fisheye(lens_size: int, fov_deg: float) -> dict:
    """Equisolid fisheye focal from FoV: r = 2 f sin(θ/2); edge r = lens_size/2."""
    half = math.radians(fov_deg) / 2
    focal = (lens_size / 2) / (2 * math.sin(half / 2))
    c = lens_size / 2
    return {
        "model": "OPENCV_FISHEYE",
        "width": lens_size, "height": lens_size,
        "fx": focal, "fy": focal, "cx": c, "cy": c,
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "source": "nominal", "fov_deg": fov_deg,
    }


def parse_factory(insv: Path) -> list[list[float]] | None:
    """Best-effort: pull the per-lens calibration float strings from the trailer.
    Format seen in the wild:  2_<float>_<float>_..._W_H_<lens>_<float>_..._W_H_<lens>_
    Returns the raw float groups per lens; mapping floats→intrinsics is camera-RE
    work, so we expose them for inspection rather than guess wrong."""
    data = insv.read_bytes()
    # Find the offset string(s): start with "2_" ... containing the sensor dims.
    m = re.findall(rb"2_[0-9eE_.\-+]*?_10496_5248_\d+_[0-9eE_.\-+]*?_10496_5248_\d+_", data)
    if not m:
        return None
    groups: list[list[float]] = []
    for chunk in m[:1]:                      # first record is enough
        for part in chunk.split(b"_10496_5248_"):
            floats = []
            for tok in part.split(b"_"):
                try:
                    floats.append(float(tok))
                except ValueError:
                    pass
            if floats:
                groups.append(floats)
    return groups or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Automatic A1 fisheye calibration (no checkerboard)")
    ap.add_argument("--insv", type=Path, help="A1 .insv to read factory calibration from")
    # A1 video = two 3840×3840 fisheye streams, ~196° FoV (per Rodrigo Polo's analysis).
    # (The 10496×5248 in the trailer is the full sensor; the encoded video is 3840² per lens.)
    ap.add_argument("--lens-size", type=int, default=3840, help="per-lens square px (A1 video = 3840)")
    ap.add_argument("--fov", type=float, default=196.0, help="lens FoV degrees (A1 ≈ 196)")
    ap.add_argument("--out", type=Path, required=True, help="output camera-init JSON")
    args = ap.parse_args()

    cam = nominal_fisheye(args.lens_size, args.fov)

    if args.insv and args.insv.is_file():
        factory = parse_factory(args.insv)
        if factory:
            cam["factory_floats"] = factory   # exposed for refinement; nominal stays the init
            print(f"  ✓ found factory calibration block ({len(factory)} lens groups) in {args.insv.name}")
            print("    (raw floats exposed as 'factory_floats'; nominal init used until float-mapping is confirmed)")
        else:
            print("  · no factory calibration block parsed — using nominal init")

    # COLMAP OPENCV_FISHEYE params order: fx, fy, cx, cy, k1, k2, k3, k4
    cam["colmap_params"] = [cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                            cam["k1"], cam["k2"], cam["k3"], cam["k4"]]
    args.out.write_text(json.dumps(cam, indent=2))
    print(f"✓ camera init → {args.out}  (focal≈{cam['fx']:.0f}px, COLMAP self-calibrates from here)")


if __name__ == "__main__":
    main()
