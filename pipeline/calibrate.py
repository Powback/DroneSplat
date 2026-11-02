#!/usr/bin/env python3
"""
calibrate.py — automatic per-lens fisheye intrinsics for the A1. NO checkerboard.

Sources, best first:
  1. MEASURED image circle per lens (measure_aperture.py): principal point = circle centre,
     focal from the circle radius + the lens FoV. The two A1 lenses sit differently on their
     sensors, so each gets its own camera.
  2. NOMINAL geometry (3840² per lens, ~196° FoV, centred) when no apertures are given.
  3. FACTORY calibration floats in the .insv trailer — found and exposed as
     `factory_floats`, NOT used: mapping them to intrinsics needs a real A1 file to verify.

Projection: 360 fisheyes are usually equisolid, r = 2f·sin(θ/2). COLMAP/GOFish/3dgrut speak
OPENCV_FISHEYE (Kannala-Brandt): r = f·θ·(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸), which is
equidistant at k = 0. Equisolid is its Taylor series: 2·sin(θ/2) = θ − θ³/24 + θ⁵/1920 −
θ⁷/322560 + θ⁹/92897280 − … → k = (−1/24, 1/1920, −1/322560, 1/92897280), exact to
<1e-8 rad at the A1's 98° edge. (Writing k = 0 with an equisolid focal — the old bug —
put rim rays ~256 px off.)

  python3 calibrate.py --apertures SCENE/apertures.json --out SCENE/cam_init.json
  python3 calibrate.py --lens-size 3840 --fov 196 --out cam_init.json      # nominal
Output: {"lenses": {"lens0": cam, "lens1": cam}, ...lens0's fields at top level}.
"""
import argparse, json, math, re, sys
from pathlib import Path

KB = {"equisolid": (-1 / 24, 1 / 1920, -1 / 322560, 1 / 92897280),
      "equidistant": (0.0, 0.0, 0.0, 0.0)}


def edge_radius_per_focal(theta: float, projection: str) -> float:
    return 2 * math.sin(theta / 2) if projection == "equisolid" else theta


def lens_camera(width: int, height: int, cx: float, cy: float, r_edge: float,
                fov_deg: float, projection: str, source: str) -> dict:
    """OPENCV_FISHEYE camera whose `fov_deg` edge lands on the image-circle radius `r_edge`."""
    f = r_edge / edge_radius_per_focal(math.radians(fov_deg) / 2, projection)
    k1, k2, k3, k4 = KB[projection]
    return {
        "model": "OPENCV_FISHEYE", "width": width, "height": height,
        "fx": f, "fy": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "k3": k3, "k4": k4,
        "colmap_params": [f, f, cx, cy, k1, k2, k3, k4],
        "projection": projection, "fov_deg": fov_deg, "source": source,
    }


def parse_factory(insv: Path) -> list[list[float]] | None:
    """Best-effort: pull the per-lens calibration float strings from the trailer.
    Format seen in the wild:  2_<float>_<float>_..._W_H_<lens>_<float>_..._W_H_<lens>_
    Returns the raw float groups per lens; mapping floats→intrinsics is camera-RE
    work, so we expose them for inspection rather than guess wrong."""
    data = insv.read_bytes()
    m = re.findall(rb"2_[0-9eE_.\-+]*?_10496_5248_\d+_[0-9eE_.\-+]*?_10496_5248_\d+_", data)
    if not m:
        return None
    groups: list[list[float]] = []
    for part in m[0].split(b"_10496_5248_"):
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
    ap = argparse.ArgumentParser(description="Automatic per-lens A1 fisheye calibration (no checkerboard)")
    ap.add_argument("--apertures", type=Path, help="measure_aperture.py JSON (per-lens image circles)")
    ap.add_argument("--insv", type=Path, help="A1 .insv to expose factory calibration floats from")
    # A1 video = two 3840×3840 fisheye streams, ~196° FoV (per Rodrigo Polo's analysis).
    ap.add_argument("--lens-size", type=int, default=3840, help="per-lens square px (nominal path)")
    ap.add_argument("--fov", type=float, default=196.0, help="lens FoV degrees (A1 ≈ 196)")
    ap.add_argument("--projection", choices=sorted(KB), default="equisolid")
    ap.add_argument("--out", type=Path, required=True, help="output camera-init JSON")
    args = ap.parse_args()

    lenses = {}
    if args.apertures:
        for lens, a in json.loads(args.apertures.read_text()).items():
            lenses[lens] = lens_camera(a["width"], a["height"], a["cx"], a["cy"], a["r"],
                                       args.fov, args.projection, "measured-circle")
    else:
        s = args.lens_size
        for lens in ("lens0", "lens1"):
            lenses[lens] = lens_camera(s, s, s / 2, s / 2, s / 2, args.fov, args.projection, "nominal")

    out = {**lenses[sorted(lenses)[0]], "lenses": lenses}
    if args.insv and args.insv.is_file():
        factory = parse_factory(args.insv)
        if factory:
            out["factory_floats"] = factory
            print(f"  · factory calibration block found ({len(factory)} lens groups) — exposed as "
                  "'factory_floats', not used until the float mapping is verified on a real file")
    args.out.write_text(json.dumps(out, indent=2))
    for lens, c in sorted(lenses.items()):
        print(f"  {lens}: f={c['fx']:.1f}px  c=({c['cx']:.1f}, {c['cy']:.1f})  {c['projection']}, "
              f"{c['fov_deg']:.0f}° [{c['source']}]")
    print(f"✓ camera init → {args.out}")


if __name__ == "__main__":
    main()
