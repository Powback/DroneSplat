#!/usr/bin/env python3
"""
stitch_a1.py — unlock Antigravity A1 footage for the Insta360 MediaSDK.

THE PROBLEM (credit: u/TomMooreJD, r/AntigravityA1):
The public Insta360 MediaSDK refuses to stitch A1 files — the A1 reports lens
type 155 (video) / 112 (photo), which isn't in the SDK's dispatcher. So batch /
headless stitching is impossible without a patch.

THE FIX:
A1 optics ≈ Insta360 X4 (8K dual-fisheye, square sensor, ~2.4% size delta). Byte-
patch the lens type 155 → 113 (X4 video) and the SDK stitches it cleanly. Same
length → preserves all MP4 box offsets. For .insp photos, rename → .insv so the
file routes through the looser video stitcher, then apply the same patch.

This script does ONLY the patch + rename (no SDK needed here). Downstream you run
the MediaSDK (in Docker) on the patched file, then fix orientation with ffmpeg:
    ffmpeg -i stitched.mp4 -vf "vflip,hflip" oriented.mp4     # drone mounts upside-down

  python3 stitch_a1.py captures/raw/FLIGHT.insv -o captures/patched/
  python3 stitch_a1.py captures/raw/SHOT.insp  -o captures/patched/   # → SHOT.insv
"""
import argparse
import shutil
import sys
from pathlib import Path

# Observed in real A1 files (sensor 10496×5248).  155=video lens, 112=photo lens.
# X4 equivalents: 113=video, 71=photo.  The video path (113) is the one that works.
A1_VIDEO = b"_10496_5248_155_"
X4_VIDEO = b"_10496_5248_113_"
EXPECTED = 4  # occurrences the OP observed in the trailer


def patch_bytes(data: bytes, frm: bytes, to: bytes) -> tuple[bytes, int]:
    if len(frm) != len(to):
        sys.exit(f"✗ refusing length-changing patch ({len(frm)}→{len(to)}) — would break MP4 offsets")
    n = data.count(frm)
    return data.replace(frm, to), n


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch A1 .insv/.insp → MediaSDK-stitchable")
    ap.add_argument("input", type=Path, help="A1 .insv or .insp")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("."), help="output dir")
    ap.add_argument("--from-lens", default=A1_VIDEO.decode(), help="byte pattern to find")
    ap.add_argument("--to-lens", default=X4_VIDEO.decode(), help="byte pattern to write")
    args = ap.parse_args()

    src = args.input
    if not src.is_file():
        sys.exit(f"✗ no such file: {src}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # .insp → .insv rename trick (routes through the looser video stitcher)
    suffix = src.suffix.lower()
    if suffix == ".insp":
        out = args.out_dir / (src.stem + ".insv")
        print(f"  .insp → .insv rename trick: {out.name}")
    elif suffix == ".insv":
        out = args.out_dir / src.name
    else:
        sys.exit(f"✗ expected .insv or .insp, got {suffix}")

    data = src.read_bytes()
    patched, n = patch_bytes(data, args.from_lens.encode(), args.to_lens.encode())
    if n == 0:
        sys.exit(f"✗ pattern {args.from_lens!r} not found — is this really A1 footage? "
                 f"(check lens type with: strings '{src}' | grep _10496_5248_)")
    if n != EXPECTED:
        print(f"  ⚠️ patched {n} occurrence(s) (OP saw {EXPECTED}); build may differ — verify the stitch")
    out.write_bytes(patched)
    print(f"✓ patched {n}× lens {args.from_lens.split('_')[-2]}→{args.to_lens.split('_')[-2]} → {out}")
    print("  next: run Insta360 MediaSDK (Docker) on this file, then ffmpeg -vf 'vflip,hflip'")


if __name__ == "__main__":
    main()
