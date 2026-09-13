#!/usr/bin/env python3
"""
extract_fisheye.py — pull raw dual-fisheye frames from an A1 .insv. NO SDK.

The .insv is an ordinary MP4: the video frames are standard codec; only the
trailer (calibration) is proprietary, and ffmpeg ignores it. So we extract the
raw fisheye frames directly — the fully-open, geometrically-correct route the
experienced A1 scanners recommend (skip the cosmetic equirect stitch).

  python3 extract_fisheye.py captures/raw/FLIGHT.insv --out captures/FLIGHT_fish --fps 2

Output frames feed:  OpenCV fisheye calibration (Kannala-Brandt) → COLMAP
OPENCV_FISHEYE SfM → train.  See docs/a1-pipeline.md.

⚠️ The A1's exact stream layout (two separate video tracks vs one side-by-side
   dual-fisheye frame) needs confirming on a real file — this probes and handles
   both, but verify the first extraction visually.
"""
import argparse, json, subprocess, sys
from pathlib import Path


def ffprobe_streams(src: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-select_streams", "v", str(src)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout).get("streams", [])


def extract(src: Path, out_dir: Path, fps: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    streams = ffprobe_streams(src)
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    if not vstreams:
        sys.exit("✗ no video streams found — is this a valid .insv?")

    print(f"  {len(vstreams)} video stream(s):")
    for s in vstreams:
        print(f"    idx {s['index']}: {s.get('width')}x{s.get('height')} {s.get('codec_name')}")

    if len(vstreams) >= 2:
        # Two tracks = one lens each. Extract both, tagged by lens.
        for li, s in enumerate(vstreams[:2]):
            patt = str(out_dir / f"lens{li}_%05d.jpg")
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
                 "-map", f"0:{s['index']}", "-vf", f"fps={fps}", "-q:v", "2", patt],
                check=True,
            )
        print(f"✓ extracted 2 lenses → {out_dir}  (split per track)")
    else:
        # Single track — likely side-by-side dual-fisheye. Extract whole frames;
        # split L/R downstream once you confirm the layout.
        patt = str(out_dir / "dual_%05d.jpg")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
             "-vf", f"fps={fps}", "-q:v", "2", patt],
            check=True,
        )
        print(f"✓ extracted dual-fisheye frames → {out_dir}")
        print("  (single track: inspect a frame; if side-by-side, split L|R before calibration)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract raw dual-fisheye frames from A1 .insv (no SDK)")
    ap.add_argument("input", type=Path, help="A1 .insv")
    ap.add_argument("--out", type=Path, required=True, help="output frame dir")
    ap.add_argument("--fps", type=float, default=2.0, help="frame sampling rate")
    args = ap.parse_args()
    if not args.input.is_file():
        sys.exit(f"✗ no such file: {args.input}")
    print(f"▶ probing {args.input.name}")
    extract(args.input, args.out, args.fps)


if __name__ == "__main__":
    main()
