#!/usr/bin/env python3
"""
extract_fisheye.py — pull raw dual-fisheye frames from an A1 .insv / .insp. NO SDK.

The .insv is an ordinary MP4: the video frames are standard codec; only the
trailer (calibration) is proprietary, and ffmpeg ignores it. So we extract the
raw fisheye frames directly — the fully-open, geometrically-correct route the
experienced A1 scanners recommend (skip the cosmetic equirect stitch).

Handles every A1 layout we know of, always writing one frame per lens:
  - .insv, two video tracks (the A1's 2× 3840² HEVC) → lens0_*, lens1_*
  - .insv, one side-by-side track (W = 2H)         → split into lens0_* | lens1_*
  - .insp photo (JPEG of both circles side by side + trailer) → lens0_00000 | lens1_00000
  - a folder of .insv/.insp from one flight         → all of the above, numbered on

Refuses truncated downloads: the A1 writes `moov` after the huge `mdat`, so a cut-off
file loses `moov` and ffmpeg fails late with "moov atom not found" (docs/a1-pipeline.md).

  python3 extract_fisheye.py captures/raw/FLIGHT.insv --out captures/FLIGHT_fish --fps 2

Output frames feed: per-lens aperture + calibration → COLMAP → train. See docs/a1-pipeline.md.
"""
import argparse, json, struct, subprocess, sys
from pathlib import Path

LENSES = ("lens0", "lens1")


def ffprobe_streams(src: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-select_streams", "v", str(src)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout).get("streams", [])


def check_mp4(src: Path) -> None:
    """Walk the top-level MP4 boxes: `moov` must exist and no box may run past EOF."""
    size = src.stat().st_size
    seen, pos = set(), 0
    with open(src, "rb") as fh:
        while pos + 8 <= size:
            fh.seek(pos)
            n, kind = struct.unpack(">I4s", fh.read(8))
            if n == 1:                                   # 64-bit largesize
                n = struct.unpack(">Q", fh.read(8))[0]
            elif n == 0:                                 # box runs to EOF
                n = size - pos
            if n < 8:
                break                                    # the Insta360 trailer isn't box-shaped
            if pos + n > size:
                sys.exit(f"✗ {src.name}: '{kind.decode(errors='replace')}' box needs {pos + n:,} bytes, "
                         f"file has {size:,} — truncated download? Re-download it.")
            seen.add(kind)
            pos += n
    if b"moov" not in seen:
        sys.exit(f"✗ {src.name}: no 'moov' atom — truncated download (A1 writes moov last).")


def ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


def split_sbs(frames: list[Path], out_dir: Path, start: int) -> int:
    """Split side-by-side dual-fisheye frames into lens0 (left) | lens1 (right)."""
    try:
        import cv2
    except ImportError:
        sys.exit("✗ pip install opencv-python numpy")
    n = start
    for f in frames:
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        half = w // 2
        cv2.imwrite(str(out_dir / f"lens0_{n:05d}.jpg"), im[:, :half], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / f"lens1_{n:05d}.jpg"), im[:, half:half * 2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        f.unlink()
        n += 1
    return n


def extract_video(src: Path, out_dir: Path, fps: float, start: int) -> int:
    check_mp4(src)
    vstreams = [s for s in ffprobe_streams(src) if s.get("codec_type") == "video"]
    if not vstreams:
        sys.exit(f"✗ {src.name}: no video streams — is this a valid .insv?")
    print(f"  {src.name}: {len(vstreams)} video stream(s): "
          + ", ".join(f"{s.get('width')}x{s.get('height')} {s.get('codec_name')}" for s in vstreams))
    if len(vstreams) >= 2:
        # Two tracks = one lens each (the A1's layout). Same fps filter → frame k of
        # each track is the same instant.
        for lens, s in zip(LENSES, vstreams[:2]):
            ffmpeg(["-i", str(src), "-map", f"0:{s['index']}", "-vf", f"fps={fps}", "-q:v", "2",
                    "-start_number", str(start), str(out_dir / f"{lens}_%05d.jpg")])
        # Tracks can differ by a frame at the tail; keep only complete pairs.
        names = [{p.name[6:] for p in out_dir.glob(f"{lens}_*.jpg")} for lens in LENSES]
        for lens, own, other in ((LENSES[0], names[0], names[1]), (LENSES[1], names[1], names[0])):
            for tail in own - other:
                (out_dir / f"{lens}_{tail}").unlink()
        return len(names[0] & names[1])
    s = vstreams[0]
    if int(s.get("width", 0)) != 2 * int(s.get("height", 0)):
        sys.exit(f"✗ {src.name}: single {s.get('width')}x{s.get('height')} track isn't side-by-side "
                 "dual-fisheye (expected W = 2H) — stitched equirect? Use pipeline/equirect/ instead.")
    tmp = out_dir / "_sbs"
    tmp.mkdir(exist_ok=True)
    ffmpeg(["-i", str(src), "-vf", f"fps={fps}", "-q:v", "2", str(tmp / "dual_%05d.jpg")])
    n = split_sbs(sorted(tmp.glob("dual_*.jpg")), out_dir, start)
    tmp.rmdir()
    return n


def extract_photo(src: Path, out_dir: Path, start: int) -> int:
    # .insp = JPEG (both circles side by side) + Insta360 trailer. JPEG decoders stop at
    # EOI, so the trailer is ignored; ffmpeg's image2 demuxer reads it like any .jpg.
    tmp = out_dir / "_sbs"
    tmp.mkdir(exist_ok=True)
    ffmpeg(["-f", "image2", "-c:v", "mjpeg", "-i", str(src), "-frames:v", "1", "-q:v", "2",
            str(tmp / "dual_00000.jpg")])
    n = split_sbs(sorted(tmp.glob("dual_*.jpg")), out_dir, start)
    tmp.rmdir()
    print(f"  {src.name}: photo → lens0/lens1")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract raw per-lens fisheye frames from A1 .insv/.insp (no SDK)")
    ap.add_argument("input", type=Path, help="A1 .insv / .insp, or a folder of them (one flight)")
    ap.add_argument("--out", type=Path, required=True, help="output frame dir")
    ap.add_argument("--fps", type=float, default=2.0, help="video frame sampling rate")
    args = ap.parse_args()
    if args.input.is_dir():
        srcs = sorted(p for p in args.input.iterdir()
                      if p.suffix.lower() in (".insv", ".insp") and not p.name.upper().startswith("LRV"))
    elif args.input.is_file():
        srcs = [args.input]
    else:
        sys.exit(f"✗ no such file: {args.input}")
    if not srcs:
        sys.exit(f"✗ no .insv/.insp in {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in srcs:
        n = extract_photo(src, args.out, n) if src.suffix.lower() == ".insp" \
            else extract_video(src, args.out, args.fps, n)
    pairs = len(list(args.out.glob("lens0_*.jpg")))
    print(f"✓ {pairs} frame pairs (lens0 + lens1) → {args.out}")


if __name__ == "__main__":
    main()
