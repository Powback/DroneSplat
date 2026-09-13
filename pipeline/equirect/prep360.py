#!/usr/bin/env python3
"""
prep360.py — Antigravity A1 360° video → clean perspective image set for VGGT.

Pipeline (CPU-only, no GPU needed):
  1. Extract frames from equirectangular video (ffmpeg, fps sampling).
  2. Drop motion-blurred frames (Laplacian-variance sharpness, best-per-batch).
  3. Reproject each equirect frame → N perspective virtual cameras
     (default 18: 8 yaw × 2 pitch ±35° ring + zenith + nadir).

Output is a flat folder of perspective JPEGs that VGGT / InstantSplat consume
directly — VGGT solves the poses, so NOTHING here is manually posed, and there is
NO COLMAP step. No masking (a drone has no operator/selfie-stick in frame).

  python3 prep360.py --video captures/scene.mp4 --out captures/scene_persp
"""
import argparse, json, math, os, subprocess, sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("✗ opencv-python missing.  pip install opencv-python numpy")


# ── 1. frame extraction ───────────────────────────────────────────────────────
def extract_frames(video: Path, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.jpg")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2", pattern],
        check=True,
    )
    return sorted(out_dir.glob("frame_*.jpg"))


# ── 2. sharpness filter (Laplacian variance, best-per-batch) ──────────────────
def sharpness(img_path: Path) -> float:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_sharp(frames: list[Path], batch: int, keep: int) -> list[Path]:
    chosen: list[Path] = []
    for i in range(0, len(frames), batch):
        group = frames[i : i + batch]
        ranked = sorted(group, key=sharpness, reverse=True)
        chosen.extend(ranked[:keep])
    return sorted(chosen)


# ── 3. equirectangular → perspective virtual cameras ──────────────────────────
def virtual_cameras(ring_yaws: int = 8, pitch_deg: float = 35.0) -> list[tuple[float, float]]:
    """(yaw°, pitch°) for each virtual camera. Ring at ±pitch + 2 poles."""
    cams: list[tuple[float, float]] = []
    for k in range(ring_yaws):
        yaw = 360.0 * k / ring_yaws
        cams.append((yaw, +pitch_deg))
        cams.append((yaw, -pitch_deg))
    cams.append((0.0, +90.0))   # zenith
    cams.append((0.0, -90.0))   # nadir
    return cams


def _rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return Ry @ Rx


def build_maps(eq_w: int, eq_h: int, out: int, fov_deg: float,
               yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-sampling maps for one gnomonic (perspective) view of the sphere."""
    f = (out / 2) / math.tan(math.radians(fov_deg) / 2)
    cx = cy = out / 2
    j, i = np.meshgrid(np.arange(out), np.arange(out))          # x=cols, y=rows
    x = (j - cx) / f
    y = (i - cy) / f
    z = np.ones_like(x)
    d = np.stack([x, y, z], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    world = d @ _rot(yaw, pitch).T                              # rotate into sphere frame
    dx, dy, dz = world[..., 0], world[..., 1], world[..., 2]
    lon = np.arctan2(dx, dz)                                    # [-pi, pi]
    lat = np.arcsin(np.clip(dy, -1, 1))                         # [-pi/2, pi/2]
    map_x = ((lon / (2 * math.pi)) + 0.5) * eq_w
    map_y = (0.5 - (lat / math.pi)) * eq_h
    return map_x.astype(np.float32), map_y.astype(np.float32)


def reproject(frames: list[Path], out_dir: Path, cams: list[tuple[float, float]],
              out_size: int, fov_deg: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cache sampling maps once (same for every frame of equal size).
    maps: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    n = 0
    for fi, fp in enumerate(frames):
        eq = cv2.imread(str(fp))
        if eq is None:
            continue
        h, w = eq.shape[:2]
        for ci, (yaw, pitch) in enumerate(cams):
            key = (yaw, pitch)
            if key not in maps:
                maps[key] = build_maps(w, h, out_size, fov_deg, yaw, pitch)
            mx, my = maps[key]
            view = cv2.remap(eq, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            cv2.imwrite(str(out_dir / f"f{fi:05d}_c{ci:02d}.jpg"),
                        view, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 360° video → perspective views for VGGT")
    ap.add_argument("--video", type=Path, help="equirectangular MP4 (or use --frames-dir)")
    ap.add_argument("--frames-dir", type=Path, help="pre-extracted equirect frames")
    ap.add_argument("--out", type=Path, required=True, help="output perspective-image dir")
    ap.add_argument("--fps", type=float, default=2.0, help="frame sampling rate")
    ap.add_argument("--sharp-batch", type=int, default=6, help="frames per sharpness batch")
    ap.add_argument("--sharp-keep", type=int, default=2, help="keep N sharpest per batch")
    ap.add_argument("--ring-yaws", type=int, default=8, help="yaw samples in the ring")
    ap.add_argument("--pitch", type=float, default=35.0, help="ring pitch ± degrees")
    ap.add_argument("--fov", type=float, default=90.0, help="virtual-camera FoV degrees")
    ap.add_argument("--size", type=int, default=1024, help="perspective image px (square)")
    ap.add_argument("--no-sharp", action="store_true", help="skip sharpness filtering")
    args = ap.parse_args()

    work = args.out.parent / (args.out.name + "_frames")
    if args.frames_dir:
        frames = sorted(args.frames_dir.glob("*.jpg")) + sorted(args.frames_dir.glob("*.png"))
    elif args.video:
        print(f"▶ 1/3 extracting frames @ {args.fps}fps")
        frames = extract_frames(args.video, work, args.fps)
    else:
        sys.exit("✗ need --video or --frames-dir")
    print(f"  {len(frames)} frames")

    if not args.no_sharp:
        print("▶ 2/3 sharpness filter")
        frames = select_sharp(frames, args.sharp_batch, args.sharp_keep)
        print(f"  kept {len(frames)} sharp frames")

    cams = virtual_cameras(args.ring_yaws, args.pitch)
    print(f"▶ 3/3 reprojecting → {len(cams)} virtual cams/frame")
    total = reproject(frames, args.out, cams, args.size, args.fov)

    # KNOWN intrinsics (the RealityScan/Pano2Views trick): we reprojected to a
    # rectilinear pinhole, so the focal length is exact — hand it to the pose
    # solver instead of letting it guess. Views are already undistorted.
    focal_px = (args.size / 2) / math.tan(math.radians(args.fov) / 2)
    meta = {
        "perspective_views": total, "frames": len(frames),
        "cameras_per_frame": len(cams), "fov_deg": args.fov, "size": args.size,
        "ring_yaws": args.ring_yaws, "pitch_deg": args.pitch,
        "intrinsics": {                       # ← pass to VGGT / write to COLMAP cameras.txt
            "model": "PINHOLE", "undistorted": True,
            "width": args.size, "height": args.size,
            "fx": focal_px, "fy": focal_px,
            "cx": args.size / 2, "cy": args.size / 2,
        },
        # per-virtual-camera orientation within a panorama (rig prior, optional)
        "rig": [{"cam": ci, "yaw_deg": y, "pitch_deg": p,
                 "R": _rot(y, p).tolist()} for ci, (y, p) in enumerate(cams)],
    }
    (args.out / "prep360.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ {total} perspective views → {args.out}  (focal={focal_px:.1f}px, known intrinsics emitted)")


if __name__ == "__main__":
    main()
