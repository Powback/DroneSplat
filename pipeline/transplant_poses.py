#!/usr/bin/env python3
"""
transplant_poses.py — multi-view proxy COLMAP model → fisheye TRAINING model.

reproject_sfm.py emits several perspective sub-views per fisheye frame (named
`<frame>__v<k>.jpg`) with per-view rotation R_k (rig.json). A fisheye frame is posed
if ANY of its sub-views registered; we recover the fisheye world-to-camera pose:

    R_f = R_k · R_cam_k          t_f = R_k · t_k        (R_k = view→fisheye rotation)

then write one OPENCV_FISHEYE camera + one image per frame (raw fisheye name), keeping
the proxy's points3D (world-frame, camera-model-independent). The trainer runs on the
RAW fisheye frames — full resolution; the proxy was only geometry scaffolding.

  python3 transplant_poses.py --proxy SCENE/proxy_sparse/0 --rig SCENE/proxy/rig.json \
      --cam fisheye_cam.json --ext .jpg --out SCENE/sparse/0
"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np


def q2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def main() -> None:
    ap = argparse.ArgumentParser(description="multi-view proxy → fisheye training model")
    ap.add_argument("--proxy", type=Path, required=True, help="proxy sparse model (.bin)")
    ap.add_argument("--rig", type=Path, required=True, help="rig.json from reproject_sfm")
    ap.add_argument("--cam", type=Path, required=True, help="calibrate.py JSON (training fisheye res)")
    ap.add_argument("--ext", default=".jpg", help="raw fisheye frame extension")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tmp = args.out.parent / "_proxytxt"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["colmap", "model_converter", "--input_path", str(args.proxy),
                    "--output_path", str(tmp), "--output_type", "TXT"], check=True)

    rig = {v["k"]: np.array(v["R"]) for v in json.load(open(args.rig))["views"]}

    # parse proxy images.txt (pose lines = every other non-comment line)
    lines = [l for l in (tmp / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    frames = {}   # stem -> (qvec_f, tvec_f)
    for li in range(0, len(lines), 2):
        f = lines[li].split()
        q = np.array([float(v) for v in f[1:5]]); t = np.array([float(v) for v in f[5:8]])
        name = f[9]
        stem, _, vk = name.rpartition("__v")
        k = int(vk.split(".")[0])
        if stem in frames:           # first registered sub-view per frame wins
            continue
        Rk = rig[k]
        R_cam = q2R(q)
        R_f = Rk @ R_cam
        t_f = Rk @ t
        frames[stem] = (R2q(R_f), t_f)

    cam = json.load(open(args.cam))
    (args.out / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy,k1,k2,k3,k4]\n"
        f"1 OPENCV_FISHEYE {cam['width']} {cam['height']} "
        f"{cam['fx']} {cam['fy']} {cam['cx']} {cam['cy']} "
        f"{cam['k1']} {cam['k2']} {cam['k3']} {cam['k4']}\n")

    out_lines = ["# Image list"]
    for idx, (stem, (q, t)) in enumerate(sorted(frames.items()), start=1):
        out_lines.append(f"{idx} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {stem}{args.ext}")
        out_lines.append("")          # empty points2D line (trainer uses points3D + poses)
    (args.out / "images.txt").write_text("\n".join(out_lines) + "\n")

    # points3D carry over unchanged (world frame)
    (args.out / "points3D.txt").write_text((tmp / "points3D.txt").read_text())

    print(f"✓ fisheye training model → {args.out}")
    print(f"  {len(frames)} fisheye frames posed | OPENCV_FISHEYE {cam['width']}px focal {cam['fx']:.0f}")


if __name__ == "__main__":
    main()
