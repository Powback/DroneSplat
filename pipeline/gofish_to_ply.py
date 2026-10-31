#!/usr/bin/env python3
"""
gofish_to_ply.py — GOFish checkpoint (.pt) → standard 3DGS .ply for splat.pow / SuperSplat.

GOFish (3dgrut-based) saves torch checkpoints, not a splat .ply. Its parameters use the
INRIA 3DGS convention (sigmoid opacity logits, log scales, wxyz quaternions), so they map
straight onto the standard ply fields. Two conversions:
  - SH: 3dgrut stores higher-order SH per coefficient as interleaved RGB; the ply's
    f_rest_* is channel-major (all R, then G, then B) → transpose.
  - Mip filter: GOFish keeps the 3D Mip filter out of the stored scales and adds it at
    render time; a stock viewer can't, so we bake it in (see HACK below).

Runs in the GOFish venv (needs torch + omegaconf to unpickle the checkpoint config).
  python gofish_to_ply.py --checkpoint RUN/ckpt_last.pt --out FLIGHT.ply
"""
import argparse
from pathlib import Path
import numpy as np
import torch


def np32(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="GOFish checkpoint → 3DGS .ply")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    xyz = np32(ck["positions"])
    n = xyz.shape[0]
    f_dc = np32(ck["features_albedo"])                                    # [N, 3]
    spec = np32(ck["features_specular"])                                  # [N, K*3] interleaved
    f_rest = spec.reshape(n, -1, 3).transpose(0, 2, 1).reshape(n, -1)    # → channel-major
    log_s = np32(ck["scale"])                                             # [N, 3] log scale
    logit_o = np32(ck["density"]).reshape(n, 1)                           # [N, 1] opacity logit
    rot = np32(ck["rotation"])                                            # [N, 4] wxyz

    f3d = ck.get("filter_3D")
    if f3d is not None:
        # HACK: stock 3DGS viewers have no Mip filter, so bake GOFish's view-independent 3D
        # filter in Mip-Splatting style: s' = sqrt(s² + f²), opacity scaled by Π s/s' to keep
        # each Gaussian's mass. Its per-view footprint term (render-time) is dropped. Real fix:
        # view through GOFish's own renderer, or a viewer that applies the filter at draw time.
        s = np.exp(log_s)
        f = np32(f3d).reshape(n, 1)
        s2 = np.sqrt(s * s + f * f)
        o = 1.0 / (1.0 + np.exp(-logit_o)) * np.prod(s / s2, axis=1, keepdims=True)
        o = np.clip(o, 1e-6, 1 - 1e-6)
        log_s, logit_o = np.log(s2), np.log(o / (1 - o))

    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([xyz, np.zeros_like(xyz), f_dc, f_rest, logit_o, log_s, rot],
                          axis=1).astype("<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n" + "".join(f"property float {k}\n" for k in names)
              + "end_header\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(header.encode())
        fh.write(data.tobytes())
    print(f"  ✓ {n:,} gaussians → {args.out}")


if __name__ == "__main__":
    main()
