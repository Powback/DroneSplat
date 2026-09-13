# DroneSplat — method & design decisions (A1, 2026 stack)

What the pipeline does and *why*, distilled from how the SOTA tools work. We borrow
their techniques; we don't depend on the tools themselves.

## Settled architecture
```
A1 360° video
  → prep360.py        extract frames · sharpness filter · reproject → 18 perspective views (+ known intrinsics)
  → VGGT              pose-free, COLMAP-free: cameras + dense point cloud in one forward pass
  → trainer           3DGS training (MCMC + appearance modeling)
  → .ply → .sog       cleanup + web/VR compression
```
- **No COLMAP.** Replaced by **VGGT** (feed-forward 3D foundation model; beats DUSt3R/MASt3R,
  processes many images at once). Poses are solved automatically — nothing is hand-posed.
- **No masking.** The A1 is a *drone*: no operator, arm, selfie-stick, or stitch seam in frame.
  The whole operator-removal stage that handheld-360 pipelines need simply doesn't apply.
- **Reproject before posing.** VGGT and the 3DGS rasterizer are perspective-native; raw
  equirectangular makes giant distorted gaussians. So unwrap the sphere to flat views first.

## Techniques borrowed from the top-of-the-line tools
(We studied how these work and copied the ideas, not the software.)

| Source | Technique | How we apply it |
|---|---|---|
| **RealityScan / Pano2Views** | Reproject equirect → rectilinear faces and **write KNOWN intrinsics** (focal from FoV) so the solver skips undistortion & doesn't re-solve intrinsics | `prep360.py` emits `intrinsics` (PINHOLE, exact focal, undistorted=true) in `prep360.json` |
| **RealityScan** | Default 90° faces, **widen to 120° for overlap** so low-texture regions (wall vs sky) co-register | `--fov` + ring/pitch knobs; overlap is tunable |
| **The "18-cam" 360 rigs** | 8 yaw × 2 pitch (±35°) ring **+ zenith + nadir** — full-sphere coverage with overlap, polar cams where ERP distortion is worst | default camera layout in `virtual_cameras()` |
| **LichtFeld Studio** | **MCMC densification** (better convergence, fewer floaters) | prefer an MCMC-capable trainer |
| **LichtFeld Studio** | **Bilateral-grid appearance modeling** — handles per-frame exposure swings | important for aerial (auto-exposure as drone flies sun→shade) |
| **VGGT / InstantSplat** | SfM-free init from a 3D foundation model | VGGT as the pose/geometry step |

## Capture (unchanged, still matters most)
- Plan the orbit + altitude layers; avoid redundant passes.
- Clips **< 5 min**.
- 4K out-of-box fine. Obstacle avoidance keeps ~5–7 m off objects; signal drops ~700 m.
- Garbage in → garbage out: smooth, well-exposed, well-covered footage beats any trainer setting.

## Hardware reality
- The training stack is **CUDA-only** (VGGT, gsplat/rasterizer kernels).
- **Docker can't use the Mac's Metal GPU** → the Compose stack targets the **5090 / Linux**.
- 5090 = Blackwell **sm_120**, needs **CUDA 12.8+ / PyTorch cu128**. (LichtFeld is already CUDA-12.8
  native, which is why CUDA 12.8 is our base image — it makes Blackwell painless.)
- On the Mac (no Docker GPU), the fallback is the native **Brush** (wgpu/Metal) trainer.

## Open frontier (not in the default path)
- **Splatter-360 / 360-GeoGS** — feed-forward models that train *directly* on equirectangular
  (skip reprojection). Cutting-edge; some have non-commercial weights. Worth revisiting later.

## Sources
- [VGGT (feed-forward 3D foundation model)](https://github.com/facebookresearch/vggt)
- [InstantSplat — SfM-free GS in seconds (VGGT/MASt3R init)](https://github.com/NVlabs/InstantSplat)
- [LichtFeld Studio (open CUDA 12.8 trainer, MCMC, appearance modeling)](https://github.com/MrNeRF/LichtFeld-Studio)
- [RealityScan 2.2 + Pano2Views (360 → cubemap + known-intrinsics XMP)](https://radiancefields.com/realityscan-2.2-adds-360-and-amd-gpu-support)
- [Gesplat / VGGT-X / AnySplat (VGGT-based pose-free GS)](https://dekuliutesla.github.io/vggt-x.github.io/)
