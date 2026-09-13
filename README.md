# DroneSplat — Gaussian Splatting from the Antigravity A1

Turn Antigravity A1 360° drone footage into 3D Gaussian Splats — **the proper way:
fisheye-native, zero resampling.** Fully open-source (ffmpeg + COLMAP + 3DGUT). No SDK,
no cloud, no undistortion loss.

```
.insv → extract fisheye → calibrate → reproject→perspective (SfM proxy) → COLMAP PINHOLE
      → transplant poses → TRAIN on RAW fisheye → .ply → splat.pow

      train = 3DGUT (volumetric)  OR  gsplat 2DGS (surfels / surface-splatting)
```

**Why split SfM from training** (validated on real A1 footage, CPU COLMAP): raw 196° fisheye
**won't register** in COLMAP — 3/118 frames. A perspective proxy registers **38/118 (6138 pts)**.
So pose on the proxy, then **train on the raw fisheye** (full resolution — only training quality
matters for detail; the proxy is throwaway geometry). Undistorting the *training* data would
dilute detail (DirectFisheye-GS 2026). Full rationale + numbers: **[docs/360-proper.md](docs/360-proper.md)**.

| Trainer (`TRAINER=`) | Output | Resolution | Runs on | Status |
|---|---|---|---|---|
| **`3dgut`** ⭐ | volumetric splats | full, zero resample | RTX 5090 (CUDA) | built — validates on the card |
| **`surfels`** | **2DGS surfels** (gsplat, fisheye-native) | full, zero resample | RTX 5090 (CUDA) | built — validates on the card |
| Brush (Mac) | volumetric splats | lossy (reproject needed) | Mac Studio, today | front-end ✅ proven on real footage |

Both CUDA trainers are **fisheye-native** (gsplat handles OpenCV fisheye + surfel mode in one
rasterizer). Surfel/surface-splatting details: **[docs/surface-splatting.md](docs/surface-splatting.md)**.

> Brush is pinhole-only, so the Mac path *must* reproject (lossy) — there is no
> resolution-preserving Mac route. Proper-360 needs the GPU. Use the Mac path for a fast
> look before the 5090 lands, not for final quality.

## Layout
```
pipeline/                 footage → splat (the work)
├── extract_fisheye.py    .insv → raw dual-fisheye frames (ffmpeg, no SDK)   ✅ real-data tested
├── calibrate.py          auto KB-fisheye intrinsics (factory parse + nominal) ✅ real-data tested
├── makesplat             Mac quick-test orchestrator (lossy; needs reproject step)
└── equirect/             alternate equirectangular route (not the proper path)
    ├── stitch_a1.py      Insta360-SDK byte-patch → stitched equirect
    └── prep360.py        equirect → 18 perspective views
docker/                   PROPER 5090 pipeline (fisheye-native)
├── Dockerfile            CUDA 12.8 + 3dgrut + COLMAP + front-end
└── run.sh                extract → COLMAP fisheye → 3DGUT
docker-compose.yml        builds/runs the proper pipeline
viewer/                   splat.pow web viewer (self-contained Astro app)  ✅ live
docs/
├── 360-proper.md         THE decision: fisheye-native, no resample
├── a1-pipeline.md        field intel: file format, SDK-bypass, capture rules
├── surface-splatting.md  mesh/collision track (MILo / 2DGS / 3DGUT)
└── training-method.md    background + borrowed techniques
captures/  exports/       data in/out (gitignored)
```

## Run it — proper pipeline (RTX 5090 / Linux + CUDA)
```bash
docker compose build
docker compose run --rm splat /work/in/FLIGHT.insv                      # volumetric (3DGUT)
docker compose run --rm -e TRAINER=surfels splat /work/in/FLIGHT.insv   # surfels (gsplat 2DGS)
```
> ⚠️ Not yet executed — no 5090 in hand. Built correct-by-construction; the sm_120 kernel
> build of `3dgrut` is the main first-run risk (see `docker/Dockerfile`).

## Run it — CPU path (works today, no GPU) ✅
```bash
pipeline/makesplat captures/raw/FLIGHT.insv            # → exports/FLIGHT.ply
```
`extract fisheye → reproject perspective proxy → COLMAP → OpenSplat (CPU) → .ply`.
**Validated end-to-end on real A1 footage** — produced a real splat 100% on CPU.
Lossy + slow (OpenSplat is pinhole → trains on the proxy; ~100× a GPU), but proves every
stage. Prereqs: `ffmpeg`, `colmap`, OpenSplat built at `/opt/OpenSplat/build/opensplat`,
`numpy opencv-python`. Quality/speed → the GPU pipeline above.

## The A1 (confirmed specs)
- `.insv` = MP4 + Insta360 trailer. **Two 3840×3840 HEVC streams** (one per lens), ~196° FoV.
- `extract_fisheye.py` pulls both lenses via ffmpeg — **no SDK** (the trailer is just metadata).
- `calibrate.py` reads factory calibration from the trailer + a nominal init (3840 px / 196°
  → focal ≈ 1272 px); COLMAP refines it during bundle adjustment. No checkerboard.

## Capture rules (what makes a good A1 splat)
- ✅ **Orbit** a target; multi-altitude orbits + figure-8. ❌ never hover or linear-flyover.
- Clips short; clear weather, low wind.
- Obstacle avoidance keeps ~5–7 m off objects; signal drops near ~700 m.
- "Don't walk; orbit." Trajectory (parallax) matters, not orientation.

## Viewer
`splat.pow` (Astro + nginx behind Traefik) renders any `.ply`/`.splat`. Live now with a demo
splat; drop a pipeline output into `viewer/public/` to view your own.

## Web / VR output
`.ply` → SuperSplat cleanup → `.sog`/`.spz` (~90% smaller). Pipeline auto-emits `.sog` if
`splat-transform` is present.
