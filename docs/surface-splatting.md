# Surfel / surface-splatting track

Surface splatting represents the scene as **flat, oriented 2D gaussian disks (surfels)**
that lie on surfaces — instead of the volumetric ellipsoid "blobs" of 3DGS. Each surfel
is view-consistent (every ray samples the same planar gaussian), giving cleaner geometry,
normals, and depth. This is a *representation* choice; a mesh is an optional downstream
export, not the point.

## The key finding: gsplat does fisheye-native surfels
Earlier I thought fisheye-native and surfels couldn't coexist off-the-shelf. Wrong —
**[gsplat](https://github.com/nerfstudio-project/gsplat)** does both in one rasterizer:
- **Fisheye distortion native** — OpenCV fisheye model (k1–k4) directly in the rasterizer.
  No undistort, no reproject, no resample (the whole point — see [360-proper.md](360-proper.md)).
- **Surfel / 2DGS mode** — `with_eval3d` (evaluate the gaussian in 3D world space, i.e. as a
  flat disk) + `return_normals` (per-pixel normals). That *is* surface splatting.

So one rasterizer = **fisheye-native surface splatting**. No compromise on resolution.

## Wired in
`TRAINER=surfels` selects it in the pipeline (`docker/run.sh`):
```
extract_fisheye → calibrate (OPENCV_FISHEYE k1–k4) → COLMAP fisheye SfM
  → gsplat 2DGS (--camera-model fisheye)            → surfel .ply
```
```bash
docker compose run --rm -e TRAINER=surfels splat /work/in/FLIGHT.insv
```
- `TRAINER=3dgut` → volumetric splats (3DGUT). `TRAINER=surfels` → this surfel path.
- ⚠️ Built, **not run** — CUDA/5090 only. The gsplat `2dgs` subcommand / `--camera-model
  fisheye` flags are best-guess; verify against `examples/simple_trainer.py` on first run.

## Other surfel methods (for reference / fallback)
| Method | Fisheye? | Note |
|---|---|---|
| **gsplat 2DGS** ⭐ | ✅ native (k1–k4) | what we wired — the only fisheye-native surfel path |
| 2DGS (hbb1) | ❌ pinhole | the original surfel method; would need reproject |
| Gaussian Surfels (turandai) | ❌ pinhole | surfels + normal supervision; pinhole |
| 360-GS / Seam360GS | equirect | surface splatting *on equirect* (needs the stitch) |

## Optional: mesh from surfels (only if you later want collision geometry)
Surfels extract to a mesh cleanly (TSDF/Poisson on their normals+depth). Methods like
**MILo** wrap a surfel rasterizer for low-poly meshes. Not wired — a future stage if the
game-level collision need comes back. Don't confuse it with the surfel representation itself.

## Sources
- [gsplat (fisheye + 2DGS/normals)](https://github.com/nerfstudio-project/gsplat) ·
  [2DGS](https://github.com/hbb1/2d-gaussian-splatting) ·
  [Gaussian Surfels](https://github.com/turandai/gaussian-surfels) ·
  [Seam360GS (ICCV'25)](https://arxiv.org/pdf/2508.20080)
