# The PROPER way to do 360 → Gaussian Splatting (no resolution loss)

The core decision, settled. TL;DR: **never undistort. Use a native rasterizer.**

## Why reproject/undistort is the WRONG way (confirmed, not opinion)
Warping fisheye → pinhole (cubemap / perspective crops / equirect→perspective) is what most
casual pipelines do (incl. our `makesplat` and the Reddit A1 guy). The 2026 literature is
explicit that it destroys detail:
> "Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger
> area, **diluting detail density — causing 3DGS to overfit low-frequency zones, producing
> blur and floating artifacts**" + black borders that "negate the fisheye's large-FOV
> advantage." — *DirectFisheye-GS (2026)*

Also: a ≥180° fisheye **cannot** be undistorted to a single pinhole (infinite size at 180°),
so you're forced into multiple crops, each resampled. Lossy twice over.

## The proper way: native rasterizers (zero resample)
Modify the GS rasterizer's projection math to consume the 360/fisheye directly. Every native
sensor pixel is used; no warp, no seam, no black borders. Papers show native ERP/fisheye
**beats** the reproject (MVG) approach in PSNR/SSIM "even without explicit distortion correction."

### Fisheye-native — BEST for the A1 (it shoots raw dual-fisheye)
| Method | What | Note |
|---|---|---|
| **DirectFisheye-GS** (2026) ⭐ | Kannala-Brandt rasterization in 3DGS + **cross-view joint optimization** | current best; fixes boundary artifacts of Fisheye-GS; native fisheye in, no preprocessing |
| **Fisheye-GS** (2024) | equidistant projection + Jacobian; lightweight module, bolts onto **gsplat/FlashGS** | proven, simple, the baseline |
| **3DGUT** (NVIDIA, nv-tlabs/3dgrut) | Unscented-Transform rasterization, handles **any distorted camera** | also underpins the surfels/mesh track → unifying choice |

### ERP-native — if you ever stitch to equirect
| Method | What |
|---|---|
| **OmniGS** (2024) | derives spherical-camera derivatives, splats directly onto the equirect plane (no cube-map / tangent-plane) — the foundational 360 rasterizer |
| **Seam360GS** (ICCV 2025) | builds on OmniGS; specifically handles the dual-fisheye **seam** |

## ⚠️ Validated reality: SfM ≠ training (tested on real A1 footage, CPU COLMAP)
You **cannot** pose raw 196° fisheye in COLMAP — measured:

| SfM input | Registered | Points |
|---|---|---|
| raw fisheye, 20 frames | 2/20 | 53 |
| raw fisheye, 118 frames | 3/118 | 95 |
| **perspective proxy (110°), 118 frames** | **38/118** | **6,138** |

So the resolution rule applies to **training**, not pose-solving. Split them:
- **Pose:** reproject fisheye → a 110° perspective proxy (`reproject_sfm.py`), COLMAP **PINHOLE**.
  The proxy is centered on the optical axis → the pose (R,t) applies directly to the raw frame.
- **Train:** raw full-res fisheye + those poses, fisheye-native rasterizer. **Zero resolution loss**
  where it matters (training); the proxy is throwaway geometry scaffolding.

## Poser choice (`POSER=`) — COLMAP isn't enough for 100%
Validated on CPU: COLMAP on the perspective proxy caps **~32%** registration (38/118 dense
single-view; wide baselines / wide FoV are its weakness). Pushing higher with COLMAP means
dense×multi-view = impractical on CPU. **The fix is a learning poser**, not COLMAP tuning:

| `POSER=` | Engine | Input | Registration | Runs |
|---|---|---|---|---|
| `colmap` | classical SfM | perspective proxy | ~32% (validated) | CPU ✅ |
| `vggt` | VGGT feed-forward, pose-free | perspective proxy | high (robust to wide baseline) | CUDA |
| *(future)* `mast3r` | MASt3R | perspective proxy | high | CUDA |
| *(future)* `panovggt` | **PanoVGGT** — native 360 | **raw fisheye, no proxy** | high | CUDA |

VGGT/MASt3R are pinhole-trained → still use the proxy. **PanoVGGT** (arXiv Mar 2026) is built
for panoramic input → could skip the proxy entirely. multi-view proxy code (`reproject_sfm.py`,
5 views/frame) is ready to feed whichever poser.

## Decision for DroneSplat
- **Pipeline:** `extract_fisheye` → `reproject_sfm` (multi-view proxy) → **POSER** (colmap CPU /
  vggt GPU) → `transplant_poses` → **TRAINER** (3dgut / gsplat-2dgs surfels) on the raw fisheye.
  Training never resamples. Poser is the lever for registration%.
- Validated on CPU ✅: extract, calibrate, reproject, COLMAP SfM, transplant. GPU stages
  (vggt poser, 3dgut/gsplat training): built, not yet run — need the 5090.
- **Compromise path (today, lossy):** fisheye → perspective crops → **Brush** (Metal). Brush is
  **pinhole-only**, so there is *no* resolution-preserving Mac route — native-fisheye GS needs
  the GPU. The Mac path is for quick tests before the 5090 lands, not final quality.
- **3DGUT is the strategic pick:** distorted-camera-native AND the basis of the surface/mesh
  (surfels) track — one rasterizer family covers proper-360 *and* collision meshes.

## ⚠️ 5090 build note
All native rasterizers are custom CUDA extensions pinned to old CUDA (11.8). Blackwell sm_120
needs CUDA 12.8 + torch cu128 + `TORCH_CUDA_ARCH_LIST="12.0"`, patching each `setup.py`.
(Same concern as docs/surface-splatting.md.)

## Sources
- [DirectFisheye-GS (2026)](https://arxiv.org/abs/2604.00648) · [Fisheye-GS (2024)](https://arxiv.org/abs/2409.04751)
- [OmniGS (2024)](https://arxiv.org/abs/2404.03202) · [Seam360GS (ICCV 2025)](https://arxiv.org/pdf/2508.20080)
- [3DGUT / 3dgrut (NVIDIA)](https://github.com/nv-tlabs/3dgrut)
