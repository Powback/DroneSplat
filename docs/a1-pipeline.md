# A1 → splat: real-world intel (from the field, not theory)

Source: u/TomMooreJD's writeup + the expert pushback in the comments
(r/AntigravityA1/comments/1t2ok7q, r/GaussianSplatting). The most battle-tested
A1 knowledge we have. Read before touching the pipeline.

## Two ways to get frames out of an .insv

| | **A. Raw fisheye (DEFAULT — fully open)** | **B. Stitched equirect (proprietary)** |
|---|---|---|
| Tools | ffmpeg + OpenCV + COLMAP | Insta360 MediaSDK (apply for access) + byte-patch |
| Open source? | ✅ 100% | ❌ needs the proprietary `.deb` |
| Geometry | ✅ accurate (calibrated lenses) | ⚠️ cosmetic stitch, seam artifacts |
| Cost | one-time lens calibration (checkerboard) | SDK application (~12h), per-file byte-patch |
| Our scripts | `extract_fisheye.py` | `stitch_a1.py` |

**Use A.** The experts in the thread agree it's both open *and* better. B is only a
lazy convenience if you specifically want pre-stitched equirect and don't mind the SDK.

### A. Raw fisheye — the open path
```
.insv → ffmpeg extract dual-fisheye frames (extract_fisheye.py, NO SDK)
      → split L/R lens (if single side-by-side track)
      → OpenCV fisheye calibration (cv2.fisheye, Kannala-Brandt ≈ equisolid), one-time/lens
      → COLMAP with OPENCV_FISHEYE intrinsics  (or undistort → perspective pinhole)
      → Brush / InstantSplat → .ply
```
- `.insv` is a plain MP4; ffmpeg reads the standard-codec video frames and ignores the
  proprietary trailer. No SDK touches the pixels.
- Calibrate once per physical lens; reuse the intrinsics for every later flight.
- COLMAP natively supports the fisheye (Kannala-Brandt) camera model — feed frames directly,
  no stitch, no equirect distortion.

### B. (optional, proprietary) Stitched equirect via the SDK
- The public **Insta360 MediaSDK rejects A1 files**. A1 reports **lens type 155**
  (video) / **112** (photo); the SDK dispatcher only knows X3/X4/X5/etc. → no batch
  stitching, GUI-only, one file at a time. This blocks ALL automation.
- **Fix (our `stitch_a1.py`):** A1 optics ≈ Insta360 **X4**. Byte-patch the lens type
  `155 → 113` (X4 video) in the `.insv` — 4 occurrences in the trailer, same length so
  MP4 offsets survive. SDK then stitches it cleanly (A1's real per-unit calibration
  floats are in the same string and stay correct).
- **Orientation:** drone is mounted upside-down vs X4 handheld → post-stitch
  `ffmpeg -vf "vflip,hflip"`.
- **Photos (.insp):** rename `.insp → .insv` (routes through the looser video
  stitcher), apply the same 155→113 patch, extract the JPG with ffmpeg.
- You still need MediaSDK access (free dev application at insta360.com/sdk, ~12h).
  It runs as x86_64 Linux in Docker (Rosetta on Mac, no GPU).

### A1 file format (CONFIRMED — Rodrigo Polo's analysis of a real file)
- `.insv` = MP4 (`ftyp … mdat … moov`) + proprietary Insta360 trailer. UUID
  `8db42d694ccc418790edff439fe026bf` (shared across X3/X4/X5/A1).
- **Two video streams, 3840×3840 each**, H.265/HEVC, 29.97 fps, ~90 Mbps/stream.
  → confirms `extract_fisheye.py`'s **two-track** branch is correct (one lens per track).
- **Lens FoV ≈ 196°** (≈16° edge overlap for stitching). Stitched equirect = 7680×3840.
- Sensor 10496×5248 (full); the *encoded video* is 3840² per lens → `calibrate.py` nominal
  init uses **3840 px / 196°** (focal ≈ 1272 px). Trailer offset string still keys on the
  `10496_5248_<lens_type>` sensor dims for the byte-patch / factory-cal parse.
- ⚠️ `moov` sits **after** the huge `mdat` → a truncated download (common with anonymous
  Google-Drive large-file fetches) loses `moov` and ffmpeg reports "moov atom not found".
  Always verify `mdat` declared size vs file size before processing.

## The proven Mac-native pipeline (runs TODAY, no 5090)
```
.insv/.insp → stitch_a1.py patch → MediaSDK (Docker) → equirect mp4
            → ffmpeg vflip,hflip
            → ffmpeg cubemap (v360=e:flat:h_fov=90:v_fov=90, 6 faces, 1536²)
            → COLMAP automatic_reconstructor (SIMPLE_PINHOLE, f=cx=cy=768; ~30min CPU)
            → Brush (Apple Silicon / WGPU-Metal, ~30k steps ~30min)
            → .ply
```
All open-source, Mac-native or Mac-via-Docker. No CUDA, no cloud.

## ⚠️ The equirectangular controversy (important)
The experienced A1 scanners in the thread pushed back on equirect:
- **"Equirectangular is a mistake. The stitches are designed to look good, not be
  mathematically accurate."** The cosmetic stitch warps geometry → hurts SfM, worst
  at the **seam** and **close-up/indoors**. The OP tested equirect vs fisheye and
  conceded equirect "sucked."
- **Geometrically-correct path:** work from the **raw dual-fisheye**, calibrate each
  lens with the **Kannala-Brandt model** (360 fisheyes are usually **equisolid**, not
  equidistant). Offline calibration: <0.2px error @4K (~100 shots of a checkerboard on
  a big TV, tripod). Online (during SLAM): ~0.8px. COLMAP supports fisheye/KB intrinsics.
  Bonus: with a one-time calibration you **bypass the SDK entirely** — just ffmpeg the
  raw frames. (Agisoft Metashape can also solve the fisheye intrinsics: set equisolid,
  align, export.)
- **BUT for far/open outdoor scenes + seam masking, equirect is fine** — the
  lens-separation parallax is negligible at distance.

### What this means for US (drone, outdoor game-level scenes)
Our scenes are **far/open** → equirect+cubemap is acceptable to start. Treat **raw
fisheye + Kannala-Brandt** as the **v2 quality upgrade** (and for any close/indoor work).

## Capture rules (what actually makes a good A1 splat)
360 sees every direction every frame, so **camera-body trajectory** is everything:
- ❌ Hover in one spot — no parallax
- ❌ Linear flyover — each point seen from too narrow an angle
- ✅ **Orbit a target** — convergent multi-view
- ✅ **Multi-altitude orbits + figure-8** — best for outdoor scenes / buildings
- "Don't walk; orbit." (A handheld room walk-through → "needle-storm.")

## Sources
- [r/AntigravityA1 writeup + byte-patch deep dive](https://www.reddit.com/r/AntigravityA1/comments/1t2ok7q/)
- [r/GaussianSplatting thread (fisheye-vs-equirect debate)](https://www.reddit.com/r/GaussianSplatting/comments/1t2otb7/)
- [packet39.com — A primer on Gaussian Splats](https://packet39.com/blog/a-primer-on-gaussian-splats/)
- [Brush (Apple Silicon trainer)](https://github.com/ArthurBrussee/brush)
