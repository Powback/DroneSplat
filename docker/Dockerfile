# DroneSplat — PROPER 360 pipeline for the RTX 5090 (Blackwell / sm_120).
# Fisheye-NATIVE: raw dual-fisheye → COLMAP fisheye SfM → 3DGUT training. NO undistort,
# NO reproject, NO resample — every native pixel used (see docs/360-proper.md).
# 3DGUT (nv-tlabs/3dgrut) is distorted-camera-native AND the basis of the mesh/surfels track.
#
# ⚠️ NOT yet executed — no 5090 in the build env. Validate on first run on the card.
#    These are custom CUDA kernels; the sm_120 build is the main risk (notes below).

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0"          # Blackwell sm_120 — every kernel must target this
ENV PIP_NO_CACHE_DIR=1

# ── System deps (+ COLMAP for fisheye SfM, ffmpeg for extraction) ─────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg cmake ninja-build build-essential \
    python3 python3-pip python3-venv python3-dev \
    colmap \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── PyTorch cu128 (sm_120 kernels for Blackwell) ──────────────────────────────
RUN pip install --break-system-packages --upgrade pip \
 && pip install --break-system-packages torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128

# ── Front-end deps (prep / extract / calibrate) ───────────────────────────────
RUN pip install --break-system-packages numpy opencv-python-headless

# ── 3DGUT / 3dgrut — distorted-camera-native rasterizer (build for sm_120) ─────
WORKDIR /opt
RUN git clone --recursive https://github.com/nv-tlabs/3dgrut.git
WORKDIR /opt/3dgrut
# The repo's installer builds the CUDA kernels. TORCH_CUDA_ARCH_LIST=12.0 (above) makes
# them target Blackwell. If the build pins an older arch, patch the setup/cmake arch list.
# Verify against the repo's install_env_uv.sh / README on first build.
RUN pip install --break-system-packages -e . || \
    bash install_env_uv.sh || \
    echo "⚠️ 3dgrut build needs a look on the 5090 — see its README, patch sm_120, rebuild"

# ── gsplat — fisheye-native rasterizer with 2DGS SURFEL mode (the surfel track) ─
# gsplat natively handles OpenCV fisheye distortion (k1–k4) AND 2D/surfel rendering
# (with_eval3d + return_normals). One rasterizer = fisheye-native surface splatting.
WORKDIR /opt
RUN git clone https://github.com/nerfstudio-project/gsplat.git
WORKDIR /opt/gsplat
RUN pip install --break-system-packages -e . || \
    echo "⚠️ gsplat kernels need sm_120 build on the 5090 — TORCH_CUDA_ARCH_LIST=12.0, rebuild"
RUN pip install --break-system-packages -r examples/requirements.txt || true

# ── VGGT — learning-based pose-free SfM (POSER=vggt), robust where COLMAP fails ─
# Runs on the perspective proxy (VGGT is pinhole-trained). Far higher registration
# than COLMAP on wide-baseline / wide-FoV drone footage. CUDA only.
WORKDIR /opt
RUN git clone https://github.com/facebookresearch/vggt.git \
 && pip install --break-system-packages -e vggt || \
    echo "⚠️ vggt install needs a look on the 5090 — see repo README"

# ── Web/VR compression ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
 && npm i -g @playcanvas/splat-transform || true \
 && rm -rf /var/lib/apt/lists/*

# ── Project entrypoint ────────────────────────────────────────────────────────
WORKDIR /work
COPY pipeline/extract_fisheye.py pipeline/calibrate.py \
     pipeline/reproject_sfm.py pipeline/transplant_poses.py /opt/DroneSplat/
COPY docker/run.sh /opt/DroneSplat/run.sh
RUN chmod +x /opt/DroneSplat/run.sh
ENTRYPOINT ["/opt/DroneSplat/run.sh"]
