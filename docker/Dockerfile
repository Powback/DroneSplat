# DroneSplat — PROPER 360 pipeline for CUDA: RTX 5090 (sm_120) or RTX 3090 on gpu-195 (sm_86).
# Fisheye-NATIVE: raw dual-fisheye → COLMAP fisheye SfM → 3DGUT training. NO undistort,
# NO reproject, NO resample — every native pixel used (see docs/360-proper.md).
# 3DGUT (nv-tlabs/3dgrut) is distorted-camera-native AND the basis of the mesh/surfels track.
#
# ⚠️ NOT yet executed — no 5090 in the build env. Validate on first run on the card.
#    These are custom CUDA kernels; the sm_120 build is the main risk (notes below).

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
# Kernel targets: 8.6 = RTX 3090 (gpu-195, Ampere), 12.0 = RTX 5090 (Blackwell). One image
# runs on both; narrow it (--build-arg CUDA_ARCHS=8.6) for faster builds. No inline
# comments on ENV lines: Docker parses them as extra key=value pairs and fails the build.
ARG CUDA_ARCHS="8.6;12.0"
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}"
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

# ── SAM 3 — text-prompted segmentation for people masks (MASK_PEOPLE=1) ───────
# Latest Segment Anything as of 2026-11 (3.1 only changed video tracking). Needs py3.12 +
# torch ≥2.7 — the system env above. Weights are gated (facebook/sam3 on Hugging Face):
# fetched at run time with HF_TOKEN into the persisted /root/.cache, never baked in.
WORKDIR /opt
RUN git clone https://github.com/facebookresearch/sam3.git \
 && pip install --break-system-packages -e sam3 || \
    echo "⚠️ sam3 install needs a look — MASKER=maskrcnn still works (torchvision)"

# ── GOFish — full-FoV (>180°) fisheye GS + aligned mesh (TRAINER=gofish) ──────
# Own venv: GOFish vendors its own `threedgrut` package (would shadow /opt/3dgrut above)
# and targets Python 3.11 (image ships 3.12) → uv-managed 3.11 venv at /opt/gofish-venv.
# Pinned to the commit run.sh was written against; bump deliberately.
ARG GOFISH_REF=1a33e7a9ff73a58aefced35f23f320be3b7b76ac
ARG SLANG_VER=2026.5.2
RUN apt-get update && apt-get install -y --no-install-recommends libgmp-dev libcgal-dev \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages uv \
 && uv venv -p 3.11 /opt/gofish-venv
ENV GOFISH_PY=/opt/gofish-venv/bin/python
WORKDIR /opt
RUN git clone --recursive https://github.com/tuallen/GOFish.git gofish \
 && git -C gofish checkout "$GOFISH_REF" && git -C gofish submodule update --init --recursive
WORKDIR /opt/gofish
RUN uv pip install --python "$GOFISH_PY" torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
 && uv pip install --python "$GOFISH_PY" -e ".[eval]" \
 && uv pip install --python "$GOFISH_PY" --no-build-isolation -r requirements_extra.txt \
 || echo "⚠️ GOFish install needs a look on the 5090 — see its README (CUDA ext build for sm_120)"
# Slang compiler — GOFish's renderer compiles Slang kernels (on first run).
RUN mkdir -p /opt/slang \
 && curl -L "https://github.com/shader-slang/slang/releases/download/v${SLANG_VER}/slang-${SLANG_VER}-linux-x86_64.tar.gz" \
    | tar -xz -C /opt/slang
ENV PATH="/opt/slang/bin:${PATH}"
# Mesh extraction: GOF's tetra-triangulation (CGAL), built against the venv's torch.
RUN git clone https://github.com/autonomousvision/gaussian-opacity-fields /opt/gaussian-opacity-fields \
 && git -C /opt/gaussian-opacity-fields checkout 5245b20e5d11acd6d1ff5af4b890dc2bedd99693 \
 && cd /opt/gaussian-opacity-fields/submodules/tetra-triangulation \
 && sed -i 's/_GLIBCXX_USE_CXX11_ABI=0/_GLIBCXX_USE_CXX11_ABI=1/' cmake/FindTorch.cmake \
 && CPATH=/usr/local/cuda/include cmake \
      -DCMAKE_PREFIX_PATH="$($GOFISH_PY -c 'import torch; print(torch.utils.cmake_prefix_path)')" . \
 && make && uv pip install --python "$GOFISH_PY" -e . \
 || echo "⚠️ tetra-triangulation build failed — GOFISH_MESH=0 still trains; fix for meshes"

# ── Web/VR compression ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
 && npm i -g @playcanvas/splat-transform || true \
 && rm -rf /var/lib/apt/lists/*

# ── Project entrypoint ────────────────────────────────────────────────────────
WORKDIR /work
COPY pipeline/extract_fisheye.py pipeline/calibrate.py \
     pipeline/reproject_sfm.py pipeline/transplant_poses.py \
     pipeline/measure_aperture.py pipeline/gofish_to_ply.py pipeline/mask_people.py /opt/DroneSplat/
COPY docker/run.sh /opt/DroneSplat/run.sh
RUN chmod +x /opt/DroneSplat/run.sh
ENTRYPOINT ["/opt/DroneSplat/run.sh"]
