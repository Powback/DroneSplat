#!/usr/bin/env bash
# run.sh — PROPER 360 pipeline (in-container, RTX 5090).
#   .insv → extract fisheye → [perspective-proxy SfM] → train on RAW fisheye → .ply
# Training is fisheye-native (no resample). SfM uses a perspective proxy because raw
# 196° fisheye does NOT register in COLMAP (validated: 3/118 vs 38/118). See docs/360-proper.md.
set -euo pipefail

IN="${1:-}"; [[ -z "$IN" ]] && { echo "usage: <compose run> splat /work/in/<file.insv>"; exit 1; }
NAME="$(basename "${IN%.*}")"
FISH="/work/tmp/${NAME}_fish"; SCENE="/work/tmp/${NAME}_scene"; OUT="/work/out"
FPS="${FPS:-2}"; FOV="${SFM_FOV:-110}"
mkdir -p "$FISH" "$SCENE/images" "$OUT"
DS=/opt/DroneSplat

echo "▶ 1/5  extract raw dual-fisheye (ffmpeg, no SDK)"
python "$DS/extract_fisheye.py" "$IN" --out "$FISH" --fps "$FPS"
cp "$FISH"/*.jpg "$SCENE/images/"          # RAW fisheye = training images

echo "▶ 2/5  auto-calibrate (KB fisheye)"
python "$DS/calibrate.py" --insv "$IN" --out "$SCENE/cam_init.json"

echo "▶ 3/5  reproject fisheye → ${FOV}° perspective proxy (SfM only)"
FOCAL="$(python -c "import json;print(json.load(open('$SCENE/cam_init.json'))['fx'])")"
SIZE="$(python -c "import json;print(json.load(open('$SCENE/cam_init.json'))['width'])")"
python "$DS/reproject_sfm.py" --in "$SCENE/images" --out "$SCENE/proxy" \
  --fisheye-size "$SIZE" --fisheye-focal "$FOCAL" --fov "$FOV"
PFOCAL="$(python -c "import math;print(($SIZE/2)/math.tan(math.radians($FOV)/2))")"

POSER="${POSER:-colmap}"   # colmap (CPU, ~32% on wide-FoV) | vggt (GPU, robust — pushes registration up)
echo "▶ 4/5  pose proxy  [POSER=$POSER] → transplant onto fisheye"
mkdir -p "$SCENE/proxy_sparse"
case "$POSER" in
  colmap)
    # Classical SfM. Caps low on wide-baseline drone footage (validated). CPU-OK.
    DB="$SCENE/db.db"
    colmap feature_extractor --database_path "$DB" --image_path "$SCENE/proxy" \
      --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
      --ImageReader.camera_params "$PFOCAL,$PFOCAL,$((SIZE/2)),$((SIZE/2))" --SiftExtraction.use_gpu 0
    colmap exhaustive_matcher --database_path "$DB" --SiftMatching.use_gpu 0
    colmap mapper --database_path "$DB" --image_path "$SCENE/proxy" --output_path "$SCENE/proxy_sparse"
    ;;
  vggt)
    # Learning-based, pose-free — robust where COLMAP fails (wide baseline / FoV).
    # VGGT's demo_colmap.py exports a COLMAP-format model from the proxy images.
    # CUDA only. Verify flags against the vggt repo on first run.
    cd /opt/vggt
    python demo_colmap.py --scene_dir "$SCENE/proxy" --output_dir "$SCENE/proxy_sparse/0" \
      || { echo "⚠️ vggt demo_colmap CLI differs — see repo, update run.sh"; exit 1; }
    ;;
  *) echo "✗ unknown POSER: $POSER (use colmap | vggt)"; exit 1 ;;
esac
[[ -d "$SCENE/proxy_sparse/0" ]] || { echo "✗ proxy pose step found no model — check coverage (orbit, not hover)"; exit 1; }
# transplant: proxy poses + points3D → OPENCV_FISHEYE model over the raw fisheye frames
python "$DS/transplant_poses.py" --proxy "$SCENE/proxy_sparse/0" --rig "$SCENE/proxy/rig.json" \
  --cam "$SCENE/cam_init.json" --out "$SCENE/sparse/0"

TRAINER="${TRAINER:-3dgut}"   # 3dgut = volumetric splats | surfels = gsplat 2DGS surface-splatting
echo "▶ 5/5  training  [TRAINER=$TRAINER · fisheye-native, raw frames, no resample]"
case "$TRAINER" in
  3dgut)
    # 3dgrut: volumetric 3DGS, consumes the COLMAP fisheye scene directly.
    # Confirm entry point/config against the 3dgrut README on first run.
    cd /opt/3dgrut
    python train.py --config-name apps/colmap_3dgut.yaml \
      path="$SCENE" out_dir="$SCENE/out" experiment_name="$NAME" \
      || { echo "⚠️ 3dgrut CLI/config differs in your checkout — see README, update run.sh"; exit 1; }
    ;;
  surfels)
    # gsplat 2DGS: SURFEL / surface-splatting (flat oriented disks) with the fisheye
    # camera model — distortion handled natively (k1–k4), surfels via 2D-eval+normals.
    # Verify the 2dgs subcommand / --camera-model flag against gsplat examples on first run.
    cd /opt/gsplat/examples
    python simple_trainer.py 2dgs \
      --data-dir "$SCENE" --data-factor 1 \
      --camera-model fisheye \
      --result-dir "$SCENE/out" \
      || { echo "⚠️ gsplat 2DGS CLI differs in your checkout — see examples/simple_trainer.py, update run.sh"; exit 1; }
    ;;
  *) echo "✗ unknown TRAINER: $TRAINER (use 3dgut | surfels)"; exit 1 ;;
esac

PLY="$(find "$SCENE/out" -name '*.ply' | sort | tail -1)"
[[ -n "$PLY" ]] && cp "$PLY" "$OUT/${NAME}.ply" && echo "  ✓ $OUT/${NAME}.ply"
command -v splat-transform >/dev/null && [[ -f "$OUT/${NAME}.ply" ]] && \
  splat-transform "$OUT/${NAME}.ply" "$OUT/${NAME}.sog" && echo "  ✓ $OUT/${NAME}.sog"
echo "✓ done → exports/${NAME}.ply  (fisheye-native, zero resample)"
