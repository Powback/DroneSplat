#!/usr/bin/env bash
# run.sh — PROPER 360 pipeline (in-container, RTX 5090).
#   .insv → extract fisheye → [perspective-proxy SfM] → train on RAW fisheye → .ply
# Training is fisheye-native (no resample). SfM uses a perspective proxy because raw
# 196° fisheye does NOT register in COLMAP (validated: 3/118 vs 38/118). See docs/360-proper.md.
set -euo pipefail

IN="${1:-}"; [[ -z "$IN" ]] && { echo "usage: <compose run> splat /work/in/<file.insv|.insp|flight-folder>"; exit 1; }
NAME="$(basename "${IN%.*}")"
FISH="/work/tmp/${NAME}_fish"; SCENE="/work/tmp/${NAME}_scene"; OUT="/work/out"
FPS="${FPS:-2}"; FOV="${SFM_FOV:-110}"
mkdir -p "$FISH" "$SCENE/images" "$OUT"
DS=/opt/DroneSplat

echo "▶ 1/5  extract raw per-lens fisheye (ffmpeg, no SDK; .insv / .insp / flight folder)"
python "$DS/extract_fisheye.py" "$IN" --out "$FISH" --fps "$FPS"
cp "$FISH"/*.jpg "$SCENE/images/"          # RAW fisheye = training images

echo "▶ 2/5  per-lens calibration (measured image circle → KB equisolid)"
python "$DS/measure_aperture.py" --in "$SCENE/images" --out "$SCENE/apertures.json"
INSV=(); [[ -f "$IN" && "${IN,,}" == *.insv ]] && INSV=(--insv "$IN")
python "$DS/calibrate.py" --apertures "$SCENE/apertures.json" "${INSV[@]}" --out "$SCENE/cam_init.json"

echo "▶ 3/5  reproject fisheye → ${FOV}° perspective proxy (SfM only)"
python "$DS/reproject_sfm.py" --in "$SCENE/images" --out "$SCENE/proxy" --cam "$SCENE/cam_init.json" --fov "$FOV"
# Proxy camera = what reproject_sfm actually wrote (rig.json), not the fisheye size.
read -r PFOCAL PSIZE < <(python -c "import json;r=json.load(open('$SCENE/proxy/rig.json'));print(r['pfocal'],r['out_size'])")
SIZE="$PSIZE"

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

TRAINER="${TRAINER:-3dgut}"   # 3dgut = volumetric splats | surfels = gsplat 2DGS surface-splatting | gofish = full-FoV GS + mesh
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
  gofish)
    # GOFish: full-FoV (>180°) fisheye GS + aligned collision mesh (docs/360-proper.md).
    # Same COLMAP OPENCV_FISHEYE scene; FIORD preset = Insta360 dual 200° fisheye, the
    # closest match to the A1. Lives in its own venv (see Dockerfile).
    GF=/opt/gofish; GPY="${GOFISH_PY:-/opt/gofish-venv/bin/python}"
    GDS="${GOFISH_DOWNSAMPLE:-1}"     # 1 = full native res (our rule); 2/4 if VRAM runs out
    GMESH="${GOFISH_MESH:-1}"         # 1 = Geometry Objective + mesh.ply, 0 = splats only
    GRUN="$SCENE/gofish_run"; mkdir -p "$GRUN" "$SCENE/out"
    # Calibrated domain: each lens's measured image circle (step 2) → GOFish never trains on
    # the black corners. Camera ids follow transplant_poses: sorted lenses → 1, 2.
    mapfile -t APER < <(python -c "
import json
for i, (_, a) in enumerate(sorted(json.load(open('$SCENE/apertures.json')).items()), 1):
    print('--aperture', i, a['cx'], a['cy'], a['r_train'])")
    (cd "$GF" && "$GPY" -m gofish.camera.make_domain "$SCENE/sparse/0" "$SCENE/domain.json" $(printf '%s ' "${APER[@]}"))
    if [[ "$GDS" != 1 ]]; then        # GOFish reads images_<k>/ at native//k
      python - "$SCENE/images" "$SCENE/images_$GDS" "$GDS" <<'PY'
import sys, cv2; from pathlib import Path
src, dst, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); dst.mkdir(exist_ok=True)
for p in sorted(src.glob("*.jpg")):
    im = cv2.imread(str(p)); h, w = im.shape[:2]
    cv2.imwrite(str(dst / p.name), cv2.resize(im, (w // k, h // k), interpolation=cv2.INTER_AREA))
PY
    fi
    # Renderer compiles one curvature bound per camera+resolution ("complete-admission table").
    (cd "$GF" && "$GPY" -m gofish.tools.build_calibration_table \
      --colmap "$SCENE/sparse/0/cameras.txt" --downsample "$GDS")
    GARGS=(--config-name apps/fiord path="$SCENE" out_dir="$GRUN" experiment_name="$NAME"
           "dataset.camera_domains=['$SCENE/domain.json']" dataset.downsample_factor="$GDS"
           dataset.test_split_interval=0 dataset.load_exif=false)   # train on every frame; ffmpeg frames carry no EXIF
    [[ "$GMESH" == 1 ]] && GARGS+=(geometry=sof)
    gofish_train() { (cd "$GF" && "$GPY" train.py "${GARGS[@]}") 2>&1 | tee "$GRUN/train.log"; return "${PIPESTATUS[0]}"; }
    if ! gofish_train; then
      # A table miss prints the exact build_calibration_table command; run it once and retry.
      CMDS="$(grep -oE 'python -m gofish\.tools\.build_calibration_table.*' "$GRUN/train.log" || true)"
      [[ -n "$CMDS" ]] || { echo "⚠️ GOFish train failed — see $GRUN/train.log"; exit 1; }
      while read -r c; do (cd "$GF" && eval "${c/python/$GPY}"); done <<< "$CMDS"
      gofish_train || { echo "⚠️ GOFish train failed after table update — see $GRUN/train.log"; exit 1; }
    fi
    CKPT="$(find "$GRUN" -name 'ckpt_last.pt' | head -1)"
    [[ -n "$CKPT" ]] || CKPT="$(find "$GRUN" -name 'ckpt_*.pt' | sort -V | tail -1)"
    [[ -n "$CKPT" ]] || { echo "✗ GOFish wrote no checkpoint — see $GRUN/train.log"; exit 1; }
    "$GPY" "$DS/gofish_to_ply.py" --checkpoint "$CKPT" --out "$SCENE/out/${NAME}.ply"
    if [[ "$GMESH" == 1 ]]; then
      GEO="$(find "$GRUN" -path '*ours_30000*' -name 'ckpt_30000.pt' | head -1)"   # geometry checkpoint
      if [[ -n "$GEO" ]] && (cd "$GF" && "$GPY" mesh.py --checkpoint "$GEO" --out "$GRUN/mesh"); then
        cp "$GRUN/mesh/mesh.ply" "$OUT/${NAME}.mesh.ply" && echo "  ✓ $OUT/${NAME}.mesh.ply"
      else
        echo "⚠️ GOFish mesh step failed/skipped (geometry ckpt: ${GEO:-none}) — splats still exported"
      fi
    fi
    ;;
  *) echo "✗ unknown TRAINER: $TRAINER (use 3dgut | surfels | gofish)"; exit 1 ;;
esac

PLY="$(find "$SCENE/out" -name '*.ply' | sort | tail -1)"
[[ -n "$PLY" ]] && cp "$PLY" "$OUT/${NAME}.ply" && echo "  ✓ $OUT/${NAME}.ply"
command -v splat-transform >/dev/null && [[ -f "$OUT/${NAME}.ply" ]] && \
  splat-transform "$OUT/${NAME}.ply" "$OUT/${NAME}.sog" && echo "  ✓ $OUT/${NAME}.sog"
echo "✓ done → exports/${NAME}.ply  (fisheye-native, zero resample)"
