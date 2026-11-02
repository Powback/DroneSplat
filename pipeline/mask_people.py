#!/usr/bin/env python3
"""
mask_people.py — mask people (and other movers) out of the raw fisheye frames.

Moving people break both stages: SfM matches features on them (wrong poses) and the trainer
bakes them in as floaters/ghosts. Masking them out of both is enough — GOFish/3dgrut read a
per-frame `<image>_mask.png` (255 = learn, 0 = ignore) and drop those pixels from the loss;
COLMAP ignores features under a mask.

Detectors are trained on perspective photos, and people near the 196° rim are badly
distorted, so we never detect on the fisheye directly: each lens is rendered into 9
perspective views (centre + ring) that cover the whole circle, a segmenter masks them
(SAM 3 with the text prompt "person" by default; torchvision Mask R-CNN as --model maskrcnn),
and every fisheye pixel looks its ray up in each view through the same Kannala-Brandt model
the trainers use. Union, then dilate = padding (motion blur, hair, shadows' edges).

  python3 mask_people.py --in SCENE/images --cam SCENE/cam_init.json --out SCENE/masks
Writes masks/<frame stem>.png: 255 = keep, 0 = masked. Needs torch (CUDA if present) + sam3.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("✗ pip install opencv-python numpy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproject_sfm import build_map, rot  # noqa: E402  same lens model + view rotations

# Detection rig: centre + 8 views ~55° off-axis; at 100° FoV each reaches ~105° off-axis,
# past the A1's 98° rim, with overlap between neighbours.
RIG = [(0, 0), (55, 0), (-55, 0), (0, 55), (0, -55), (40, 40), (-40, 40), (40, -40), (-40, -40)]
COCO = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "bus": 6, "truck": 8, "dog": 18}


def fisheye_rays(cam: dict, scale: float) -> np.ndarray:
    """Unit rays (fisheye frame) for every pixel of the lens at `scale` × native size."""
    w, h = int(cam["width"] * scale), int(cam["height"] * scale)
    j, i = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (j + 0.5) / scale - 0.5 - cam["cx"]
    y = (i + 0.5) / scale - 0.5 - cam["cy"]
    td = np.hypot(x, y) / cam["fx"]                        # distorted angle θ_d
    k1, k2, k3, k4 = cam["k1"], cam["k2"], cam["k3"], cam["k4"]
    th = td.copy()
    for _ in range(12):                                    # Newton: θ(1+k1θ²+…) = θ_d
        t2 = th * th
        f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - td
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        th = np.clip(th - f / df, 0, math.pi)
    ph = np.arctan2(y, x)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


def lookup_maps(rays: np.ndarray, R: np.ndarray, out_size: int, fov_deg: float):
    """Fisheye pixel → view pixel maps (cv2.remap), -1 where the view can't see the ray."""
    fp = (out_size / 2) / math.tan(math.radians(fov_deg) / 2)
    d = rays @ R                                           # fisheye frame → view frame
    z = d[..., 2]
    ok = z > 1e-3
    zs = np.where(ok, z, 1.0)
    u = np.where(ok, fp * d[..., 0] / zs + out_size / 2, -1)
    v = np.where(ok, fp * d[..., 1] / zs + out_size / 2, -1)
    return u.astype(np.float32), v.astype(np.float32)


class Sam3Detector:
    """SAM 3 text-prompted segmentation (facebookresearch/sam3): every instance matching the
    noun phrase(s) in one pass. Latest SAM as of 2026-11 (3.1's changes are video-only).
    Weights are gated on Hugging Face (facebook/sam3): accept the licence, set HF_TOKEN."""

    def __init__(self, prompts: list[str], score: float):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from PIL import Image
        self.torch, self.Image = torch, Image
        try:
            self.proc = Sam3Processor(build_sam3_image_model())
        except Exception as e:  # gated checkpoint: say what to do instead of a raw HF traceback
            sys.exit(f"✗ SAM 3 weights unavailable ({type(e).__name__}: {e}).\n"
                     "  Accept the licence at https://huggingface.co/facebook/sam3 and pass HF_TOKEN,\n"
                     "  or run with --model maskrcnn.")
        self.prompts, self.score = prompts, score
        print(f"  SAM 3, prompts {prompts}, score ≥ {score}")

    def __call__(self, views_bgr: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for v in views_bgr:
            state = self.proc.set_image(self.Image.fromarray(v[..., ::-1].copy()))
            hit = np.zeros(v.shape[:2], bool)
            for p in self.prompts:
                o = self.proc.set_text_prompt(state=state, prompt=p)
                masks, scores = o["masks"], o["scores"]
                if len(scores) == 0:
                    continue
                m = self.torch.as_tensor(masks).reshape(len(scores), *masks.shape[-2:])
                keep = self.torch.as_tensor(scores).flatten().to(m.device) >= self.score
                if keep.any():
                    hit |= (m[keep] > 0.5).any(0).cpu().numpy()
            out.append(hit.astype(np.uint8) * 255)
        return out


class Detector:
    def __init__(self, classes: list[int], score: float):
        import torch, torchvision
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().to(self.dev)
        self.classes, self.score = set(classes), score
        print(f"  Mask R-CNN (COCO) on {self.dev}, classes {sorted(self.classes)}, score ≥ {score}")

    def __call__(self, views_bgr: list[np.ndarray]) -> list[np.ndarray]:
        t = self.torch
        batch = [t.from_numpy(v[..., ::-1].copy()).permute(2, 0, 1).float().div(255).to(self.dev)
                 for v in views_bgr]
        with t.no_grad():
            outs = self.net(batch)
        masks = []
        for o, v in zip(outs, views_bgr):
            keep = (o["scores"] >= self.score) & t.tensor([int(l) in self.classes for l in o["labels"]],
                                                          dtype=t.bool, device=o["scores"].device)
            m = (o["masks"][keep, 0] > 0.5).any(0) if keep.any() else t.zeros(v.shape[:2], dtype=t.bool)
            masks.append(m.cpu().numpy().astype(np.uint8) * 255)
        return masks


def mask_frame(im: np.ndarray, cam: dict, views, detect, scale: float, dilate: int) -> np.ndarray:
    """views: [(fwd_mx, fwd_my, back_u, back_v)] per rig view. Returns 255 keep / 0 masked."""
    if im.shape[:2] != (cam["height"], cam["width"]):
        im = cv2.resize(im, (cam["width"], cam["height"]))
    persp = [cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
             for mx, my, _, _ in views]
    hit = np.zeros(views[0][2].shape, np.uint8)
    for m, (_, _, u, v) in zip(detect(persp), views):
        if m.any():
            hit |= cv2.remap(m, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    hit = cv2.resize(hit, (cam["width"], cam["height"]), interpolation=cv2.INTER_NEAREST)
    if dilate > 0 and hit.any():
        hit = cv2.dilate(hit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2))
    return np.where(hit > 0, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="mask people out of raw fisheye frames")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="raw fisheye frames (lensN_*.jpg)")
    ap.add_argument("--cam", type=Path, required=True, help="calibrate.py JSON (per-lens KB intrinsics)")
    ap.add_argument("--out", type=Path, required=True, help="mask dir (<stem>.png, 255 keep / 0 masked)")
    ap.add_argument("--model", choices=["sam3", "maskrcnn"], default="sam3",
                    help="sam3 = SAM 3 text prompts (gated weights, HF_TOKEN); maskrcnn = torchvision COCO")
    ap.add_argument("--classes", default="person",
                    help=f"comma list; sam3: any noun phrases, maskrcnn: from {sorted(COCO)}")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--view-size", type=int, default=1024, help="detection view px")
    ap.add_argument("--view-fov", type=float, default=100.0)
    ap.add_argument("--scale", type=float, default=0.5, help="mask lookup at this × native, then upsampled")
    ap.add_argument("--dilate", type=int, default=16, help="grow masks by this many native px")
    args = ap.parse_args()

    lenses = json.loads(args.cam.read_text())["lenses"]
    frames = sorted(p for p in args.inp.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and not p.stem.endswith("_mask"))
    if not frames:
        sys.exit(f"✗ no frames in {args.inp}")
    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    detect = (Sam3Detector(names, args.score) if args.model == "sam3"
              else Detector([COCO[c] for c in names], args.score))
    args.out.mkdir(parents=True, exist_ok=True)
    rigs = {}
    for lens, cam in lenses.items():
        rays = fisheye_rays(cam, args.scale)
        rigs[lens] = [(*build_map(cam, args.view_size, args.view_fov, rot(y, p))[:2],
                       *lookup_maps(rays, rot(y, p), args.view_size, args.view_fov)) for y, p in RIG]
    masked = 0
    for n, f in enumerate(frames, 1):
        lens = f.stem.split("_")[0]
        if lens not in rigs:
            sys.exit(f"✗ {f.name}: no intrinsics for lens '{lens}' in {args.cam}")
        m = mask_frame(cv2.imread(str(f)), lenses[lens], rigs[lens], detect, args.scale, args.dilate)
        cv2.imwrite(str(args.out / f"{f.stem}.png"), m)
        masked += int((m == 0).any())
        if n % 50 == 0 or n == len(frames):
            print(f"  {n}/{len(frames)} frames, {masked} with people masked")
    print(f"✓ masks → {args.out}")


if __name__ == "__main__":
    main()
