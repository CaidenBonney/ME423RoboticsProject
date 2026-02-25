#!/usr/bin/env python3
"""
Run BlurBall on a single image and save an overlay visualization.

Usage:
  python blurball_single_image_infer.py \
    --repo_root /path/to/blurball \
    --weights /path/to/blurball_weights.pth \
    --image /path/to/frame.png \
    --out /path/to/overlay.png \
    --device cpu

Notes:
- BlurBall is multi-frame by default (frames_in=3). For a single image, we feed
  3 copies of the same frame. BlurBall's own docs warn duplicated frames can be
  problematic in re-encoded videos, but for a single frame sanity-check this is
  typically fine. :contentReference[oaicite:4]{index=4}
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf


def draw_bbox_and_center(img_bgr: np.ndarray, x: float, y: float, box_size: int = 20) -> np.ndarray:
    """Draw a fixed-size bbox centered at (x,y) and a center point."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = int(round(x)), int(round(y))

    # Clamp center within image for drawing
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    half = box_size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w - 1, cx + half), min(h - 1, cy + half)

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
    return out


@torch.no_grad()
def run_single_image(cfg, repo_root: Path, image_path: Path, weights_path: Path, device: str, out_path: Path):
    # Make repo importable
    sys.path.insert(0, str(repo_root / "src"))

    # Import AFTER sys.path update (repo-local modules)
    from detectors import build_detector
    from trackers import build_tracker
    from utils.image import get_affine_transform
    import torchvision.transforms as T

    # Force device in cfg where possible
    cfg.runner.device = device

    detector = build_detector(cfg)
    tracker = build_tracker(cfg)

    # Some detector implementations will move the model to the configured device internally,
    # but we keep tensors on CPU unless you run CUDA.
    if device.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")

    # Load image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img_bgr.shape[:2]

    # Build affine transform exactly like repo inference does (maps model coords -> original frame coords)
    c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    s = max(h, w) * 1.0

    trans = np.stack(
        [
            get_affine_transform(
                c, s, 0,
                [cfg.model.inp_width, cfg.model.inp_height],
                inv=1,
            )
            for _ in range(cfg.model.frames_in)
        ],
        axis=0,
    )
    trans = torch.tensor(trans, dtype=torch.float32)[None, :].to(torch_device)

    preprocess_frame = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((cfg.model.inp_height, cfg.model.inp_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create a 3-frame window from the same image (BlurBall default frames_in=3) :contentReference[oaicite:5]{index=5}
    frames_processed = [preprocess_frame(img_bgr) for _ in range(cfg.model.frames_in)]
    input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(torch_device)

    # Run detector
    batch_results, _hms_vis = detector.run_tensor(input_tensor, trans)

    # batch_results[0] is a dict keyed by output frame index (0..frames_out-1)
    out_dict = batch_results[0]
    if len(out_dict) == 0:
        print("No outputs returned by detector.")
        return

    # Prefer the last output frame index if present
    preferred_idx = cfg.model.frames_out - 1
    if preferred_idx in out_dict:
        preds = out_dict[preferred_idx]
    else:
        preds = out_dict[next(iter(out_dict.keys()))]

    # Run tracker update once to get a single (x,y)
    result = tracker.update(preds)

    x = float(result.get("x", -1))
    y = float(result.get("y", -1))
    visi = int(result.get("visi", 0))
    score = float(result.get("score", 0.0))

    print(f"Prediction: x={x:.2f}, y={y:.2f}, visibility={visi}, score={score:.4f}")
    if "length" in result and "angle" in result:
        print(f"Blur attrs: length={float(result['length']):.3f}, angle={float(result['angle']):.3f}")

    overlay = img_bgr.copy()
    if visi == 1 and x >= 0 and y >= 0:
        overlay = draw_bbox_and_center(overlay, x, y, box_size=24)
        cv2.putText(
            overlay,
            f"score={score:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            overlay,
            "BALL NOT FOUND",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved overlay: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True, type=Path, help="Path to cloned cogsys-tuebingen/blurball repo")
    ap.add_argument("--weights", required=True, type=Path, help="Path to BlurBall pretrained weights file")
    ap.add_argument("--image", required=True, type=Path, help="Path to input image")
    ap.add_argument("--out", required=True, type=Path, help="Path to save overlay image")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    ap.add_argument("--score_threshold", type=float, default=0.7, help="Recommended 0.7 for 1-step inference")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    if not (repo_root / "src" / "configs").exists():
        raise FileNotFoundError(f"Could not find configs at: {repo_root / 'src' / 'configs'}")

    # Compose Hydra config from repo
    config_dir = str(repo_root / "src" / "configs")
    overrides = [
        f"WASB_ROOT={repo_root}",  # global path in config :contentReference[oaicite:6]{index=6}
        f"detector.model_path={args.weights}",
        "detector.step=1",  # 1-step inference for better single-frame-ish behavior :contentReference[oaicite:7]{index=7}
        f"detector.postprocessor.score_threshold={args.score_threshold}",
        "runner.vis_result=False",
        "runner.vis_hm=False",
        "runner.vis_traj=False",
        f"runner.device={args.device}",
        # input_vid is required by some runners, but we won't use the runner; still set a dummy.
        "input_vid=dummy.mp4",
    ]

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="inference_blurball", overrides=overrides)

    # Optional: print the resolved config
    # print(OmegaConf.to_yaml(cfg))

    run_single_image(
        cfg=cfg,
        repo_root=repo_root,
        image_path=args.image,
        weights_path=args.weights,
        device=args.device,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()