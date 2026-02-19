#####################################################################################
##### THE FOLLOWING HAS BEEN PASTED FROM CHATGPT. IT CONTAINS UNCHECKED CODE ########
#####################################################################################

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Detection:
    center_xy: Tuple[float, float]
    radius: float
    score: float  # higher is better (you can define how)

def circularity(area: float, perimeter: float) -> float:
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))

def fit_circle_from_contour(cnt: np.ndarray) -> Tuple[Tuple[float, float], float]:
    (x, y), r = cv2.minEnclosingCircle(cnt)
    return (float(x), float(y)), float(r)

def edge_circle_fallback(gray: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[Detection]:
    """
    Fallback: run Canny + contour circle-fit inside ROI, return best circular contour.
    roi: (x0, y0, x1, y1) inclusive-exclusive.
    """
    x0, y0, x1, y1 = roi
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    patch_blur = cv2.GaussianBlur(patch, (5, 5), 0)
    edges = cv2.Canny(patch_blur, 60, 160)

    # Close small gaps in edges
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        per = cv2.arcLength(cnt, True)
        circ = circularity(area, per)
        if circ < 0.5:
            continue

        (cx, cy), r = fit_circle_from_contour(cnt)

        # Score: prefer circular + larger area, but not crazy large
        score = circ * np.sqrt(area)

        # Convert to full-image coords
        full_cx = cx + x0
        full_cy = cy + y0

        det = Detection(center_xy=(full_cx, full_cy), radius=r, score=score)
        if (best is None) or (det.score > best.score):
            best = det

    return best

class PingPongBallSegmenter:
    def __init__(
        self,
        # Expected radius range in pixels (set loosely at first, tighten later)
        radius_range: Tuple[float, float] = (6.0, 60.0),
        # Minimum circularity threshold
        min_circularity: float = 0.6,
        # Foreground mask cleanup
        morph_kernel: Tuple[int, int] = (5, 5),
        # Background subtractor history and sensitivity
        mog2_history: int = 300,
        mog2_var_threshold: float = 20.0,
        detect_shadows: bool = True,
    ):
        self.r_min, self.r_max = radius_range
        self.min_circ = min_circularity
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)

        # Background subtractor: best for stationary camera
        self.bgs = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_var_threshold,
            detectShadows=detect_shadows
        )

        # If you're feeding only occasional frames, you can keep learningRate small/0 later.
        self.default_learning_rate = 0.01

    def _postprocess_mask(self, fgmask: np.ndarray) -> np.ndarray:
        """
        MOG2 mask: typically
          0   = background
          255 = foreground
          127 = shadow (if detectShadows=True)
        We drop shadows, keep only real foreground.
        """
        # Keep only definite foreground
        mask = (fgmask == 255).astype(np.uint8) * 255

        # Clean up speckles & fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        return mask

    def detect(self, gray: np.ndarray, learning_rate: Optional[float] = None) -> Optional[Detection]:
        if learning_rate is None:
            learning_rate = self.default_learning_rate

        if gray.ndim != 2:
            raise ValueError("Input must be grayscale (H x W).")

        # Mild blur helps stability
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        fgmask = self.bgs.apply(gray_blur, learningRate=learning_rate)
        mask = self._postprocess_mask(fgmask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue

            per = cv2.arcLength(cnt, True)
            circ = circularity(area, per)
            if circ < self.min_circ:
                continue

            (cx, cy), r = fit_circle_from_contour(cnt)
            if not (self.r_min <= r <= self.r_max):
                continue

            # Score: prefer more circular and reasonable size
            score = circ * np.sqrt(area)

            det = Detection(center_xy=(cx, cy), radius=r, score=score)
            if (best is None) or (det.score > best.score):
                best = det

        # If mask-based detection fails, try edge fallback around the largest blob area (or whole image)
        if best is None:
            # Try edge-based search on full image (still lightweight)
            best = edge_circle_fallback(gray_blur, (0, 0, gray.shape[1], gray.shape[0]))

        return best

def draw_detection(vis_bgr: np.ndarray, det: Detection) -> np.ndarray:
    out = vis_bgr.copy()
    cx, cy = det.center_xy
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(det.radius)), (0, 255, 0), 2)
    cv2.circle(out, (int(round(cx)), int(round(cy))), 2, (0, 0, 255), -1)
    cv2.putText(
        out, f"r={det.radius:.1f} score={det.score:.1f}",
        (int(cx) + 10, int(cy) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
    )
    return out

if __name__ == "__main__":
    # Your current line (single image):
    # im = cv2.imread("pictures/ball_3.jpg", cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread("pictures/ball_3.jpg", cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError("Could not read image. Check the path.")

    # For visualization
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Instantiate segmenter
    seg = PingPongBallSegmenter(
        radius_range=(6, 80),        # widen initially; tighten once you know typical pixel radius
        min_circularity=0.55,        # loosen a bit for partial blobs
        mog2_history=200,
        mog2_var_threshold=16,
        detect_shadows=True
    )

    # IMPORTANT:
    # Background subtractors need a few frames to build a model.
    # With a single image, we can "prime" it by feeding the same image a few times
    # (in a real system, you'd feed actual earlier frames).
    for _ in range(30):
        _ = seg.detect(gray, learning_rate=0.2)  # learn background quickly

    # Now "freeze" learning (learning_rate=0) so the ball isn't absorbed into background
    det = seg.detect(gray, learning_rate=0.0)

    if det is None:
        print("No ball detected.")
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
    else:
        print(f"Detected center={det.center_xy}, radius={det.radius:.2f}, score={det.score:.2f}")
        vis = draw_detection(bgr, det)

        # Show intermediate mask too (optional)
        # Recompute mask for viewing
        fgmask = seg.bgs.apply(cv2.GaussianBlur(gray, (5, 5), 0), learningRate=0.0)
        mask = (fgmask == 255).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, seg.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, seg.kernel, iterations=2)

        cv2.imshow("mask", mask)
        cv2.imshow("detection", vis)
        cv2.waitKey(0)