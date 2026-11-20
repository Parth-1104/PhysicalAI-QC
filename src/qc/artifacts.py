from __future__ import annotations
import cv2
import numpy as np
from src.qc.schema import Episode


def apply_sanitization(frames: np.ndarray, ep: Episode, mode: str = "workspace_soft") -> np.ndarray:
    """
    Modes:
      - 'workspace_soft': Keeps a wide workspace/ROI sharp with smooth feathering; lightly blurs outer margins.
      - 'outer_light': Gentle outer vignette blur (keeps ~85% of scene clear).
      - 'none': Unmodified frames (useful when testing raw provenance signing).
    """
    if mode == "none":
        return frames.copy()

    out = np.empty_like(frames)
    t_len, h, w, _ = frames.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    for i in range(t_len):
        frame = frames[i]
        blur_k = 15 if mode == "workspace_soft" else 11
        blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)

        if mode == "outer_light":
            cx, cy = w * 0.5, h * 0.5
            rx, ry = w * 0.45, h * 0.45
        else:
            cx, cy = float(ep.detected_ee_px[i, 0]), float(ep.detected_ee_px[i, 1])
            rx, ry = w * 0.36, h * 0.36

        # Smooth elliptical alpha (1.0 inside workspace, smooth falloff outside)
        dist = np.sqrt(((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2)
        alpha = np.clip(1.0 - np.maximum(dist - 0.75, 0.0) / 0.55, 0.0, 1.0)[..., None]

        out[i] = np.clip(alpha * frame.astype(np.float32) + (1.0 - alpha) * blurred.astype(np.float32), 0, 255).astype(np.uint8)

    return out


def build_contact_sheet(frames: np.ndarray, label: str) -> np.ndarray:
    """Creates a 2x2 keyframe contact sheet with triage label banner."""
    n = len(frames)
    idxs = np.linspace(0, max(n - 1, 0), 4, dtype=int)
    tiles = []
    for idx in idxs:
        tile = frames[idx].copy()
        cv2.putText(
            tile,
            f"f={idx}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )
        tiles.append(tile)

    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    grid = np.vstack([top, bot])

    banner = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner,
        label[:90],
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return np.vstack([banner, grid])