from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from src.qc.schema import Episode


def load_real_video(
    video_path: str,
    collector_id: str = "real_cam_01",
    max_frames: int = 180,
    target_width: int = 640,
) -> Episode:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if raw_fps and raw_fps > 1.0 else 30.0

    frames_list = []
    timestamps = []
    frame_idx = 0

    while len(frames_list) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = (pos_msec / 1000.0) if pos_msec and pos_msec > 0 else (frame_idx / fps)

        h, w = bgr.shape[:2]
        if w != target_width:
            scale = target_width / float(w)
            target_height = max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_list.append(rgb)
        timestamps.append(ts)
        frame_idx += 1

    cap.release()

    if len(frames_list) < 2:
        raise ValueError(f"Video has fewer than 2 readable frames: {video_path}")

    frames = np.stack(frames_list, axis=0).astype(np.uint8)
    t_len, h, w, _ = frames.shape
    video_ts = np.array(timestamps, dtype=np.float64)
    # Without external robot/IMU logs yet, align reference timestamps to video_ts
    joint_ts = video_ts.copy()
    imu_ts = video_ts.copy()

    # Estimate real 2D motion center-of-mass using dense optical flow magnitude
    detected_px = np.zeros((t_len, 2), dtype=np.float32)
    detected_px[0] = [w / 2.0, h / 2.0]

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, t_len):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        weights = np.maximum(mag - 1.0, 0.0)

        if float(np.sum(weights)) > 1e-3:
            yy, xx = np.mgrid[0:h, 0:w]
            cx = float(np.sum(xx * weights) / np.sum(weights))
            cy = float(np.sum(yy * weights) / np.sum(weights))
            detected_px[i] = [cx, cy]
        else:
            detected_px[i] = detected_px[i - 1]

        prev_gray = curr_gray

    # Normalize 2D motion into a 2-channel trajectory proxy for motion/idle checks
    norm_traj = np.column_stack([
        detected_px[:, 0] / float(w),
        detected_px[:, 1] / float(h),
    ]).astype(np.float32)

    return Episode(
        episode_id=path.stem,
        collector_id=collector_id,
        fps=fps,
        video_ts=video_ts,
        joint_ts=joint_ts,
        imu_ts=imu_ts,
        frames=frames,
        joint_pos=norm_traj,
        projected_ee_px=detected_px.copy(),
        detected_ee_px=detected_px,
        ground_truth_fault="real_unlabeled",
        metadata={"source_file": str(path)},
    )

from typing import Sequence

def load_group_photos(
    image_paths: Sequence[str],
    collector_id: str = "group_photo_batch",
    fps: float = 5.0,
    target_width: int = 640,
) -> Episode:
    valid_paths = [Path(p) for p in image_paths if p and Path(p).exists()]
    if not valid_paths:
        raise ValueError("No valid photo files provided.")

    frames_list = []
    for p in valid_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if w != target_width:
            scale = target_width / float(w)
            target_height = max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frames_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if not frames_list:
        raise ValueError("Could not decode any uploaded images.")

    # If only 1 photo is uploaded, duplicate once so trajectory/delta checks have >= 2 frames
    if len(frames_list) == 1:
        frames_list.append(frames_list[0].copy())

    frames = np.stack(frames_list, axis=0).astype(np.uint8)
    t_len, h, w, _ = frames.shape
    ts = np.arange(t_len, dtype=np.float64) / max(fps, 1.0)
    center_px = np.tile(np.array([[w / 2.0, h / 2.0]], dtype=np.float32), (t_len, 1))
    norm_traj = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (t_len, 1))

    return Episode(
        episode_id=f"group_{len(valid_paths)}_photos",
        collector_id=collector_id,
        fps=fps,
        video_ts=ts,
        joint_ts=ts.copy(),
        imu_ts=ts.copy(),
        frames=frames,
        joint_pos=norm_traj,
        projected_ee_px=center_px.copy(),
        detected_ee_px=center_px,
        ground_truth_fault="group_photo_batch",
        metadata={"num_photos": len(valid_paths)},
    )