from __future__ import annotations
import numpy as np
import cv2
from src.qc.schema import Episode


def make_clean_episode(episode_id: str = "ep_001", collector_id: str = "col_01", seed: int = 42) -> Episode:
    rng = np.random.default_rng(seed)
    fps = 30.0
    duration = 4.0
    n_frames = int(fps * duration)

    t = np.arange(n_frames, dtype=np.float64) / fps
    video_ts = t.copy()
    joint_ts = t + rng.normal(0.0, 0.0002, size=n_frames)  # ~0.2ms jitter
    imu_ts = t + rng.normal(0.0, 0.0002, size=n_frames)

    # Smooth sinusoidal trajectory for 6 joints
    phases = np.linspace(0, 2 * np.pi, n_frames)[:, None]
    freqs = np.linspace(0.5, 2.0, 6)[None, :]
    joint_pos = (0.6 * np.sin(phases * freqs)).astype(np.float32)

    # Smooth 2D end-effector trajectory in a 640x480 frame
    cx = 320.0 + 110.0 * np.cos(t * 1.3)
    cy = 240.0 + 75.0 * np.sin(t * 2.1)
    projected_ee_px = np.stack([cx, cy], axis=1).astype(np.float32)
    # Small natural tracking noise (~1.5px)
    detected_ee_px = (projected_ee_px + rng.normal(0.0, 1.5, size=projected_ee_px.shape)).astype(np.float32)

    # Render textured background + moving target object + moving end-effector marker
    frames = np.zeros((n_frames, 480, 640, 3), dtype=np.uint8)
    base_bg = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(0, 480, 40):
        cv2.line(base_bg, (0, y), (640, y), (45, 50, 60), 1)
    for x in range(0, 640, 40):
        cv2.line(base_bg, (x, 0), (x, 480), (45, 50, 60), 1)

    for i in range(n_frames):
        frame = base_bg.copy()
        # Static target block
        cv2.rectangle(frame, (360, 210), (410, 260), (60, 180, 90), -1)
        # Moving end-effector / hand marker
        px, py = int(np.clip(detected_ee_px[i, 0], 15, 625)), int(np.clip(detected_ee_px[i, 1], 15, 465))
        cv2.circle(frame, (px, py), 14, (230, 140, 40), -1)
        frames[i] = frame

    return Episode(
        episode_id=episode_id,
        collector_id=collector_id,
        fps=fps,
        video_ts=video_ts,
        joint_ts=joint_ts,
        imu_ts=imu_ts,
        frames=frames,
        joint_pos=joint_pos,
        projected_ee_px=projected_ee_px,
        detected_ee_px=detected_ee_px,
        ground_truth_fault="clean",
    )


def inject_fault(ep: Episode, fault_type: str, seed: int = 99) -> Episode:
    rng = np.random.default_rng(seed)
    out = Episode(
        episode_id=f"{ep.episode_id}_{fault_type}",
        collector_id=ep.collector_id,
        fps=ep.fps,
        video_ts=ep.video_ts.copy(),
        joint_ts=ep.joint_ts.copy(),
        imu_ts=ep.imu_ts.copy(),
        frames=ep.frames.copy(),
        joint_pos=ep.joint_pos.copy(),
        projected_ee_px=ep.projected_ee_px.copy(),
        detected_ee_px=ep.detected_ee_px.copy(),
        ground_truth_fault=fault_type,
    )

    if fault_type == "calibration_drift":
        # Camera mount shifts mid-episode, causing >18px reprojection error
        mid = len(out.frames) // 2
        out.detected_ee_px[mid:] += np.array([22.0, -16.0], dtype=np.float32)

    elif fault_type == "sync_lag":
        # Robot joint stream lags video by 80ms
        out.joint_ts += 0.080

    elif fault_type == "blur_or_idle":
        # Heavy motion blur in second half + frozen motion
        mid = len(out.frames) // 2
        for i in range(mid, len(out.frames)):
            out.frames[i] = cv2.GaussianBlur(out.frames[mid], (35, 35), 10)
        out.joint_pos[mid:] = out.joint_pos[mid]

    elif fault_type == "motion_spike":
        # Dangerous sudden joint jump / jerk spike
        spike_idx = len(out.joint_pos) // 3
        out.joint_pos[spike_idx : spike_idx + 2] += 2.5

    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    return out