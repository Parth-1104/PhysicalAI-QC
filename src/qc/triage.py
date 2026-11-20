from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import cv2
import numpy as np
from src.qc.schema import Episode


@dataclass
class TriageResult:
    episode_id: str
    decision: str
    risk_score: float
    reasons: List[str]
    metrics: Dict[str, float]


def compute_sharpness_scores(frames: np.ndarray) -> np.ndarray:
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scores.append(lap_var)
    return np.array(scores, dtype=np.float64)


def evaluate_episode(ep: Episode, is_video_proxy: bool = False) -> TriageResult:
    reasons: List[str] = []

    max_joint_offset_ms = float(np.max(np.abs(ep.joint_ts - ep.video_ts)) * 1000.0)
    max_imu_offset_ms = float(np.max(np.abs(ep.imu_ts - ep.video_ts)) * 1000.0)
    max_sync_drift_ms = max(max_joint_offset_ms, max_imu_offset_ms)

    reproj_err_px = np.linalg.norm(ep.detected_ee_px - ep.projected_ee_px, axis=1)
    p95_reproj_err = float(np.percentile(reproj_err_px, 95))
    max_reproj_err = float(np.max(reproj_err_px))

    sharpness = compute_sharpness_scores(ep.frames)
    min_sharpness = float(np.min(sharpness))
    mean_sharpness = float(np.mean(sharpness))

    traj = ep.joint_pos.astype(np.float64)
    if is_video_proxy and len(traj) >= 5:
        # Light 3-frame moving average to suppress optical-flow centroid jitter
        kernel = np.ones(3, dtype=np.float64) / 3.0
        traj = np.column_stack([
            np.convolve(traj[:, 0], kernel, mode="same"),
            np.convolve(traj[:, 1], kernel, mode="same"),
        ])

    dt = 1.0 / max(ep.fps, 1.0)
    if len(traj) >= 3:
        vel = np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt
        acc_series = np.abs(np.diff(vel)) / dt
        max_joint_acc = float(np.max(acc_series)) if len(acc_series) > 0 else 0.0
        idle_thresh = 0.02 if is_video_proxy else 0.05
        idle_fraction = float(np.mean(vel < idle_thresh))
    else:
        max_joint_acc = 0.0
        idle_fraction = 0.0

    # Thresholds calibrated separately for robot telemetry vs. optical-flow video proxy
    acc_warn = 12000.0 if is_video_proxy else 60.0
    acc_reject = 25000.0 if is_video_proxy else 250.0
    acc_scale = 18000.0 if is_video_proxy else 80.0
    idle_warn = 0.75 if is_video_proxy else 0.45
    sharp_warn = 25.0 if is_video_proxy else 15.0
    sharp_reject = 8.0 if is_video_proxy else 5.0

    sync_risk = float(np.clip((max_sync_drift_ms - 10.0) / 40.0, 0.0, 1.0))
    calib_risk = float(np.clip((p95_reproj_err - 6.0) / 18.0, 0.0, 1.0))
    motion_risk = float(np.clip((max_joint_acc - acc_warn) / acc_scale, 0.0, 1.0))

    sharpness_penalty = max(0.0, (sharp_warn - min_sharpness) / max(sharp_warn, 1.0))
    idle_penalty = max(0.0, (idle_fraction - idle_warn) / max(1.0 - idle_warn, 0.1))
    blur_idle_risk = float(np.clip(0.70 * sharpness_penalty + 0.30 * idle_penalty, 0.0, 1.0))

    if max_sync_drift_ms > 20.0:
        reasons.append(f"SYNC_DRIFT:{max_sync_drift_ms:.1f}ms")
    if p95_reproj_err > 10.0:
        reasons.append(f"CALIB_DRIFT_P95:{p95_reproj_err:.1f}px")
    if max_joint_acc > acc_warn:
        reasons.append(f"MOTION_SPIKE:{max_joint_acc:.1f}")
    if min_sharpness < sharp_warn:
        reasons.append(f"LOW_SHARPNESS:{min_sharpness:.1f}")
    if idle_fraction > idle_warn:
        reasons.append(f"HIGH_IDLE:{idle_fraction:.0%}")

    risk_score = float(
        np.clip(
            0.35 * calib_risk + 0.25 * sync_risk + 0.25 * motion_risk + 0.15 * blur_idle_risk,
            0.0,
            1.0,
        )
    )

    hard_fail = (
        risk_score >= 0.65
        or max(sync_risk, calib_risk) >= 0.95
        or max_joint_acc >= acc_reject
        or min_sharpness < sharp_reject
    )
    needs_review = (
        risk_score >= 0.30
        or len(reasons) >= 2
        or min_sharpness < sharp_warn
        or idle_fraction > idle_warn
    )

    if hard_fail:
        decision = "AUTO_REJECT"
    elif needs_review:
        decision = "HUMAN_REVIEW"
    else:
        decision = "AUTO_PASS"

    return TriageResult(
        episode_id=ep.episode_id,
        decision=decision,
        risk_score=round(risk_score, 4),
        reasons=reasons if reasons else ["OK"],
        metrics={
            "max_sync_drift_ms": round(max_sync_drift_ms, 2),
            "p95_reproj_err_px": round(p95_reproj_err, 2),
            "max_reproj_err_px": round(max_reproj_err, 2),
            "min_sharpness": round(min_sharpness, 2),
            "mean_sharpness": round(mean_sharpness, 2),
            "max_joint_acc": round(max_joint_acc, 2),
            "idle_fraction": round(idle_fraction, 3),
        },
    )