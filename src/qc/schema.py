from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Episode:
    episode_id: str
    collector_id: str
    fps: float

    video_ts: np.ndarray
    joint_ts: np.ndarray
    imu_ts: np.ndarray

    frames: np.ndarray

    joint_pos: np.ndarray

    projected_ee_px: np.ndarray
    detected_ee_px: np.ndarray

    ground_truth_fault: str = "clean"
    metadata: Dict[str, str] = field(default_factory=dict)