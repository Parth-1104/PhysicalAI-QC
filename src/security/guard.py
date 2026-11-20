from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Any
import cv2
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from src.qc.schema import Episode


@dataclass
class SignedManifest:
    episode_id: str
    collector_id: str
    payload_sha256: str
    signature_hex: str
    public_key_pem: str
    qc_summary: Dict[str, Any]


class SecurityGuard:
    def __init__(self, private_key: Ed25519PrivateKey | None = None) -> None:
        self.private_key = private_key or Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def sanitize_frames_around_workspace(self, ep: Episode, keep_radius_px: int = 120) -> np.ndarray:
        """
        Example task-preserving redaction: keeps a circular workspace zone around the
        active end-effector sharp, while softly blurring outer background regions.
        Swap/extend with face & OCR bounding-box masking for full PII/IP sanitization.
        """
        sanitized = np.empty_like(ep.frames)
        h, w, _ = ep.frames.shape[1:]
        yy, xx = np.ogrid[:h, :w]

        for i in range(len(ep.frames)):
            frame = ep.frames[i]
            blurred = cv2.GaussianBlur(frame, (25, 25), 8)
            cx, cy = float(ep.detected_ee_px[i, 0]), float(ep.detected_ee_px[i, 1])
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            # Smooth feather mask around workspace
            alpha = np.clip(1.0 - (dist_sq - (keep_radius_px * 0.6) ** 2) / ((keep_radius_px * 1.4) ** 2), 0.0, 1.0)
            alpha_3 = alpha[..., None]
            sanitized[i] = (frame * alpha_3 + blurred * (1.0 - alpha_3)).astype(np.uint8)

        return sanitized

    def sign_sanitized_episode(self, ep: Episode, sanitized_frames: np.ndarray, qc_summary: Dict[str, Any]) -> SignedManifest:
        hasher = hashlib.sha256()
        hasher.update(ep.episode_id.encode("utf-8"))
        hasher.update(ep.collector_id.encode("utf-8"))
        hasher.update(sanitized_frames.tobytes())
        hasher.update(ep.joint_pos.tobytes())
        digest = hasher.digest()

        signature = self.private_key.sign(digest)
        pub_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        return SignedManifest(
            episode_id=ep.episode_id,
            collector_id=ep.collector_id,
            payload_sha256=digest.hex(),
            signature_hex=signature.hex(),
            public_key_pem=pub_pem,
            qc_summary=qc_summary,
        )

    @staticmethod
    def verify_manifest(manifest: SignedManifest, sanitized_frames: np.ndarray, joint_pos: np.ndarray) -> bool:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        hasher = hashlib.sha256()
        hasher.update(manifest.episode_id.encode("utf-8"))
        hasher.update(manifest.collector_id.encode("utf-8"))
        hasher.update(sanitized_frames.tobytes())
        hasher.update(joint_pos.tobytes())
        computed_digest = hasher.digest()

        if computed_digest.hex() != manifest.payload_sha256:
            return False

        pub_key = load_pem_public_key(manifest.public_key_pem.encode("utf-8"))
        if not isinstance(pub_key, Ed25519PublicKey):
            return False

        try:
            pub_key.verify(bytes.fromhex(manifest.signature_hex), computed_digest)
            return True
        except Exception:
            return False
    @staticmethod
    def save_manifest(manifest: SignedManifest, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.__dict__, f, indent=2)