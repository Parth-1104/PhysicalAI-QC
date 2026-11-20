from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from src.qc.video_loader import load_real_video
from src.qc.triage import evaluate_episode
from src.security.guard import SecurityGuard


def save_rgb_video(frames, out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, h, w, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EgoGuard-QC on a real video file.")
    parser.add_argument("--video", required=True, help="Path to real video (.mp4, .mov, etc.)")
    parser.add_argument("--collector", default="parth_m1_cam", help="Collector ID")
    parser.add_argument("--max-frames", type=int, default=180, help="Max frames to process")
    args = parser.parse_args()

    ep = load_real_video(args.video, collector_id=args.collector, max_frames=args.max_frames)
    res = evaluate_episode(ep)

    print(f"Episode:    {ep.episode_id} ({len(ep.frames)} frames @ {ep.fps:.1f} FPS)")
    print(f"Decision:   {res.decision}")
    print(f"Risk Score: {res.risk_score:.3f}")
    print(f"Reasons:    {res.reasons}")
    print(f"Metrics:    {res.metrics}")

    out_dir = Path("outputs") / ep.episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if res.decision in ("AUTO_PASS", "HUMAN_REVIEW"):
        guard = SecurityGuard()
        sanitized_frames = guard.sanitize_frames_around_workspace(ep)
        manifest = guard.sign_sanitized_episode(
            ep,
            sanitized_frames,
            qc_summary={"decision": res.decision, "risk_score": res.risk_score, "reasons": res.reasons, "metrics": res.metrics},
        )
        assert guard.verify_manifest(manifest, sanitized_frames, ep.joint_pos), "Manifest verification failed!"

        sanitized_video_path = out_dir / f"{ep.episode_id}_sanitized.mp4"
        manifest_path = out_dir / f"{ep.episode_id}_manifest.json"
        save_rgb_video(sanitized_frames, sanitized_video_path, ep.fps)
        guard.save_manifest(manifest, str(manifest_path))

        print(f"Saved sanitized video -> {sanitized_video_path}")
        print(f"Saved signed manifest -> {manifest_path}")
    else:
        print("Episode auto-rejected; skipped signing and dataset export.")


if __name__ == "__main__":
    main()