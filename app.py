from __future__ import annotations
import json
import tempfile
from typing import List, Optional
import cv2
import gradio as gr
import numpy as np

from src.qc.artifacts import apply_sanitization, build_contact_sheet
from src.qc.triage import evaluate_episode
from src.qc.video_loader import load_group_photos, load_real_video
from src.security.guard import SecurityGuard, SignedManifest


def _write_mp4(frames: np.ndarray, fps: float) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _, h, w, _ = frames.shape
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp_path


def run_pipeline(
    video_file: Optional[str],
    group_photos: Optional[List[str]],
    collector_id: str,
    max_frames: int,
    sanitize_mode: str,
):
    cid = (collector_id or "prod_collector_01").strip()
    if video_file:
        ep = load_real_video(video_file, collector_id=cid, max_frames=int(max_frames), target_width=640)
        res = evaluate_episode(ep, is_video_proxy=True)
    elif group_photos:
        ep = load_group_photos(group_photos, collector_id=cid, target_width=640)
        res = evaluate_episode(ep, is_video_proxy=True)
    else:
        raise gr.Error("Upload a video clip OR select one or more group photos.")

    guard = SecurityGuard()
    sanitized_frames = apply_sanitization(ep.frames, ep, mode=sanitize_mode)
    manifest = guard.sign_sanitized_episode(
        ep,
        sanitized_frames,
        qc_summary={
            "decision": res.decision,
            "risk_score": res.risk_score,
            "reasons": res.reasons,
            "metrics": res.metrics,
            "sanitize_mode": sanitize_mode,
        },
    )
    is_valid = guard.verify_manifest(manifest, sanitized_frames, ep.joint_pos)
    out_mp4 = _write_mp4(sanitized_frames, ep.fps)
    sheet = build_contact_sheet(sanitized_frames, f"{res.decision} | risk={res.risk_score} | {', '.join(res.reasons)}")

    summary_md = (
        f"### Decision: `{res.decision}`  |  Risk Score: `{res.risk_score:.3f}`\n"
        f"- **Reasons:** `{', '.join(res.reasons)}`\n"
        f"- **Metrics:** `{res.metrics}`\n"
        f"- **Ed25519 Signature Valid:** `{'True' if is_valid else 'False'}`"
    )
    state = {"manifest": manifest.__dict__, "frames": sanitized_frames, "joint_pos": ep.joint_pos}
    return summary_md, out_mp4, sheet, json.dumps(manifest.__dict__, indent=2), state


def test_tamper(state: Optional[dict]):
    if not state:
        return "Run an episode or group photo batch first."
    manifest = SignedManifest(**state["manifest"])
    frames = state["frames"].copy()
    joint_pos = state["joint_pos"]
    before = SecurityGuard.verify_manifest(manifest, frames, joint_pos)
    frames[0, 0, 0, 0] ^= 1
    after = SecurityGuard.verify_manifest(manifest, frames, joint_pos)
    return f"**Before mutation:** `{before}`  |  **After 1-bit flip:** `{after}` (Tamper caught: `{not after}`)"


with gr.Blocks(title="EgoGuard-QC — Production Data QC & Provenance") as demo:
    gr.Markdown("# EgoGuard-QC: Physical AI Data Triage, Sanitization & Cryptographic Provenance")

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Option A: Upload Video (.mp4 / .mov / .webm)")
            photos_in = gr.Files(label="Option B: Upload Group Photos (Multiple Images)", file_types=["image"], file_count="multiple")
            collector_in = gr.Textbox(label="Collector / Rig ID", value="edge_rig_01")
            max_frames_in = gr.Slider(15, 180, value=90, step=15, label="Max Video Frames")
            mode_in = gr.Radio(["workspace_soft", "outer_light", "none"], value="workspace_soft", label="Sanitization Mode")
            run_btn = gr.Button("Run QC + Sign", variant="primary")
            tamper_btn = gr.Button("Verify 1-Bit Tamper Detection")

        with gr.Column():
            status_md = gr.Markdown("### Status: Ready")
            video_out = gr.Video(label="Sanitized Preview")
            sheet_out = gr.Image(label="Keyframe Contact Sheet")
            tamper_out = gr.Markdown("")

    manifest_out = gr.Code(label="Signed Manifest (JSON)", language="json")
    app_state = gr.State()

    run_btn.click(
        run_pipeline,
        inputs=[video_in, photos_in, collector_in, max_frames_in, mode_in],
        outputs=[status_md, video_out, sheet_out, manifest_out, app_state],
    )
    tamper_btn.click(test_tamper, inputs=[app_state], outputs=[tamper_out])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)