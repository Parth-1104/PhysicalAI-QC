from __future__ import annotations
from collections import Counter
from src.benchmarks.synthetic_generator import make_clean_episode, inject_fault
from src.qc.triage import evaluate_episode
from src.security.guard import SecurityGuard


def main() -> None:
    episodes = []
    fault_cycle = ["clean", "clean", "clean", "calibration_drift", "sync_lag", "blur_or_idle", "motion_spike"]

    for idx in range(28):
        base = make_clean_episode(episode_id=f"ep_{idx:03d}", collector_id=f"col_{idx % 4:02d}", seed=100 + idx)
        fault = fault_cycle[idx % len(fault_cycle)]
        if fault == "clean":
            episodes.append(base)
        else:
            episodes.append(inject_fault(base, fault, seed=200 + idx))

    guard = SecurityGuard()
    decisions = Counter()
    fault_caught = 0
    fault_total = 0

    for ep in episodes:
        res = evaluate_episode(ep)
        decisions[res.decision] += 1

        if ep.ground_truth_fault != "clean":
            fault_total += 1
            if res.decision in ("AUTO_REJECT", "HUMAN_REVIEW"):
                fault_caught += 1

        if res.decision == "AUTO_PASS":
            clean_frames = guard.sanitize_frames_around_workspace(ep)
            manifest = guard.sign_sanitized_episode(
                ep,
                clean_frames,
                qc_summary={"risk_score": res.risk_score, "reasons": res.reasons, "metrics": res.metrics},
            )
            verified = guard.verify_manifest(manifest, clean_frames, ep.joint_pos)
            assert verified, f"Signature verification failed for {ep.episode_id}"

        print(f"{ep.episode_id:24s} | gt={ep.ground_truth_fault:18s} | {res.decision:12s} | risk={res.risk_score:.3f} | {res.reasons}")

    total = len(episodes)
    human_needed = decisions["HUMAN_REVIEW"]
    human_reduction_pct = (1.0 - (human_needed / total)) * 100.0
    recall_pct = (fault_caught / max(1, fault_total)) * 100.0

    print("\n--- Summary ---")
    print(f"Total episodes:      {total}")
    print(f"Auto-Pass:           {decisions['AUTO_PASS']}")
    print(f"Auto-Reject:         {decisions['AUTO_REJECT']}")
    print(f"Human Review Queue:  {human_needed}")
    print(f"Human QC Reduction:  {human_reduction_pct:.1f}%")
    print(f"Fault Catch Recall:  {recall_pct:.1f}% ({fault_caught}/{fault_total})")


if __name__ == "__main__":
    main()