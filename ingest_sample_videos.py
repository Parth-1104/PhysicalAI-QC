from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "qualiadev/qsim-shelf-restocking-mujoco-300-lerobot"
OUT_DIR = Path("samples/ego_10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()
files = [f for f in api.list_repo_files(repo_id=REPO_ID, repo_type="dataset") if f.endswith((".mp4", ".webm", ".avi"))]

saved = 0
for idx, repo_file in enumerate(files[:10], start=1):
    local_file = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=repo_file)
    ext = Path(repo_file).suffix
    out_path = OUT_DIR / f"sample_{idx:02d}{ext}"
    out_path.write_bytes(Path(local_file).read_bytes())
    saved += 1
    print(f"Saved {repo_file} -> {out_path}")

print(f"Total saved: {saved} videos in {OUT_DIR}")