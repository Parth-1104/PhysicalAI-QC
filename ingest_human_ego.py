from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Voxel51/Egocentric_10K_subset"
OUT_DIR = Path("samples/human_ego")
OUT_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()
mp4_files = [
    f for f in api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    if f.lower().endswith(".mp4")
]

saved = 0
for idx, repo_file in enumerate(mp4_files[:5], start=1):
    local_file = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=repo_file)
    out_path = OUT_DIR / f"human_ego_{idx:02d}.mp4"
    out_path.write_bytes(Path(local_file).read_bytes())
    saved += 1
    print(f"Saved {repo_file} -> {out_path}")

print(f"Total human egocentric clips saved: {saved} in {OUT_DIR}")