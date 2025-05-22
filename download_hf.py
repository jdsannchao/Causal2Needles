import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "causal2needles/Causal2Needles"
repo_type = "dataset"
local_root = "datasets"

target_prefixes = [
    "questions/symon/1needle/",
    "questions/symon/2needle/",
    "questions/yms/1needle/",
    "questions/yms/2needle/",
    "videos/symon/",
    "videos/yms/"
]

# List all files in the repo
all_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Collect matched files
files_to_download = []
for prefix in target_prefixes:
    files_to_download.extend([f for f in all_files if f.startswith(prefix)])

# Add annotations.json explicitly
files_to_download.append("annotations.json")

# Download each file and save under datasets/
for filename in files_to_download:
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type
    )

    local_target_path = os.path.join(local_root, filename)
    os.makedirs(os.path.dirname(local_target_path), exist_ok=True)

    with open(cached_path, "rb") as src, open(local_target_path, "wb") as dst:
        dst.write(src.read())

    print(f"Saved: {local_target_path}")
