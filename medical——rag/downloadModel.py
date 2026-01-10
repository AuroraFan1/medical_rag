# download_model.py
from huggingface_hub import snapshot_download

# 下载整个模型
model_path = snapshot_download(
    repo_id="BAAI/bge-large-zh-v1.5",
    local_dir="./bge-large-zh-v1.5",
    local_dir_use_symlinks=False
)

print(f"模型已下载到: {model_path}")