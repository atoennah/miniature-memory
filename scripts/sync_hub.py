import os
import argparse
import shutil
from huggingface_hub import HfApi, snapshot_download

# Configuration (Env Vars are safer for agents)
HF_TOKEN = os.getenv("HF_TOKEN") # Important!
MODEL_REPO = os.getenv("HF_MODEL_REPO") # Important!
DATA_REPO = os.getenv("HF_DATA_REPO")   # Important!

api = HfApi(token=HF_TOKEN) # Important!

def push_state(target):
    """Agent uploads its work to the cloud."""
    if target in ["all", "data"]:
        print(f"Uploading Data to {DATA_REPO}...")
        api.upload_folder(
            folder_path="dataset/processed",
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_in_repo="processed",
            allow_patterns=["*.txt"]
        )
        api.upload_file(
            path_or_fileobj="dataset/metadata/urls.jsonl",
            path_in_repo="metadata/urls.jsonl",
            repo_id=DATA_REPO,
            repo_type="dataset"
        )
        api.upload_file(
            path_or_fileobj="dataset/README.md",
            path_in_repo="README.md",
            repo_id=DATA_REPO,
            repo_type="dataset"
        )

    if target in ["all", "model"]:
        print(f"Uploading Model to {MODEL_REPO}...")
        # Upload checkpoints to a dedicated folder for better organization
        api.upload_folder(
            folder_path="training/checkpoints",
            repo_id=MODEL_REPO,
            repo_type="model",
            path_in_repo="checkpoints"
        )
        # Upload the config file to the repo root
        api.upload_file(
            path_or_fileobj="training/configs/small.yaml",
            path_in_repo="config.yaml",
            repo_id=MODEL_REPO,
            repo_type="model"
        )


def pull_state(target):
    """Agent downloads previous work to resume."""
    if target in ["all", "data"]:
        print(f"Downloading Data from {DATA_REPO}...")
        snapshot_download(
            repo_id=DATA_REPO,
            repo_type="dataset",
            local_dir="dataset",
            allow_patterns=["processed/*", "metadata/*", "README.md"]
        )

    if target in ["all", "model"]:
        print(f"Downloading Model from {MODEL_REPO}...")
        # Download the repo content into the training directory
        snapshot_download(
            repo_id=MODEL_REPO,
            repo_type="model",
            local_dir="training",
            allow_patterns=["checkpoints/*", "config.yaml"]
        )

        # The config is downloaded to training/config.yaml, move it to the correct location
        downloaded_config_path = "training/config.yaml"
        destination_config_path = "training/configs/small.yaml"
        if os.path.exists(downloaded_config_path):
            print(f"Moving downloaded config to {destination_config_path}")
            os.makedirs(os.path.dirname(destination_config_path), exist_ok=True)
            shutil.move(downloaded_config_path, destination_config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--target", choices=["all", "data", "model"], default="all")
    args = parser.parse_args()

    if args.action == "push":
        push_state(args.target)
    elif args.action == "pull":
        pull_state(args.target)
