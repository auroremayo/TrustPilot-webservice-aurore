import os
import subprocess

def setup_git_auth():
    user = os.getenv("GIT_USER")
    token = os.getenv("GIT_TOKEN")

    repo_url = f"https://{user}:{token}@github.com/auroremayo/TrustPilot-webservice-aurore.git"

    subprocess.run(["git", "remote", "add", "origin", repo_url], capture_output=True)
    subprocess.run(["git", "remote", "set-url", "origin", repo_url])