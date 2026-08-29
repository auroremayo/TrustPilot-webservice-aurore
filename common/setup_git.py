import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def setup_git_auth(base_dir=None):
    user = os.getenv("GIT_USER", "auroremayo").strip().strip('"')
    token = os.getenv("GIT_TOKEN", "").strip().strip('"')
    email = os.getenv("GIT_EMAIL", f"{user}@users.noreply.github.com").strip().strip('"')

    cwd = str(base_dir) if base_dir else None

    # Configuration identité Git
    subprocess.run(["git", "config", "user.name", user], cwd=cwd, check=False)
    subprocess.run(["git", "config", "user.email", email], cwd=cwd, check=False)

    if user and token:
        repo_url = f"https://{user}:{token}@github.com/auroremayo/TrustPilot-webservice-aurore.git"
        res = subprocess.run(["git", "remote", "set-url", "origin", repo_url], cwd=cwd, capture_output=True)
        if res.returncode != 0:
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=cwd, check=False)
        logger.info("✅ Git remote 'origin' configuré.")
    else:
        logger.warning("⚠️ GIT_USER ou GIT_TOKEN manquant pour la configuration Git.")