import os
import subprocess

def setup_dvc():
    user = os.getenv("DAGSHUB_USER", "3a9961a0e64b794ef7443c9217b5599716604b63")
    token = os.getenv("DAGSHUB_TOKEN", "3a9961a0e64b794ef7443c9217b5599716604b63")

    subprocess.run(["dvc", "remote", "add", "origin", "s3://dvc"])
    subprocess.run(["dvc", "remote", "modify", "origin", "endpointurl", "https://dagshub.com/auroremayo/TrustPilot-webservice-aurore.s3"])
    subprocess.run(["dvc", "remote", "modify", "origin", "access_key_id", user])
    subprocess.run(["dvc", "remote", "modify", "origin", "secret_access_key", token])