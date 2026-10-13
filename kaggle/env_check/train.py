"""Kaggle diagnostic: which accelerator and network access does this account get?"""
import socket
import subprocess

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout or "NO GPU")
for host in ("github.com", "huggingface.co", "datashare.ed.ac.uk", "pypi.org"):
    try:
        print(host, "->", socket.gethostbyname(host))
    except OSError as e:
        print(host, "-> NO NETWORK:", e)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch import failed", e)
