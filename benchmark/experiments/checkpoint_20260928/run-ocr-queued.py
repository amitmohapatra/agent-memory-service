import os
import subprocess
import sys
import time
from pathlib import Path

root = Path.cwd()
for wait_pid in (32270,):
    print("Waiting for owned prerequisite", wait_pid, flush=True)
    for _ in range(2880):
        try:
            os.kill(wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(10)
    else:
        raise RuntimeError("Prerequisite did not exit within eight hours")
image = "memory-ocr-headless-screen:20260928"
image_id = subprocess.check_output(
    ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
).strip()
command = [
    "docker",
    "run",
    "--rm",
    "--name",
    "memory-multilingual-ocr-20260928",
    "--network",
    "none",
    "--cpus",
    "2",
    "--memory",
    "4g",
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--tmpfs",
    "/tmp:rw,nosuid,size=512m",
    "--workdir",
    "/workspace",
    "-e",
    "PYTHONPATH=/workspace/src",
    "-e",
    "PYTHONDONTWRITEBYTECODE=1",
    "-e",
    "HF_HUB_OFFLINE=1",
    "-e",
    "TRANSFORMERS_OFFLINE=1",
    "-e",
    "MEMORY_DOCLING_ARTIFACTS=/models/docling",
    "-e",
    "OMP_THREAD_LIMIT=2",
    "-e",
    "OMP_NUM_THREADS=2",
    "-e",
    "OPENBLAS_NUM_THREADS=1",
]
for local, remote, mode in [
    ("src", "/workspace/src", "ro"),
    ("benchmark", "/workspace/benchmark", "ro"),
    ("tests/eval/golden", "/workspace/tests/eval/golden", "ro"),
    ("benchmark/results", "/results", "rw"),
]:
    command.extend(["-v", str(root / local) + ":" + remote + ":" + mode])
command.extend(
    [
        "-v",
        "/Users/ricky/usage_data/agent-memory-service/models/docling:/models/docling:ro",
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "-m",
        "benchmark.multilingual_ocr",
        "--output",
        "/results/multilingual_ocr_tesseract.json",
        "--image-digest",
        image_id,
    ]
)
print("Starting isolated OCR screen", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
with (root / ".bench_data/multilingual-ocr-screen.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
print("OCR exit", result.returncode, flush=True)
if result.returncode:
    sys.exit(result.returncode)
env = dict(os.environ)
env.update(
    PYTHONPATH=".:src:.sdk-test-deps:sdk/python/src:/Users/ricky/usage_data/ams-hindsight-benchmark/.hindsight-venv/lib/python3.12/site-packages",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="2",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)
print("Starting fixed multilingual fusion screen", flush=True)
with (root / ".bench_data/multilingual-fusion-screen.log").open("w") as log:
    result = subprocess.run(
        [sys.executable, "-u", ".bench_data/multilingual-fusion-screen.py"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
print("Multilingual fusion exit", result.returncode, flush=True)
sys.exit(result.returncode)
