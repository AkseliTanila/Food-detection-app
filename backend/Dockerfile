# Use Python base image
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ARG QWEN_VL_MODEL=Qwen/Qwen3-VL-2B-Instruct
ENV QWEN_VL_MODEL=${QWEN_VL_MODEL}
ENV HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
    HF_DATASETS_CACHE=/opt/hf_cache/datasets \
    PRELOADED_MODEL_DIR=/opt/models/qwen3-vl-2b-instruct

WORKDIR /app

# Install system deps needed for building transformers from source and bitsandbytes
RUN apt-get update && apt-get install -y git build-essential wget && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
# Install a torchvision wheel that matches the CUDA version of the preinstalled torch
# The base image provides torch (CUDA 12.x). Install torchvision from the PyTorch
# extra index for CUDA 12.8 wheels to ensure ABI compatibility.
RUN pip install --no-cache-dir --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu128 torchvision --upgrade || true

# Install remaining Python dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Pre-download the Qwen weights so the container can serve immediately on start
RUN mkdir -p ${PRELOADED_MODEL_DIR} && \
    python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("QWEN_VL_MODEL")
target_dir = os.environ.get("PRELOADED_MODEL_DIR")

print(f"Downloading {model_id} into {target_dir} ...")
snapshot_download(
	repo_id=model_id,
	local_dir=target_dir,
	local_dir_use_symlinks=False,
	resume_download=True,
)
PY

ENV QWEN_VL_LOCAL_PATH=${PRELOADED_MODEL_DIR}

# Copy the app
COPY . .

# Increase shared memory for large model workloads
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]