# Food-detection-app

AI based food detection application that uses Qwen3-VL for on-device multimodal inference.

## Prerequisites

- NVIDIA GPU with drivers >= 535 and at least 10 GB VRAM (for Qwen3-VL-2B).
- Docker 24+ with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed so `docker info` lists `Runtimes: nvidia`.
- (Optional) `huggingface-cli` logged in on the host if you plan to use gated/private models; public Qwen weights download anonymously.

## Building with the model baked in

1. Clone the repository and `cd` into the root (`Food-detection-app`).
2. (First run only) warm up the local Hugging Face cache so repeated builds are faster:
   ```powershell
   mkdir hf_cache -Force
   ```
3. Build the images. The backend Dockerfile now downloads `Qwen/Qwen3-VL-2B-Instruct` during the image build and stores it under `/opt/models/qwen3-vl-2b-instruct`, which is exposed to FastAPI via `QWEN_VL_LOCAL_PATH`.
   ```powershell
   docker compose build backend
   ```
4. Start the full stack with GPU passthrough:
   ```powershell
   docker compose up
   ```
   Compose already sets `gpus: all`, `shm_size`, and relaxed `ulimits` for the backend service so the container can see your GPU.

## Verifying the deployment

- Open `http://localhost:3000` and upload an image. The backend should start responding immediately because the weights were loaded at container boot.
- Health endpoint (`http://localhost:8000/health`) shows `model_loaded: true` and `gpu_available: true` once the Qwen weights are in memory.
- Logs from the backend container will print `📦 Using preloaded Qwen weights at /opt/models/qwen3-vl-2b-instruct` if the baked-in snapshot is detected.

## Tuning / overrides

- To pull a different Qwen model at build time, pass a build arg: `docker compose build --build-arg QWEN_VL_MODEL=Qwen/Qwen3-VL-4B-Instruct backend`.
- To rebuild without re-downloading, keep the `hf_cache` folder intact; it is mounted into `/opt/hf_cache` inside the backend container.
- Runtime generation knobs (top_p, tokens, etc.) stay configurable through the environment variables already defined in `compose.yaml`.

## Remote inference and mode selection

This project supports two inference flows:

- Local model (Qwen3-VL) running inside the backend container.
- Remote inference: forward the image to an external inference API and return the remote result.

You can choose the inference mode per-request from the frontend (select between `Local model` and `Remote API`), or configure a default mode with the `INFERENCE_MODE` env var.

Environment variables for remote inference (optional):

- `REMOTE_INFERENCE_URL` — the HTTP(S) endpoint that accepts a multipart `file` field and returns a JSON response.
- `REMOTE_INFERENCE_API_KEY` — optional API key to send as `Authorization: Bearer <key>`.
- `INFERENCE_MODE` — optional default mode (`local` or `api`) used when the frontend does not specify a form value.

Example `.env` entries (DO NOT commit your real keys):

```
REMOTE_INFERENCE_URL=https://your-remote-inference.example/v1/infer
REMOTE_INFERENCE_API_KEY=your_remote_api_key
INFERENCE_MODE=local
```

Recommended `.env` keys for this project (create a `.env` from `.env.example`):

```
OPENAI_API_KEY=sk-...        # Optional: OpenAI key for `inference_mode=openai`
HF_TOKEN=...                 # Optional: Hugging Face token for remote HF inference
AUTO_LOAD_LOCAL_MODEL=false  # Set true to auto-load the local Qwen model at startup (requires GPU)
USE_REMOTE_INFERENCE=false   # If true, server may use remote inference when local model not loaded
```

Behavior summary:

- If the frontend sends the form field `inference_mode=api`, the backend forwards the uploaded image to `REMOTE_INFERENCE_URL` and returns the remote response (so you can compare local vs remote outputs).
- If the frontend uses `inference_mode=local` (or no field and `INFERENCE_MODE=local`), the backend runs the local Qwen model (if loaded) and returns the generated output.

Use cases:

- Quick comparison between on-device Qwen outputs and a hosted/model-provider inference.
- Fall-back to remote inference for hosts without GPU or when local model load fails.

