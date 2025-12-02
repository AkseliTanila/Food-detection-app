# Food-detection-app

AI based food detection application that uses Qwen3-VL for on-device multimodal inference.

Application consists of a Next.js frontend, and Python backend. Application is Dockerized. Users are able to upload images to the web interface, and run the AI analysis of that image. 

Application is tailored to handle food related images, and return following parameters:
- Dish
- Ingredients
- Nutrients (per serving approximates)
- Confidence (of the AI analysis)


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
- Logs from the backend container will print `Using preloaded Qwen weights at /opt/models/qwen3-vl-2b-instruct` if the baked-in snapshot is detected.

## Tuning / overrides

- To pull a different Qwen model at build time, pass a build arg: `docker compose build --build-arg QWEN_VL_MODEL=Qwen/Qwen3-VL-4B-Instruct backend`.
- To rebuild without re-downloading, keep the `hf_cache` folder intact; it is mounted into `/opt/hf_cache` inside the backend container.
- Runtime generation knobs (top_p, tokens, etc.) stay configurable through the environment variables already defined in `compose.yaml`.
