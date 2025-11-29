from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import os
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from io import BytesIO
import time

app = FastAPI(title="Food Detection API", version="1.0.0")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI model once at startup. If a baked-in local path exists we prefer that
LOCAL_MODEL_PATH = os.environ.get("QWEN_VL_LOCAL_PATH")
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
resolved_local_path = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH) else None
MODEL_NAME = resolved_local_path or os.environ.get("QWEN_VL_MODEL", DEFAULT_MODEL_ID)
if resolved_local_path:
    print(f"📦 Using preloaded Qwen weights at {resolved_local_path}")
else:
    print(f"🌐 Using Hugging Face repo {MODEL_NAME}")

# Generation hyperparameters (defaults taken from Qwen3-VL recommendations)
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", 0.8))
GEN_TOP_K = int(os.environ.get("GEN_TOP_K", 20))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", 0.25))
GEN_REPETITION_PENALTY = float(os.environ.get("GEN_REPETITION_PENALTY", 1.0))
GEN_PRESENCE_PENALTY = float(os.environ.get("GEN_PRESENCE_PENALTY", 1.5))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", 80))
MAX_IMAGE_RESOLUTION = int(os.environ.get("MAX_IMAGE_RESOLUTION", 1024))

print("🔍 Trying to load the local model at startup (user requested). This may take a while and can OOM.")
processor = None
model = None

# Optionally enable remote inference fallback
USE_REMOTE_INFERENCE = os.environ.get("USE_REMOTE_INFERENCE", "false").lower() in ("1", "true", "yes")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def load_local_model():
    """Try to load the local model into memory. This is done on-demand and may fail
    if the host doesn't have enough memory or GPU resources. Returns a dict with status.
    """
    global model, processor
    if model is not None and processor is not None:
        return {"ok": True, "msg": "Already loaded"}

    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME,torch_dtype=torch.float16,device_map="cuda")


        # Use bitsandbytes quantization config if we requested 8-bit mode
        quant_config = None

        load_kwargs = {"device_map": "cuda"}
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        print(f"Attempting to load model {MODEL_NAME} with kwargs={load_kwargs}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, **load_kwargs)
        model.eval()
        return {"ok": True, "msg": "Model loaded"}
    except Exception as e:
        # Keep processor/model None so server remains up
        processor = None
        model = None
        return {"ok": False, "msg": str(e)}


# If the user asked for automatic loading, attempt it now
startup_result = load_local_model()
if not startup_result.get("ok"):
    print(f"⚠️ Startup load failed: {startup_result.get('msg')}")
else:
    print("✅ Startup load succeeded.")

# ==========================
# Routes
# ==========================


@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)) -> Dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    file_size = len(contents)
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    image = Image.open(BytesIO(contents)).convert("RGB")
    # Downscale extremely large photos to keep the vision encoder fast
    if max(image.size) > MAX_IMAGE_RESOLUTION:
        image.thumbnail((MAX_IMAGE_RESOLUTION, MAX_IMAGE_RESOLUTION))

    if model is None or processor is None:
        # Model is not available locally. If the server is configured to use remote
        # inference and a HF token is present, that would be used (not implemented yet).
        if USE_REMOTE_INFERENCE and HF_TOKEN:
            raise HTTPException(status_code=501, detail="Remote inference is configured but not implemented in this container. Set up HF_TOKEN and a remote client or enable local model loading via /load-model.")

        # Otherwise instruct the user to either load the local model or configure remote inference
        raise HTTPException(
            status_code=503,
            detail=("Qwen3-VL model is not loaded on the server. To enable local inference, call GET /load-model or set USE_REMOTE_INFERENCE=true with an HF_TOKEN for remote fallback.")
        )

    user_prompt = (
        "You are a food recognition model. Respond using exactly this template without extra commentary: \n"
        "Dish: <concise name>\n"
        "Ingredients: comma-separated list\n"
        "Nutrition (per serving, approximate): Calories <number> kcal; Protein <number> g; Carbs <number> g; Fat <number> g\n"
        "Confidence: <number>%\n"
        "Before producing the template, quickly reason about portion size and per-ingredient macros so each dish gets unique, realistic numbers.\n"
        "Examples (do not copy the numbers):\n"
        "Dish: Sushi roll\nIngredients: rice, nori, salmon, avocado\nNutrition (per serving, approximate): Calories 280 kcal; Protein 18 g; Carbs 32 g; Fat 9 g\nConfidence: 88%\n"
        "Dish: Greek salad\nIngredients: lettuce, tomato, cucumber, olives, feta cheese, dressing\nNutrition (per serving, approximate): Calories 190 kcal; Protein 6 g; Carbs 10 g; Fat 14 g\nConfidence: 74%\n"
        "Requirements: no prefatory phrases like 'This dish is'; no disclaimers; always supply a numeric confidence between 0 and 100; nutrition numbers must differ between dishes and be realistic for the detected ingredients (never reuse the same defaults)."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    start_time = time.time()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Build generation kwargs; `presence_penalty` is not accepted by most HF models' generate()
    gen_kwargs = {
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "top_p": GEN_TOP_P,
        "top_k": GEN_TOP_K,
        "temperature": GEN_TEMPERATURE,
        "repetition_penalty": GEN_REPETITION_PENALTY,
        "do_sample": True,
    }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # If inputs include input_ids, trim the prompt tokens from the generated ids
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        # generated_ids is (batch, seq_len); remove the input prefix per example
        generated_trimmed = []
        for in_ids, out_ids in zip(input_ids, generated_ids):
            in_len = in_ids.shape[0]
            generated_trimmed.append(out_ids[in_len:])
    else:
        generated_trimmed = [generated_ids[0]] if len(generated_ids) else []

    # Use processor.batch_decode to get a clean string
    output_texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_text = output_texts[0] if output_texts else ""
    end_time = time.time()

    return {
        "success": True,
        "filename": file.filename,
        "size_bytes": file_size,
        "analysis": {
            "output": output_text,
            "processing_time_ms": int((end_time - start_time) * 1000),
            "model_version": MODEL_NAME,
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "food-detection-api",
        "model_loaded": model is not None and processor is not None,
        "gpu_available": torch.cuda.is_available(),
        "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
    }


@app.get("/load-model")
def load_model_endpoint():
    """Trigger loading of the local model on demand. This may use a lot of memory and can OOM if the host is not capable."""
    result = load_local_model()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("msg", "unknown error"))
    return {"status": "loaded", "message": result.get("msg")}
