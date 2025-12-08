from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
import os
import torch
import re
from fastapi import Form
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from io import BytesIO
import time
import base64
import json
import httpx

app = FastAPI(title="Food Detection API", version="1.0.0")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_MODEL_PATH = os.environ.get("QWEN_VL_LOCAL_PATH")
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
resolved_local_path = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH) else None
MODEL_NAME = resolved_local_path or os.environ.get("QWEN_VL_MODEL", DEFAULT_MODEL_ID)
if resolved_local_path:
    print(f"Using preloaded Qwen weights at {resolved_local_path}")
else:
    print(f"Using Hugging Face repo {MODEL_NAME}")

# Generation hyperparameters
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", 0.8))
GEN_TOP_K = int(os.environ.get("GEN_TOP_K", 20))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", 0.4))
GEN_REPETITION_PENALTY = float(os.environ.get("GEN_REPETITION_PENALTY", 1.0))
GEN_PRESENCE_PENALTY = float(os.environ.get("GEN_PRESENCE_PENALTY", 1.5))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", 120))
MAX_IMAGE_RESOLUTION = int(os.environ.get("MAX_IMAGE_RESOLUTION", 1024))
DEFAULT_TEST_PHOTO_DIR = os.path.abspath(
    os.environ.get(
        "TEST_PHOTO_DIR",
        os.path.join(os.path.dirname(__file__), "..", "photoai", "src", "app", "testPhotos")
    )
)
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

USER_PROMPT = (
    "You are a food recognition model. Respond using exactly this template:\n"
    "Dish: <concise name>\n"
    "Ingredients: comma-separated list\n"
    "Nutrition (per serving, approximate): Calories <number> kcal; Protein <number> g; Carbs <number> g; Fat <number> g\n"
    "Confidence: <number>% (use 95-100 only when the dish is unmistakable; drop to 60-80 if multiple foods appear or plating is atypical; go below 60 when unsure). Never repeat the same confidence twice in a row across different photos. add or subtract from -2% to +2%\n"
    "\n"
    "Rules:\n"
    "- Do NOT show any reasoning.\n"
    "- Confidence must reflect visual certainty:\n"
    "  * 95-100%: extremely obvious single food (banana, fried egg, pizza slice).\n"
    "  * 80-94%: identifiable but has variations or partial occlusion.\n"
    "  * 60-79%: mixed dishes with some ambiguity.\n"
    "  * 30-59%: unclear angle or visually similar alternatives.\n"
    "  * 0-29%: very high uncertainty.\n"
    "- Nutrition numbers must vary between dishes and be realistic.\n"
)

processor = None
model = None

# Optionally enable remote inference fallback
USE_REMOTE_INFERENCE = os.environ.get("USE_REMOTE_INFERENCE", "false").lower() in ("1", "true", "yes")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-vision-preview")
REMOTE_INFERENCE_URL = os.environ.get("REMOTE_INFERENCE_URL", "")
REMOTE_INFERENCE_API_KEY = os.environ.get("REMOTE_INFERENCE_API_KEY", "")
AUTO_LOAD_LOCAL_MODEL = os.environ.get("AUTO_LOAD_LOCAL_MODEL", "true").lower() in ("1", "true", "yes")


def load_local_model():
    """Try to load the local model into memory. This is done on-demand and may fail
    if the host doesn't have enough memory or GPU resources. Returns a dict with status.
    """
    global model, processor
    if model is not None and processor is not None:
        return {"ok": True, "msg": "Already loaded"}

    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
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


# Only auto-load if enabled
if AUTO_LOAD_LOCAL_MODEL:
    startup_result = load_local_model()
    if not startup_result.get("ok"):
        print(f"Startup load failed: {startup_result.get('msg')}")
    else:
        print("Startup load succeeded.")
else:
    print("AUTO_LOAD_LOCAL_MODEL is disabled; skipping local model load at startup.")

# ==========================
# Helper functions for parsing and confidence
# ==========================


def parse_model_output(text: str) -> Dict[str, Any]:
    """
    Parse model output text into structured fields.
    Extracts: dish, ingredients, nutrition (calories, protein_g, carbs_g, fat_g), and confidence.
    """
    result = {
        "dish": None,
        "ingredients": None,
        "calories": None,
        "protein_g": None,
        "carbs_g": None,
        "fat_g": None,
        "confidence": None,
    }

    lines = (text or "").split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("dish:"):
            result["dish"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("ingredients:"):
            result["ingredients"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("nutrition"):
            # Extract nutrition from lines like:
            # Nutrition (per serving, approximate): Calories 280 kcal; Protein 18 g; Carbs 32 g; Fat 9 g
            nutrition_text = line.split(":", 1)[1] if ":" in line else ""
            nutrients = {}
            for part in nutrition_text.split(";"):
                part = part.strip()
                if "calories" in part.lower():
                    try:
                        nutrients["calories"] = int("".join(c for c in part if c.isdigit()))
                    except ValueError:
                        pass
                elif "protein" in part.lower():
                    try:
                        nutrients["protein_g"] = float("".join(c for c in part.replace("g", "") if c.isdigit() or c == "."))
                    except ValueError:
                        pass
                elif "carbs" in part.lower():
                    try:
                        nutrients["carbs_g"] = float("".join(c for c in part.replace("g", "") if c.isdigit() or c == "."))
                    except ValueError:
                        pass
                elif "fat" in part.lower():
                    try:
                        nutrients["fat_g"] = float("".join(c for c in part.replace("g", "") if c.isdigit() or c == "."))
                    except ValueError:
                        pass
            result.update(nutrients)
        elif line.lower().startswith("confidence:"):
            conf_text = line.split(":", 1)[1].strip()
            try:
                result["confidence"] = float(conf_text.replace("%", "").strip()) / 100.0
            except ValueError:
                pass

    return result


def extract_logprobs_from_openai_response(response_json: Dict) -> Optional[List[float]]:
    """
    Extract token logprobs from an OpenAI API response.
    Tries multiple possible response shapes.
    """
    if not response_json:
        return None

    # Try standard Completions shape: response_json["choices"][0]["logprobs"]["content"][i]["logprob"]
    try:
        choices = response_json.get("choices", [])
        if choices:
            logprobs_obj = choices[0].get("logprobs")
            if logprobs_obj and "content" in logprobs_obj:
                return [item.get("logprob", 0.0) for item in logprobs_obj["content"] if item.get("logprob") is not None]
    except (KeyError, TypeError, IndexError):
        pass

    # Try Responses shape or custom shape
    try:
        if "logprobs" in response_json:
            lp = response_json["logprobs"]
            if isinstance(lp, list):
                return lp
            elif isinstance(lp, dict) and "tokens" in lp:
                return lp.get("values", [])
    except (KeyError, TypeError):
        pass

    return None


def compute_confidence_from_logprobs(logprobs: List[float]) -> Optional[float]:
    """
    Compute a confidence score from token logprobs using geometric mean.
    Returns a value in [0, 1] or None if computation fails.
    """
    if not logprobs or len(logprobs) == 0:
        return None
    try:
        mean_logp = sum(logprobs) / len(logprobs)
        confidence = min(max(0.0, 1.0 + mean_logp), 1.0)  # Clamp to [0, 1]; assume logprobs ~ [-1, 0]
        return confidence
    except (ValueError, TypeError):
        return None


def _prepare_image(image_bytes: bytes) -> Image.Image:
    """Helper to prepare image from bytes."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > MAX_IMAGE_RESOLUTION:
        image.thumbnail((MAX_IMAGE_RESOLUTION, MAX_IMAGE_RESOLUTION))
    return image


def _run_generation(image: Image.Image) -> Dict[str, Any]:
    """Run local Qwen3-VL generation on an image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
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

    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        generated_trimmed: List[torch.Tensor] = []
        for in_ids, out_ids in zip(input_ids, generated_ids):
            in_len = in_ids.shape[0]
            generated_trimmed.append(out_ids[in_len:])
    else:
        generated_trimmed = [generated_ids[0]] if len(generated_ids) else []

    output_texts = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_texts[0] if output_texts else ""
    end_time = time.time()

    # Parse output into structured form
    parsed = parse_model_output(output_text)

    # Extract confidence from parsed output
    confidence_value = parsed.get("confidence")
    if confidence_value is not None:
        model_confidence = confidence_value
        model_confidence_percent = int(confidence_value * 100)
    else:
        model_confidence = None
        model_confidence_percent = None

    return {
        "output": output_text,
        "parsed": parsed,
        "model_confidence": model_confidence,
        "model_confidence_percent": model_confidence_percent,
        "processing_time_ms": int((end_time - start_time) * 1000),
    }


async def call_openai_inference(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """
    Call OpenAI Vision API with the image and return normalized analysis.
    """
    if not OPENAI_API_KEY:
        return {
            "error": "OPENAI_API_KEY not set",
            "output": None,
            "parsed": None,
            "model_confidence": None,
            "model_confidence_percent": None,
        }

    # Encode image as base64 data URL
    b64_image = base64.b64encode(file_bytes).decode("utf-8")
    media_type = content_type or "image/jpeg"
    data_url = f"data:{media_type};base64,{b64_image}"

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {"type": "text", "text": USER_PROMPT},
                            ],
                        }
                    ],
                    "max_tokens": GEN_MAX_NEW_TOKENS,
                },
            )

        response.raise_for_status()
        response_json = response.json()
        end_time = time.time()

        output_text = ""
        if "choices" in response_json and len(response_json["choices"]) > 0:
            output_text = response_json["choices"][0].get("message", {}).get("content", "")

        parsed = parse_model_output(output_text)

        # Try to extract logprobs and compute confidence
        model_confidence = None
        model_confidence_percent = None

        logprobs = extract_logprobs_from_openai_response(response_json)
        if logprobs:
            model_confidence = compute_confidence_from_logprobs(logprobs)
            if model_confidence is not None:
                model_confidence_percent = int(model_confidence * 100)

        # Fallback: use parsed confidence if available
        if model_confidence is None and parsed.get("confidence") is not None:
            model_confidence = parsed["confidence"]
            model_confidence_percent = int(model_confidence * 100)

        return {
            "output": output_text,
            "parsed": parsed,
            "model_confidence": model_confidence,
            "model_confidence_percent": model_confidence_percent,
            "processing_time_ms": int((end_time - start_time) * 1000),
            "raw_openai_response": response_json,
        }

    except Exception as e:
        return {
            "error": str(e),
            "output": None,
            "parsed": None,
            "model_confidence": None,
            "model_confidence_percent": None,
        }


async def call_remote_inference(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """
    Forward image to a remote inference endpoint and normalize the response.
    """
    if not REMOTE_INFERENCE_URL:
        return {
            "error": "REMOTE_INFERENCE_URL not configured",
            "output": None,
            "parsed": None,
            "model_confidence": None,
            "model_confidence_percent": None,
        }

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (filename, file_bytes, content_type or "image/jpeg")}
            headers = {}
            if REMOTE_INFERENCE_API_KEY:
                headers["Authorization"] = f"Bearer {REMOTE_INFERENCE_API_KEY}"

            response = await client.post(
                REMOTE_INFERENCE_URL,
                files=files,
                headers=headers,
            )

        response.raise_for_status()
        response_json = response.json()
        end_time = time.time()

        # Normalize response
        output_text = response_json.get("output") or ""
        parsed = parse_model_output(output_text)

        model_confidence = None
        model_confidence_percent = None

        if response_json.get("model_confidence") is not None:
            model_confidence = response_json["model_confidence"]
            model_confidence_percent = int(model_confidence * 100)
        elif parsed.get("confidence") is not None:
            model_confidence = parsed["confidence"]
            model_confidence_percent = int(model_confidence * 100)

        return {
            "output": output_text,
            "parsed": parsed,
            "model_confidence": model_confidence,
            "model_confidence_percent": model_confidence_percent,
            "processing_time_ms": int((end_time - start_time) * 1000),
        }

    except Exception as e:
        return {
            "error": str(e),
            "output": None,
            "parsed": None,
            "model_confidence": None,
            "model_confidence_percent": None,
        }


# ==========================
# Routes
# ==========================


@app.post("/analyze-food")
async def analyze_food(
    file: UploadFile = File(...),
    inference_mode: str = Form(default="api"),
) -> Dict:
    """
    Analyze food in an image.

    inference_mode:
      - "local": use local Qwen3-VL model
      - "api" or "openai": use OpenAI Vision API
      - "remote": use REMOTE_INFERENCE_URL endpoint
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    file_size = len(contents)
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    image = _prepare_image(contents)
    filename = file.filename or "unknown"

    # Routing logic
    if inference_mode == "local":
        # Try local model
        if model is None or processor is None:
            raise HTTPException(
                status_code=503,
                detail="Local model not loaded. Call GET /load-model first or use inference_mode=api.",
            )
        generation = _run_generation(image)
    elif inference_mode in ("api", "openai"):
        # Use OpenAI
        generation = await call_openai_inference(contents, filename, file.content_type)
        if "error" in generation:
            raise HTTPException(status_code=503, detail=generation["error"])
    elif inference_mode == "remote":
        # Use remote endpoint
        generation = await call_remote_inference(contents, filename, file.content_type)
        if "error" in generation:
            raise HTTPException(status_code=503, detail=generation["error"])
    else:
        raise HTTPException(status_code=400, detail=f"Unknown inference_mode: {inference_mode}")

    # Build detected_foods from parsed output
    detected_foods = []
    if generation.get("parsed"):
        parsed = generation["parsed"]
        food_item = {
            "name": parsed.get("dish") or "Unknown",
            "confidence": parsed.get("confidence") or 0.5,
            "calories": parsed.get("calories"),
            "protein_g": parsed.get("protein_g"),
            "carbs_g": parsed.get("carbs_g"),
            "fat_g": parsed.get("fat_g"),
        }
        detected_foods.append(food_item)

    # Calculate total nutrition
    total_nutrition = None
    if detected_foods:
        total_calories = sum(f.get("calories") or 0 for f in detected_foods)
        total_protein = sum(f.get("protein_g") or 0 for f in detected_foods)
        total_carbs = sum(f.get("carbs_g") or 0 for f in detected_foods)
        total_fat = sum(f.get("fat_g") or 0 for f in detected_foods)

        if any([total_calories, total_protein, total_carbs, total_fat]):
            total_nutrition = {
                "calories": total_calories,
                "protein_g": total_protein,
                "carbs_g": total_carbs,
                "fat_g": total_fat,
            }

    return {
        "success": True,
        "filename": filename,
        "size_bytes": file_size,
        "analysis": {
            "output": generation.get("output"),
            "detected_foods": detected_foods,
            "total_nutrition": total_nutrition,
            "model_confidence": generation.get("model_confidence"),
            "model_confidence_percent": generation.get("model_confidence_percent"),
            "processing_time_ms": generation.get("processing_time_ms"),
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
