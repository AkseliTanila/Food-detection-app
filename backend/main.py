from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
import os
import torch
import re
import base64
import json
from fastapi import Form
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from io import BytesIO
import time
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
CONFIDENCE_REGEX = re.compile(r"confidence:\s*([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)

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

# Optionally enable remote inference fallback / OpenAI
USE_REMOTE_INFERENCE = os.environ.get("USE_REMOTE_INFERENCE", "false").lower() in ("1", "true", "yes")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
REMOTE_INFERENCE_URL = os.environ.get("REMOTE_INFERENCE_URL", "")
REMOTE_INFERENCE_API_KEY = os.environ.get("REMOTE_INFERENCE_API_KEY", "")


def load_local_model():
    """Try to load the local model into memory. This is done on-demand and may fail
    if the host doesn't have enough memory or GPU resources. Returns a dict with status.
    """
    global model, processor
    if model is not None and processor is not None:
        return {"ok": True, "msg": "Already loaded"}

    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME,torch_dtype=torch.float16,device_map="cuda")


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


startup_result = load_local_model()
if not startup_result.get("ok"):
    print(f"Startup load failed: {startup_result.get('msg')}")
else:
    print("Startup load succeeded.")


def _ensure_model_ready() -> None:
    """Raise a helpful HTTP error if inference is not ready."""
    if model is not None and processor is not None:
        return

    if USE_REMOTE_INFERENCE and HF_TOKEN:
        raise HTTPException(
            status_code=501,
            detail=(
                "Remote inference is configured but not implemented in this container. Set up HF_TOKEN and a remote client "
                "or enable local model loading via /load-model."
            ),
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "Qwen3-VL model is not loaded on the server. To enable local inference, call GET /load-model or set "
            "USE_REMOTE_INFERENCE=true with an HF_TOKEN for remote fallback."
        ),
    )


def _prepare_image(contents: bytes) -> Image.Image:
    image = Image.open(BytesIO(contents)).convert("RGB")
    if max(image.size) > MAX_IMAGE_RESOLUTION:
        image.thumbnail((MAX_IMAGE_RESOLUTION, MAX_IMAGE_RESOLUTION))
    return image


def _run_generation(image: Image.Image) -> Dict[str, Any]:
    _ensure_model_ready()

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

    return {
        "output": output_text,
        "processing_time_ms": int((end_time - start_time) * 1000),
    }


def _analyze_prepared_image(image: Image.Image, filename: str, file_size: int) -> Dict[str, Any]:
    generation = _run_generation(image)
    return {
        "success": True,
        "filename": filename,
        "size_bytes": file_size,
        "analysis": {
            **generation,
            "model_version": MODEL_NAME,
        },
    }


def _extract_confidence(output_text: str) -> Optional[float]:
    match = CONFIDENCE_REGEX.search(output_text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

# ==========================
# Helper functions
# ==========================


def parse_model_output(text: str) -> Dict[str, Any]:
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


async def call_openai_inference(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set"}

    b64 = base64.b64encode(file_bytes).decode("ascii")
    media_type = content_type or "image/jpeg"
    data_url = f"data:{media_type};base64,{b64}"

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"OpenAI request failed: {e}"}

    output_text = ""
    if isinstance(data.get("output"), list):
        parts = []
        for item in data["output"]:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                cont = item.get("content")
                if isinstance(cont, list):
                    for c in cont:
                        if isinstance(c, str):
                            parts.append(c)
                        elif isinstance(c, dict) and c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
        output_text = "\n".join([p for p in parts if p])
    if not output_text:
        choices = data.get("choices") or []
        txts = [c.get("text") for c in choices if isinstance(c, dict) and c.get("text")]
        output_text = "\n".join(txts)
    if not output_text:
        output_text = json.dumps(data)

    parsed = parse_model_output(output_text)
    model_conf = parsed.get("confidence")
    model_conf_pct = int(model_conf * 100) if model_conf is not None else None

    return {
        "output": output_text,
        "parsed": parsed,
        "model_confidence": model_conf,
        "model_confidence_percent": model_conf_pct,
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "raw_openai_response": data,
    }


async def call_remote_inference(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    if not REMOTE_INFERENCE_URL:
        return {"error": "REMOTE_INFERENCE_URL is not configured"}

    headers = {}
    if REMOTE_INFERENCE_API_KEY:
        headers["Authorization"] = f"Bearer {REMOTE_INFERENCE_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                REMOTE_INFERENCE_URL,
                headers=headers,
                files={"file": (filename, file_bytes, content_type or "image/jpeg")},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"Remote inference failed: {e}"}

    if isinstance(data, dict) and "analysis" in data:
        return data

    return {
        "success": True,
        "filename": filename,
        "size_bytes": len(file_bytes),
        "analysis": data,
    }


# ==========================
# Routes
# ==========================


@app.post("/analyze-food")
async def analyze_food(
    file: UploadFile = File(...),
    inference_mode: str = Form("local"),
) -> Dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    file_size = len(contents)
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    image = _prepare_image(contents)
    filename = file.filename or "uploaded"
    mode = (inference_mode or "local").lower().strip()

    if mode == "openai" or (mode in ("api", "remote") and OPENAI_API_KEY):
        result = await call_openai_inference(contents, filename, file.content_type)
        if result.get("error"):
            raise HTTPException(status_code=502, detail=result["error"])
        generation = result
    elif mode in ("api", "remote") and REMOTE_INFERENCE_URL:
        result = await call_remote_inference(contents, filename, file.content_type)
        if result.get("error"):
            raise HTTPException(status_code=502, detail=result["error"])
        generation = result.get("analysis", result)
    else:
        generation = _run_generation(image)

    detected_foods = []
    parsed = generation.get("parsed")
    if parsed:
        detected_foods.append({
            "name": parsed.get("dish") or "Unknown",
            "confidence": parsed.get("confidence") or 0.5,
            "calories": parsed.get("calories"),
            "protein_g": parsed.get("protein_g"),
            "carbs_g": parsed.get("carbs_g"),
            "fat_g": parsed.get("fat_g"),
        })

    total_nutrition = None
    if detected_foods:
        total_cal = sum(f.get("calories") or 0 for f in detected_foods)
        total_pro = sum(f.get("protein_g") or 0 for f in detected_foods)
        total_carbs = sum(f.get("carbs_g") or 0 for f in detected_foods)
        total_fat = sum(f.get("fat_g") or 0 for f in detected_foods)
        if any([total_cal, total_pro, total_carbs, total_fat]):
            total_nutrition = {
                "calories": total_cal,
                "protein_g": total_pro,
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


@app.get("/batch-test")
def batch_test(folder: Optional[str] = None) -> Dict[str, Any]:
    """Run the vision model against every supported image in a folder and summarize the results."""
    _ensure_model_ready()

    target_dir = os.path.abspath(folder or DEFAULT_TEST_PHOTO_DIR)
    if not os.path.isdir(target_dir):
        raise HTTPException(status_code=404, detail=f"Folder '{target_dir}' does not exist")

    image_files = [
        f for f in os.listdir(target_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise HTTPException(status_code=400, detail="No supported images found in the specified folder")

    results: List[Dict[str, Any]] = []
    confidence_values: List[float] = []
    total_latency = 0

    for filename in sorted(image_files):
        file_path = os.path.join(target_dir, filename)
        try:
            with open(file_path, "rb") as fh:
                contents = fh.read()

            image = _prepare_image(contents)
            analysis = _analyze_prepared_image(image, filename, len(contents))
            parsed_confidence = _extract_confidence(analysis["analysis"].get("output", ""))
            if parsed_confidence is not None:
                confidence_values.append(parsed_confidence)
                analysis["analysis"]["parsed_confidence"] = parsed_confidence

            total_latency += analysis["analysis"].get("processing_time_ms", 0)
            results.append(analysis)
        except Exception as exc:  # noqa: BLE001 - surface the failure content to the caller
            results.append({
                "success": False,
                "filename": filename,
                "error": str(exc),
            })

    success_count = sum(1 for item in results if item.get("success"))
    failure_count = len(results) - success_count
    avg_confidence = (
        round(sum(confidence_values) / len(confidence_values), 2)
        if confidence_values else None
    )
    avg_latency = (
        int(total_latency / success_count) if success_count else None
    )

    return {
        "folder": target_dir,
        "total_images": len(results),
        "summary": {
            "success_count": success_count,
            "failure_count": failure_count,
            "average_confidence_percent": avg_confidence,
            "average_processing_time_ms": avg_latency,
        },
        "results": results,
    }
