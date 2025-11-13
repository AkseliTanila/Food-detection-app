from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
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

# Load AI model once at startup
MODEL_NAME = "nateraw/food"

print(f"🔍 Loading model: {MODEL_NAME} ...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
print("✅ Model loaded successfully.")

# ==========================
# Routes
# ==========================
@app.get("/")
def read_root():
    return {"message": "Hello from the Python backend!"}


@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)) -> Dict:
    """
    Analyze uploaded food image with AI model.
    """
    # --- Validate file ---
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, WEBP)")

    contents = await file.read()
    file_size = len(contents)

    if file_size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    image = Image.open(BytesIO(contents)).convert("RGB")

    # --- Run model ---
    start_time = time.time()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        topk = torch.topk(probs, k=3)

    detected_foods = []
    for score, idx in zip(topk.values[0], topk.indices[0]):
        label = model.config.id2label[idx.item()]
        detected_foods.append({
            "name": label,
            "confidence": round(float(score.item()), 3)
        })

    elapsed_ms = int((time.time() - start_time) * 1000)

    # --- Build response ---
    return {
        "success": True,
        "filename": file.filename,
        "size_bytes": file_size,
        "analysis": {
            "detected_foods": detected_foods,
            "model_version": MODEL_NAME,
            "processing_time_ms": elapsed_ms,

        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "food-detection-api"}
