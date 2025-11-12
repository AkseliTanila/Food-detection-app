from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

app = FastAPI(title="Food Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from the Python backend!"}

@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)) -> Dict:
    """
    Endpoint to receive food image and return dummy analysis
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG, WEBP)"
        )

    # Read file content
    contents = await file.read()
    file_size = len(contents)

    # Validate file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 10MB limit"
        )

    # Return dummy food analysis response
    return {
        "success": True,
        "filename": file.filename,
        "size_bytes": file_size,
        "analysis": {
            "detected_foods": [
                {
                    "name": "Grilled Chicken Breast",
                    "confidence": 0.92,
                    "calories": 165,
                    "protein_g": 31,
                    "carbs_g": 0,
                    "fat_g": 3.6
                },
                {
                    "name": "Steamed Broccoli",
                    "confidence": 0.88,
                    "calories": 55,
                    "protein_g": 3.7,
                    "carbs_g": 11.2,
                    "fat_g": 0.6
                },
                {
                    "name": "Brown Rice",
                    "confidence": 0.85,
                    "calories": 216,
                    "protein_g": 5,
                    "carbs_g": 45,
                    "fat_g": 1.8
                }
            ],
            "total_nutrition": {
                "calories": 436,
                "protein_g": 39.7,
                "carbs_g": 56.2,
                "fat_g": 6.0
            },
            "processing_time_ms": 342,
            "model_version": "v1.0-demo"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "food-detection-api"}