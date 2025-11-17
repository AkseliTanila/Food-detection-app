from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO
import time
import os
import httpx
import re

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

    # --- Optional: fetch nutrition estimates ---
    # Controlled via env var NUTRITION_PROVIDER (e.g. "nutritionix").
    nutrition_provider = os.getenv("NUTRITION_PROVIDER", "")
    nutrition_estimates = []

    async def fetch_nutrition_nutritionix(query: str):
        """Call Nutritionix natural language API to estimate nutrients for a query like '100 g pizza'."""
        app_id = os.getenv("NUTRITIONIX_APP_ID")
        app_key = os.getenv("NUTRITIONIX_API_KEY")
        if not app_id or not app_key:
            return None

        url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
        headers = {
            "x-app-id": app_id,
            "x-app-key": app_key,
            "Content-Type": "application/json",
        }
        payload = {"query": query}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json()
            foods = data.get("foods") or []
            if not foods:
                return None
            f = foods[0]
            return {
                "calories": f.get("nf_calories"),
                "fat_g": f.get("nf_total_fat"),
                "carbs_g": f.get("nf_total_carbohydrate"),
                "protein_g": f.get("nf_protein"),
                "serving_weight_grams": f.get("serving_weight_grams"),
                "serving_qty": f.get("serving_qty"),
                "serving_unit": f.get("serving_unit"),
                "source": "nutritionix",
            }
        except Exception as e:
            print(f"[nutrition][fatsecret] exception while fetching nutrition for query '{query}': {e}")
            return None

    async def fetch_nutrition_fatsecret(query: str):
        """Call FatSecret Platform API to estimate nutrients for a query like '100 g pizza'.

        This implementation uses the OAuth2 client credentials flow to get an access
        token, then calls the FatSecret REST endpoint with method=foods.search.
        The response parsing is defensive because FatSecret's response shapes can
        vary; we try to extract calories/protein/carbs/fat from the first matching
        serving (preferring a 100 g serving when available).
        """
        client_id = os.getenv("FATSECRET_CLIENT_ID")
        client_secret = os.getenv("FATSECRET_CLIENT_SECRET")
        if not client_id or not client_secret:
            return None

        token_url = os.getenv("FATSECRET_TOKEN_URL", "https://oauth.fatsecret.com/connect/token")
        api_url = os.getenv("FATSECRET_API_URL", "https://platform.fatsecret.com/rest/server.api")

        # We'll try a couple of query variants (prefer 100 g), and if search returns a food id
        # we'll call the food detail endpoint to fetch serving/nutrition fields.
        query_variants = [query]
        # also try without weight suffix
        if query.startswith("100 g "):
            query_variants.append(query[len("100 g "):])

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Obtain access token via client credentials
            try:
                token_resp = await client.post(
                    token_url,
                    data={"grant_type": "client_credentials"},
                    auth=(client_id, client_secret),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            except Exception as e:
                print(f"[nutrition][fatsecret] token request failed: {e}")
                return None

            if token_resp.status_code != 200:
                print(f"[nutrition][fatsecret] token request status: {token_resp.status_code} - {token_resp.text}")
                return None
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                print(f"[nutrition][fatsecret] no access_token in token response: {token_data}")
                return None

            headers = {"Authorization": f"Bearer {access_token}"}

            # helper to parse nutrition from a food-like dict
            def _get_num(d, keys):
                if not isinstance(d, dict):
                    return None
                for k in keys:
                    v = d.get(k)
                    if v is None:
                        continue
                    try:
                        return float(v)
                    except Exception:
                        try:
                            return float(str(v).replace(',', '').strip())
                        except Exception:
                            continue
                return None

            async def parse_food_and_get_nutrition(food_item):
                # food_item may be a dict with servings or nutrition
                if not isinstance(food_item, dict):
                    return None

                # try to get servings/serving
                servings = food_item.get("servings") or food_item.get("serving")
                serving = None
                if isinstance(servings, dict):
                    s = servings.get("serving")
                    if isinstance(s, list) and s:
                        # prefer 100 g if present
                        for sv in s:
                            desc = (sv.get("serving_description") or "").lower()
                            if "100" in desc or "g" in desc:
                                serving = sv
                                break
                        if not serving:
                            serving = s[0]
                    elif isinstance(s, dict):
                        serving = s
                elif isinstance(servings, list) and servings:
                    serving = servings[0]
                else:
                    # maybe nutrition fields are directly on food_item
                    serving = food_item

                calories = _get_num(serving, ["calories", "calorie", "nf_calories", "calories_in_100g", "calories_per_serving"]) or _get_num(food_item, ["calories"])
                protein = _get_num(serving, ["protein", "protein_in_grams", "nf_protein"]) or _get_num(food_item, ["protein"])
                carbs = _get_num(serving, ["carbohydrate", "carbs", "carbohydrate_in_grams", "nf_total_carbohydrate"]) or _get_num(food_item, ["carbohydrate"])
                fat = _get_num(serving, ["fat", "fat_in_grams", "nf_total_fat"]) or _get_num(food_item, ["fat"])

                if calories is None and protein is None and carbs is None and fat is None:
                    return None

                return {
                    "calories": calories,
                    "fat_g": fat,
                    "carbs_g": carbs,
                    "protein_g": protein,
                    "serving_description": serving.get("serving_description") if isinstance(serving, dict) else None,
                    "source": "fatsecret",
                }

            # Try each variant
            last_search = None
            last_detail = None
            for q in query_variants:
                try:
                    params = {"method": "foods.search", "search_expression": q, "format": "json"}
                    # include optional region/language params
                    region = os.getenv("FATSECRET_REGION")
                    language = os.getenv("FATSECRET_LANGUAGE")
                    if region:
                        params["region"] = region
                    if language:
                        params["language"] = language

                    resp = await client.get(api_url, params=params, headers=headers)
                except Exception as e:
                    print(f"[nutrition][fatsecret] search request failed for '{q}': {e}")
                    continue

                if resp.status_code != 200:
                    print(f"[nutrition][fatsecret] search status {resp.status_code} for '{q}': {resp.text}")
                    last_search = {"status": resp.status_code, "text": resp.text}
                    continue

                try:
                    data = resp.json()
                except Exception:
                    # keep the raw text for debugging
                    last_search = {"status": resp.status_code, "text": resp.text}
                    continue
                last_search = data
                # try to locate a food item in the response
                foods = None
                if isinstance(data, dict):
                    # FatSecret often nests results under 'foods' -> 'food' or directly 'food'
                    foods = data.get("foods") or data.get("food") or data.get("foods_search_result")

                food_item = None
                if isinstance(foods, dict):
                    candidate = foods.get("food")
                    if isinstance(candidate, list) and candidate:
                        food_item = candidate[0]
                    elif isinstance(candidate, dict):
                        food_item = candidate
                elif isinstance(foods, list) and foods:
                    food_item = foods[0]

                if not food_item:
                    # nothing found for this variant, try next
                    continue

                # Try parse nutrition directly from search result
                nutrition = await parse_food_and_get_nutrition(food_item)
                if nutrition:
                    return nutrition

                # If no nutrition yet, try to get a food_id and call food.get (detail)
                # FatSecret may expose 'food_id' or 'id'
                food_id = None
                if isinstance(food_item, dict):
                    food_id = food_item.get("food_id") or food_item.get("id") or food_item.get("food_id")

                if food_id:
                    try:
                        detail_params = {"method": "food.get", "food_id": food_id, "format": "json"}
                        if region:
                            detail_params["region"] = region
                        if language:
                            detail_params["language"] = language
                        detail_resp = await client.get(api_url, params=detail_params, headers=headers)
                    except Exception as e:
                        print(f"[nutrition][fatsecret] detail request failed for id {food_id}: {e}")
                        detail_resp = None

                    if detail_resp:
                        try:
                            detail_data = detail_resp.json()
                        except Exception:
                            detail_data = {"status": detail_resp.status_code, "text": detail_resp.text}
                        last_detail = detail_data
                    if detail_resp and detail_resp.status_code == 200:
                        # try to find the food object inside detail_data
                        detail_food = None
                        if isinstance(detail_data, dict):
                            detail_food = detail_data.get("food") or detail_data.get("foods")
                        # if it's a dict with 'food', drill down
                        if isinstance(detail_food, dict) and detail_food.get("food"):
                            detail_food = detail_food.get("food")

                        nutrition = await parse_food_and_get_nutrition(detail_food or detail_data)
                        if nutrition:
                            return nutrition

                        # If still nothing, log the detail response to help debugging
                        print(f"[nutrition][fatsecret] couldn't parse nutrition from detail for id {food_id}: {detail_resp.text if detail_resp is not None else 'no response'}")
                else:
                    # No id available and couldn't parse nutrition
                    print(f"[nutrition][fatsecret] search returned a food item but no parsable nutrition and no food_id for query '{q}': {food_item}")

            # after trying all variants - return debug info so frontend can show raw responses
            print(f"[nutrition][fatsecret] no nutrition found for any variant of query '{query}'. Attaching raw responses to debug output.")
            # If FatSecret returned an error payload (for example IP restriction), add a friendly note
            error_note = None
            if isinstance(last_search, dict) and last_search.get("error"):
                err = last_search.get("error")
                # Common FatSecret error code 21 indicates invalid IP/address on some deployments
                if isinstance(err, dict) and err.get("code") == 21:
                    error_note = (
                        "FatSecret rejected the request due to an invalid/unauthorized IP address. "
                        "Check your FatSecret app settings and add the server's outbound IP to the allowed IPs, "
                        "or run the backend from an IP that is whitelisted."
                    )

            debug_obj = {
                "debug": True,
                "raw_search": last_search,
                "raw_detail": last_detail,
                "source": "fatsecret",
            }
            if error_note:
                debug_obj["error_note"] = error_note

            return debug_obj

    # For now we'll query nutrition for the top detected label using a 100g reference
    if detected_foods:
        top_label = detected_foods[0]["name"]
        # Normalize label for nutrition queries: replace underscores/dashes with spaces and remove extra punctuation
        def _clean_label(s: str) -> str:
            if not isinstance(s, str):
                return s
            s = s.replace('_', ' ').replace('-', ' ')
            s = re.sub(r"[^0-9A-Za-z\s]", ' ', s)
            s = re.sub(r"\s+", ' ', s).strip()
            return s

        top_label_clean = _clean_label(top_label)
        query = f"100 g {top_label_clean}"
        if nutrition_provider.lower() == "nutritionix":
            nutrition = await fetch_nutrition_nutritionix(query)
        elif nutrition_provider.lower() == "fatsecret":
            nutrition = await fetch_nutrition_fatsecret(query)
        else:
            nutrition = None

        if nutrition:
            # Attach nutrition fields to the top detected food so the frontend can render them
            detected_foods[0]["calories"] = nutrition.get("calories")
            detected_foods[0]["protein_g"] = nutrition.get("protein_g")
            detected_foods[0]["carbs_g"] = nutrition.get("carbs_g")
            detected_foods[0]["fat_g"] = nutrition.get("fat_g")

            nutrition_estimates.append({
                "name": top_label,
                "query": query,
                "nutrition": nutrition,
                "note": "Estimates are approximate and based on a 100 g reference amount."
            })
        else:
            print(f"[nutrition] {nutrition_provider} lookup returned no data for: {top_label}")

    # --- Build response ---
    analysis = {
        "detected_foods": detected_foods,
        "model_version": MODEL_NAME,
        "processing_time_ms": elapsed_ms,
    }

    if nutrition_estimates:
        analysis["nutrition_estimates"] = nutrition_estimates

    # Compute total_nutrition by summing per-item nutrition (if present)
    total = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    any_nutrition = False
    for f in detected_foods:
        c = f.get("calories")
        p = f.get("protein_g")
        cb = f.get("carbs_g")
        fat = f.get("fat_g")
        if c is not None or p is not None or cb is not None or fat is not None:
            any_nutrition = True
        total["calories"] += float(c) if c else 0
        total["protein_g"] += float(p) if p else 0
        total["carbs_g"] += float(cb) if cb else 0
        total["fat_g"] += float(fat) if fat else 0

    if any_nutrition:
        analysis["total_nutrition"] = {
            "calories": round(total["calories"], 1),
            "protein_g": round(total["protein_g"], 1),
            "carbs_g": round(total["carbs_g"], 1),
            "fat_g": round(total["fat_g"], 1),
        }

    return {
        "success": True,
        "filename": file.filename,
        "size_bytes": file_size,
        "analysis": analysis,
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "food-detection-api"}
