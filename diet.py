"""
diet.py — Diet-based acne risk prediction router.
Uses the trained Random Forest model (skin_health_backend/ml/).
Endpoint: POST /diet/predict (JWT required)
"""
import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from auth import get_current_user
from engine import AcneGuardEngine

router = APIRouter(prefix="/diet", tags=["Diet Analysis"])

# ── Load model once at import time ───────────────────────────────────────────
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH    = os.path.join(_BASE_DIR, "skin_health_backend", "ml", "model.joblib")
_ENCODERS_PATH = os.path.join(_BASE_DIR, "skin_health_backend", "ml", "encoders.joblib")

try:
    _model         = joblib.load(_MODEL_PATH)
    _artifacts     = joblib.load(_ENCODERS_PATH)
    _preprocessor  = _artifacts["preprocessor"]
    _target_enc    = _artifacts["target_encoder"]
    _feat_names    = _artifacts["feature_names"]
    print("[DietML] Model loaded successfully.")
except Exception as e:
    print(f"[DietML] WARNING: Could not load model — {e}")
    _model = None


# ── Schemas ──────────────────────────────────────────────────────────────────
class DietPredictRequest(BaseModel):
    diet_type:           str = Field(..., example="Veg")
    specific_food_item:  str = Field(..., example="Leafy Greens")
    sugar_intake:        str = Field(..., example="Medium")
    dairy_intake:        str = Field(..., example="Occasional")
    processed_food_freq: str = Field(..., example="Low")
    spicy_food_freq:     str = Field(..., example="Low")
    oily_food_level:     str = Field(..., example="Low")
    water_intake_liters: float = Field(..., example=2.5)
    sleep_hours:         int   = Field(..., example=7)
    stress_level:        int   = Field(..., ge=1, le=5, example=2)
    exercise:            str   = Field(..., example="Yes")
    skin_type:           str   = Field(..., example="Normal")


class DietPredictResponse(BaseModel):
    risk_level:       str
    confidence_score: float
    main_causes:      List[str]
    recommendations:  List[str]


# ── Helpers ───────────────────────────────────────────────────────────────────
_FOOD_CAT_MAP = {
    "Red Meat":          "High_Inflammation",
    "Processed Meat":    "High_Inflammation",
    "Starchy/Potato":    "High_Glycemic",
    "Balanced Veg":      "Balanced",
    "Leafy Greens":      "Antioxidant_Rich",
    "Legumes/Pulses":    "Antioxidant_Rich",
    "Fish":              "Omega3_Rich",
    "Nuts/Seeds":        "Omega3_Rich",
    "Paneer/Dairy Rich": "Hormonal_Trigger",
    "Eggs":              "Hormonal_Trigger",
    "Chicken":           "Balanced",
}


def _derive_category(food: str) -> str:
    return _FOOD_CAT_MAP.get(food, "Balanced")


def _build_causes(data: DietPredictRequest, food_cat: str, feat_importances: dict) -> List[str]:
    causes = []
    if data.sugar_intake == "High":         causes.append("High Sugar Intake")
    if data.processed_food_freq == "High":  causes.append("Processed Food")
    if data.dairy_intake == "Daily":        causes.append("High Dairy Intake")
    if data.oily_food_level == "High":      causes.append("Oily Food")
    if data.stress_level >= 4:              causes.append("High Stress")
    if data.sleep_hours < 6:               causes.append("Sleep Deprivation")
    if data.water_intake_liters < 1.5:     causes.append("Low Water Intake")
    if data.exercise == "No":               causes.append("Lack of Exercise")
    if food_cat == "High_Inflammation":     causes.append("Inflammatory Diet")
    if food_cat == "Hormonal_Trigger":      causes.append("Hormonal Triggers (Dairy/Eggs)")
    if food_cat == "High_Glycemic":         causes.append("High Glycemic Food")
    if data.skin_type == "Oily":            causes.append("Oily Skin Type")
    return causes[:5]  # top 5 max


def _build_recs(data: DietPredictRequest, risk: str, food_cat: str) -> List[str]:
    recs = []
    if food_cat == "High_Inflammation":
        recs.append("🥩 Red/processed meat is inflammatory. Switch to fish or chicken for lower acne risk.")
    if food_cat == "Hormonal_Trigger":
        recs.append("🥛 Dairy and eggs can trigger hormonal acne. Try reducing intake and monitor breakouts.")
    if food_cat == "High_Glycemic":
        recs.append("🥔 High-glycemic foods spike insulin. Replace with leafy greens or legumes.")
    if data.sugar_intake == "High":
        recs.append("🍬 Reduce added sugar — it raises IGF-1 and promotes sebum production.")
    if data.processed_food_freq == "High":
        recs.append("🍟 Cut down on processed food. Whole foods reduce systemic inflammation.")
    if data.spicy_food_freq == "High" and data.skin_type == "Oily":
        recs.append("🌶️ Spicy food + oily skin is a flare-up combo. Reduce spicy meals.")
    if data.water_intake_liters < 2.0:
        recs.append("💧 Drink at least 2.5L of water daily — hydration helps flush toxins through skin.")
    if data.sleep_hours < 7:
        recs.append("😴 Aim for 7–8 hours of sleep. Poor sleep raises cortisol and worsens acne.")
    if data.stress_level >= 4:
        recs.append("🧘 High stress detected. Try 10 min daily meditation or light exercise to lower cortisol.")
    if data.exercise == "No":
        recs.append("🏃 Regular exercise (3×/week) improves blood flow and reduces stress hormones.")
    if data.skin_type == "Oily":
        recs.append("🧴 Use a gentle foaming cleanser twice daily to manage oil production.")
    if risk == "Low" and not recs:
        recs.append("✅ Your diet and lifestyle look great! Keep up the healthy habits.")
    return recs


# ── Endpoint ─────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=DietPredictResponse)
async def predict_diet_risk(
    body: DietPredictRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Predict acne risk from diet + lifestyle inputs.
    Returns risk level (Low/Medium/High), confidence, causes, and recommendations.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Diet model not loaded. Run train_model.py first.")

    try:
        food_cat = _derive_category(body.specific_food_item)

        row = body.dict()
        row["food_category"] = food_cat
        df = pd.DataFrame([row])

        X = _preprocessor.transform(df)
        pred_idx  = _model.predict(X)[0]
        risk      = _target_enc.inverse_transform([pred_idx])[0]
        probs     = _model.predict_proba(X)[0]
        confidence = float(probs[pred_idx])

        engine = AcneGuardEngine()
        explanation = await engine.generate_diet_explanation(body.dict(), risk)
        causes = explanation.main_causes
        recs = explanation.recommendations

        return DietPredictResponse(
            risk_level=risk,
            confidence_score=round(confidence * 100, 1),
            main_causes=causes,
            recommendations=recs,
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
