from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# CONFIG
MODEL_PATH = "skin_health_backend/ml/model.joblib"
ENCODERS_PATH = "skin_health_backend/ml/encoders.joblib"

app = FastAPI(title="Skin Health Prediction API")

# Load Model & Artifacts
print("Loading model and artifacts...")
try:
    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ENCODERS_PATH)
    preprocessor = artifacts['preprocessor']
    target_encoder = artifacts['target_encoder']
    feature_names_list = artifacts['feature_names']
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None
    target_encoder = None
    feature_names_list = []

class SkinHealthInput(BaseModel):
    diet_type: str
    specific_food_item: str
    sugar_intake: str
    dairy_intake: str
    processed_food_freq: str
    spicy_food_freq: str
    oily_food_level: str
    water_intake_liters: float
    sleep_hours: int
    stress_level: int
    exercise: str
    skin_type: str

class PredictionResponse(BaseModel):
    risk_level: str
    confidence_score: float
    main_causes: list[str]
    recommendations: list[str]

def derive_food_category(specific_food: str) -> str:
    # Logic matches generate_synthetic_data.py
    if specific_food in ['Red Meat', 'Processed Meat']:
        return 'High_Inflammation'
    elif specific_food in ['Starchy/Potato', 'Balanced Veg']:
        return 'High_Glycemic'
    elif specific_food in ['Leafy Greens', 'Legumes/Pulses']:
        return 'Antioxidant_Rich'
    elif specific_food in ['Fish', 'Nuts/Seeds']:
        return 'Omega3_Rich'
    elif specific_food in ['Paneer/Dairy Rich', 'Eggs']:
        return 'Hormonal_Trigger'
    else:
        return 'Balanced'

def get_recommendations(data: SkinHealthInput, risk_level: str, food_category: str):
    recs = []
    
    # Specific Food Recs
    if food_category == 'High_Inflammation':
        recs.append("Red meat/processed meat is highly inflammatory. Consider switching to Fish or Chicken.")
    if food_category == 'Hormonal_Trigger':
        recs.append("Dairy/Eggs can be hormonal triggers. Monitor breakouts after consumption.")
    if food_category == 'High_Glycemic':
        recs.append("High glycemic foods spike insulin. Eat more leafy greens instead.")
        
    # General Diet
    if data.sugar_intake == 'High':
        recs.append("Reduce sugar intake to lower inflammation.")
    if data.processed_food_freq == 'High':
        recs.append("Avoid highly processed foods; switch to whole foods.")
    if data.spicy_food_freq == 'High' and data.skin_type == 'Oily':
        recs.append("Reduce spicy food intake to prevent flare-ups.")
        
    if data.water_intake_liters < 2.0:
        recs.append("Increase water intake to at least 2.5 liters to hydrate skin.")
        
    # Lifestyle Recs
    if data.sleep_hours < 7:
        recs.append("Improve sleep duration (aim for 7-8 hours).")
    if data.stress_level >= 4:
        recs.append("High stress detected. Try meditation or yoga.")
        
    # Skin Recs
    if data.skin_type == 'Oily' or data.oily_food_level == 'High':
        recs.append("Use a gentle foaming cleanser.")
        
    if risk_level == 'Low' and not recs:
        recs.append("Your lifestyle looks great! Keep it up.")
        
    return recs

@app.post("/predict", response_model=PredictionResponse)
def predict_acne_risk(input_data: SkinHealthInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Feature Engineering
        food_cat = derive_food_category(input_data.specific_food_item)
        
        # 2. Create DataFrame for Preprocessor
        input_dict = input_data.dict()
        input_dict['food_category'] = food_cat 
        # Don't need specific_food_item for prediction anymore, 
        # but pandas will just ignore extra cols if not selected or we can drop it.
        # Actually preprocessor selects columns by name, so extra cols are fine IF we pass DF.
        
        df_input = pd.DataFrame([input_dict])
        
        # 3. Transform Data
        X_input = preprocessor.transform(df_input)
        
        # 4. Predict
        prediction_idx = model.predict(X_input)[0]
        risk_label = target_encoder.inverse_transform([prediction_idx])[0]
        
        # 5. Probability
        probs = model.predict_proba(X_input)[0]
        confidence = float(probs[prediction_idx])
        
        # 6. Explainability (Feature Importance)
        # We find top contributing features based on global importance * input value
        # This is a heuristic. A better way uses SHAP, but for prototype:
        # We just list the inputs that correspond to high-importance features IF they are "bad" values.
        
        # Get feature importances
        importances = model.feature_importances_
        # Map importance to feature names
        # feature_names_list matches X_input columns
        
        # Create a list of (feature, importance)
        feature_imp_dict = dict(zip(feature_names_list, importances))
        
        # Sort by importance
        sorted_features = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Identify "Causes" -> top important features where user has "Bad" input
        causes = []
        bad_values = ['High', 'Daily', 'High_Inflammation', 'Hormonal_Trigger', 'High_Glycemic', 'None'] 
        # Note: 'None' for dairy is BAD? No, 'None' dairy is usually good. 
        # Wait, 'None' for exercise is BAD. Context matters.
        
        for feat_name, imp in sorted_features[:5]: # Check top 5 features
            # feat_name might be "sugar_intake" (ordinal) or "food_category_High_Inflammation" (onehot)
            
            # Case 1: Ordinal Features
            if feat_name in input_dict:
                val = input_dict[feat_name]
                if feat_name == 'sugar_intake' and val == 'High': causes.append("High Sugar Intake")
                if feat_name == 'stress_level' and val >= 4: causes.append("High Stress Level")
                if feat_name == 'sleep_hours' and val < 6: causes.append("Low Sleep Duration")
                if feat_name == 'water_intake_liters' and val < 1.5: causes.append("Low Water Intake")
                if feat_name == 'exercise' and val == 'No': causes.append("Lack of Exercise")
                if feat_name == 'processed_food_freq' and val == 'High': causes.append("Processsed Food")
            
            # Case 2: OneHot Features
            if 'food_category' in feat_name:
                # e.g. "food_category_High_Inflammation"
                # If this feature is important AND matches our category
                if food_cat in feat_name:
                     if food_cat == 'High_Inflammation': causes.append("Inflammatory Diet")
                     if food_cat == 'Hormonal_Trigger': causes.append("Hormonal Triggers (Dairy/Eggs)")

        if not causes and risk_label == 'High':
            causes.append("Combination of lifestyle factors")

        # 7. Recommendations
        recs = get_recommendations(input_data, risk_label, food_cat)
        
        return {
            "risk_level": risk_label,
            "confidence_score": round(confidence * 100, 2),
            "main_causes": list(set(causes)), # Remove duplicates
            "recommendations": recs
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Skin Health API is running"}
