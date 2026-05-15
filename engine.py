import os
import json
from google import genai
from google.genai import types
from typing import Dict, Any

from schemas import AcneAnalysis, AcneExplanation
from knowledge_base import LOCATION_INFO, ACNE_TYPE_INFO, SKIN_INDICATOR_INFO

# Initialize Gemini API using the new google-genai SDK
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# The correct 2026 model name for the new SDK
MODEL_NAME = 'gemini-2.5-flash'

SYSTEM_PROMPT = """
You are the real-time explanation engine of a skincare application called AcneGuard.

Important:
Acne detection is already completed by a separate local computer vision model.
You are NOT performing diagnosis.
You are NOT using any rule-based template system.
You are NOT generating static responses.
You must generate a dynamic, human-like explanation in real time based strictly on structured data provided.

Your task is to behave like a real dermatologist explaining results naturally to a patient.

The response must feel:
Personalized
Context-aware
Natural
Non-robotic
Easy to understand
Human-written

Do NOT generate template-style text.
Do NOT repeat generic skincare advice.
Do NOT sound mechanical.

You will receive structured analysis data.
You will also optionally receive user-inputted routine parameters (like their facewash, wash frequency, sleep habits, and products used).
You must interpret relationships between these fields.
Combine them intelligently when explaining.

If user_routine data is provided, YOU MUST personalize the explanation by referencing their specific habits (e.g., "Because you mentioned washing your face 3 times a day...", or "Your use of product X along with only 5 hours of sleep might be...").

Example:
If location is jawline + risk factor hormonal → explain hormonal link.
If inflamed areas + high sugar intake → explain inflammatory trigger.
If oily skin + papules → explain excess sebum blockage.

Do not explain fields separately. Blend them naturally like a real human would.

Do not explain fields separately. Blend them naturally like a real human would.

OUTPUT FORMAT REQUIREMENTS
1. You MUST generate the response matching the requested JSON schema exactly.
2. For the fields `why_it_happened`, `diet_suggestions`, `skincare_routine_suggestions`, and `lifestyle_adjustments`, you MUST format the text as a brief 1-2 sentence introductory paragraph, followed by a concise 2-3 bullet point list. Use the `- ` character for bullet points. Separate the intro sentence and the bullet points with a newline character (`\n`).

ACNE TYPE FOCUS
Your entire explanation and all advice MUST be heavily tailored and explicitly specific to the detected `acne_type` (e.g. if pustules are detected, all advice should specifically target treating pustules, not general acne). Do not ignore the acne type.

HARD RULES
Maximum 300 words total across all fields.
No medical jargon.
No bullet spam.
No generic "drink water and eat healthy".
Must feel written in real time.
Avoid repetitive sentence patterns.
Avoid fixed phrasing.
Do not invent new medical conditions.
Do not exaggerate severity.
Do not provide extreme dietary restrictions or medical treatment claims.

STYLE REQUIREMENT
Write as if a dermatologist is calmly explaining results to a young adult patient in a clinic.
Natural. Warm. Clear.

ADDITIONAL REQUIREMENT - FOOD RECOMMENDATION CARD
In addition to the skin explanation, generate a structured food recommendation card.
The food recommendations must:
- Be directly based on inflammation level, oiliness, and diet_risks
- Avoid generic advice like “eat healthy”
- Include 3–5 specific household foods to include
- Include 2–4 foods to reduce
- Explain briefly WHY these foods help this specific acne pattern
"""

class AcneGuardEngine:
    async def generate_explanation(self, analysis: AcneAnalysis) -> AcneExplanation:
        
        # 1. Gather all the context strings to feed to the LLM
        loc_info = LOCATION_INFO.get(analysis.location, {})
        type_info = ACNE_TYPE_INFO.get(analysis.acne_type, {})
        
        # Extract risk factors and indicators from our local KB
        risk_factors = loc_info.get("why", []) + type_info.get("causes", [])
        diet_risks = type_info.get("diet", {}).get("avoid", [])
        lifestyle_risks = loc_info.get("lifestyle", [])
        indicators = [i.value for i in analysis.indicators]
        
        confidence = analysis.localization.confidence if analysis.localization else 0.8
        
        # Build the payload
        payload = {
            "severity": analysis.severity.value,
            "acne_type": analysis.acne_type.value,
            "location": analysis.location.value,
            "ai_confidence_score": confidence,
            "risk_factors": list(set(risk_factors)),
            "diet_risks": list(set(diet_risks)),
            "lifestyle_risks": list(set(lifestyle_risks)),
            "skin_indicators": indicators
        }
        
        if getattr(analysis, "user_routine", None):
            payload["user_routine"] = analysis.user_routine.model_dump()
        
        prompt = f"USER DATA:\n{json.dumps(payload, indent=2)}\n\nGenerate the structured response."

        try:
            # 2. Call Gemini using async SDK to avoid blocking the event loop
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=[SYSTEM_PROMPT, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AcneExplanation,
                    temperature=0.7,
                )
            )
            
            # 3. Parse and return
            result_json = json.loads(response.text)
            return AcneExplanation(**result_json)
            
        except Exception as e:
            print(f"LLM Generation failed: {e}")
            # Fallback to the hardcoded rule-based engine if the API fails
            return self._generate_fallback(analysis)

    def _generate_fallback(self, analysis: AcneAnalysis) -> AcneExplanation:
        """Fallback rule-based generator in case Gemini API is down or key is missing."""
        severity = analysis.severity
        type_str = analysis.acne_type.value
        loc_str = analysis.location.value
        
        score = 85 if severity.value == "mild" else 70
        
        from schemas import FoodCard
        
        return AcneExplanation(
            skin_score=score,
            simple_summary=f"We detected {severity.value} {type_str} on your {loc_str}. This is a common issue and highly treatable.",
            why_it_happened=f"Acne on the {loc_str} is often related to general oil buildup and sometimes hormonal factors.",
            diet_suggestions="Try maintaining a balanced diet with plenty of water and reduced sugar.",
            skincare_routine_suggestions="Use a gentle cleanser and a non-comedogenic moisturizer daily.",
            lifestyle_adjustments="Ensure you get 7-8 hours of sleep and avoid touching your face.",
            recovery_expectation="You should see improvements in a few weeks with consistent care.",
            encouragement_line="You're on the right track, keep it up!",
            food_card=FoodCard(
                title="Basic Skin Support Plan",
                why_this_food="These foods generally help stabilize oil production.",
                recommended_foods=["Water", "Green leafy vegetables", "Nuts"],
                foods_to_limit=["High-sugar snacks", "Processed foods"]
            )
        )

    async def generate_diet_explanation(self, diet_data: dict, risk_level: str):
        from schemas import DietExplanation
        
        DIET_SYSTEM_PROMPT = """
        You are the real-time explanation engine for the diet risk module of AcneGuard.
        You take a user's lifestyle and diet inputs, along with their AI-predicted acne risk level, and output personalized, human-like feedback.
        
        Important:
        - Do NOT use generic template responses (e.g. 'Eat healthy', 'Drink more water').
        - Personalize everything. Reference exactly what they inputted (e.g. 'Since you sleep only 5 hours and eat high sugar...', 'Your daily dairy intake combined with high stress...').
        - Maintain an empathetic, professional tone like a dermatologist or nutritionist.
        """
        
        prompt = f"USER DIET DATA:\n{json.dumps(diet_data, indent=2)}\n\nPREDICTED RISK: {risk_level}\n\nProvide 3-5 specific 'main_causes' and 4-6 specific 'recommendations' strictly following the JSON schema."

        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=[DIET_SYSTEM_PROMPT, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DietExplanation,
                    temperature=0.7,
                )
            )
            result_json = json.loads(response.text)
            return DietExplanation(**result_json)
        except Exception as e:
            print(f"Diet LLM failed: {e}")
            return DietExplanation(
                main_causes=["Your overall dietary balance and lifestyle factors may be contributing to the skin's state."],
                recommendations=["Focus on a balanced diet.", "Ensure adequate hydration and sleep."]
            )