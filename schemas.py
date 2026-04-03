from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class AcneSeverity(str, Enum):
    CLEAR = "clear"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class AcneType(str, Enum):
    WHITEHEADS = "whiteheads"
    BLACKHEADS = "blackheads"
    PAPULES = "papules"
    PUSTULES = "pustules"
    NODULES = "nodules"


class AcneLocation(str, Enum):
    FOREHEAD = "forehead"
    CHEEKS = "cheeks"
    NOSE = "nose"
    CHIN = "chin"
    JAWLINE = "jawline"


class SkinIndicator(str, Enum):
    OILY_SKIN = "oily skin"
    INFLAMED_AREAS = "inflamed areas"
    DRYNESS = "dryness"
    UNEVEN_TEXTURE = "uneven texture"


# ──────────────────────────────────────────────────────
# Localization Schemas (AI-detected acne location)
# ──────────────────────────────────────────────────────

class RegionScores(BaseModel):
    """Normalized acne pixel density (0–1) per facial region."""
    forehead: float = Field(..., description="Acne density score for forehead region")
    cheeks: float   = Field(..., description="Acne density score for cheeks region")
    nose: float     = Field(..., description="Acne density score for nose region")
    chin: float     = Field(..., description="Acne density score for chin region")
    jawline: float  = Field(..., description="Acne density score for jawline region")


class LocalizationResult(BaseModel):
    """Full output of the AI acne localization pipeline."""
    location: AcneLocation = Field(..., description="Dominant acne region detected by AI")
    region_scores: RegionScores = Field(..., description="Per-region acne density scores")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the detected location (0–1)")
    method: str = Field(..., description="Detection method used: mediapipe_facemesh | quadrant_fallback | static_fallback")
    face_detected: bool = Field(..., description="Whether a human face was detected in the image")


# ──────────────────────────────────────────────────────
# User Routine Schema
# ──────────────────────────────────────────────────────

class UserRoutine(BaseModel):
    facewash_used: Optional[str] = Field(None, description="What facewash does the user use")
    wash_frequency: Optional[str] = Field(None, description="How much time/how often they wash their face")
    products_used: Optional[str] = Field(None, description="What other products they use for their face")
    sleep_hours: Optional[str] = Field(None, description="How much time they sleep")
    extra_details: Optional[str] = Field(None, description="Any other routine parameters")


# ──────────────────────────────────────────────────────
# Core Analysis Schema
# ──────────────────────────────────────────────────────

class AcneAnalysis(BaseModel):
    severity: AcneSeverity
    acne_type: AcneType
    location: AcneLocation
    indicators: List[SkinIndicator] = Field(default_factory=list)
    localization: Optional[LocalizationResult] = Field(
        default=None,
        description="Full AI localization result with per-region scores"
    )
    user_routine: Optional[UserRoutine] = Field(
        default=None,
        description="Optional user questionnaire parameters regarding skincare routine and lifestyle"
    )


class FoodCard(BaseModel):
    title: str = Field(..., description="Title of the food card, e.g., 'Anti-Inflammatory Plan'")
    why_this_food: str = Field(..., description="Explanation of why these foods help this specific acne pattern")
    recommended_foods: List[str] = Field(..., description="3-5 specific household foods to include")
    foods_to_limit: List[str] = Field(..., description="2-4 specific foods to reduce")


class AcneExplanation(BaseModel):
    skin_score: int = Field(..., description="Overall skin health score out of 100")
    simple_summary: str = Field(..., description="2-3 lines explaining what was detected, severity, and reassurance")
    why_it_happened: str = Field(..., description="Dynamic causes based on location and type")
    diet_suggestions: str = Field(..., description="Diet tips based on acne type and inflammation")
    skincare_routine_suggestions: str = Field(..., description="Skincare routine based on skin type")
    lifestyle_adjustments: str = Field(..., description="Recommendations for sleep, stress, hygiene")
    recovery_expectation: str = Field(..., description="Timeline based on severity")
    encouragement_line: str = Field(..., description="Supportive ending tone")
    food_card: FoodCard = Field(..., description="Personalized food recommendation card based on inflammation, oiliness, and diet risks")


class DietExplanation(BaseModel):
    main_causes: List[str] = Field(..., description="Top 3 to 5 real-time personalized reasons why their specific diet and lifestyle causes acne risk.")
    recommendations: List[str] = Field(..., description="Top 4 to 6 real-time, highly personalized lifestyle and diet adjustments based directly on their inputs.")
