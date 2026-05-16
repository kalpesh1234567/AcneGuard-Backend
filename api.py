"""
AcneGuard AI — FastAPI Server v3.0
Endpoints:
  POST /analyze           → Full analysis (CNN + localization + recommendations) [JWT required]
  POST /localize          → Standalone acne localization
  GET  /                  → Health check
  /auth/*                 → Authentication (register, login, me, forgot-password, reset-password)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import json
import time
import logging
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from schemas import AcneAnalysis, AcneExplanation, LocalizationResult
from vision import VisionService
from engine import AcneGuardEngine
from localization import AcneLocalizer, LocalizationResult as RawLocalizationResult
from auth import router as auth_router, get_current_user
from database import create_indexes, get_assessments_collection
from assessments import router as assessments_router
from diet import router as diet_router

app = FastAPI(
    title="AcneGuard AI API",
    description=(
        "Production-level acne analysis: CNN severity + "
        "MediaPipe Face Mesh localization + medical recommendations. "
        "Authentication via JWT / MongoDB Atlas."
    ),
    version="3.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://acne-guard-frontend-p8x5.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Logger ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("acneguard")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start   = time.perf_counter()
    client  = request.client.host if request.client else "unknown"
    method  = request.method
    path    = request.url.path
    qs      = f"?{request.url.query}" if request.url.query else ""

    logger.info(f"→  {method:7s} {path}{qs}  [{client}]")
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"✗  {method:7s} {path}  ERROR: {exc}")
        raise
    elapsed = (time.perf_counter() - start) * 1000
    status  = response.status_code
    icon    = "✓" if status < 400 else "✗"
    logger.info(f"{icon}  {method:7s} {path}  {status}  {elapsed:.1f}ms")
    return response

# Mount routers
app.include_router(auth_router, prefix="/auth")
app.include_router(assessments_router, prefix="/assessments")
app.include_router(diet_router)

# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    await create_indexes()

# ── Initialize Services ──────────────────────────────────────────────────────
vision_service = VisionService()
engine         = AcneGuardEngine()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Response Models ──────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    analysis:      AcneAnalysis
    explanation:   AcneExplanation
    assessment_id: Optional[str] = None   # auto-saved DB record id


class LocalizeResponse(BaseModel):
    location:      str
    region_scores: dict
    confidence:    float
    method:        str
    face_detected: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status":  "ok",
        "service": "AcneGuard AI API v3.0",
        "endpoints": {
            "analyze":          "POST /analyze  — Full AI analysis (JWT required)",
            "localize":         "POST /localize — Standalone AI localization",
            "auth_register":    "POST /auth/register",
            "auth_login":       "POST /auth/login",
            "auth_me":          "GET  /auth/me",
            "auth_forgot":      "POST /auth/forgot-password",
            "auth_reset":       "POST /auth/reset-password",
        },
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_acne(
    file: UploadFile = File(...),
    user_routine: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),   # ← JWT required
):
    """
    Upload a face image to get:
    - **severity**: CNN-detected acne severity (clear/mild/moderate/severe)
    - **location**: AI-detected facial region (MediaPipe Face Mesh)
    - **localization**: Full region scores + confidence + detection method
    - **explanation**: Medical causes, diet, lifestyle, prevention tips

    Result is automatically saved to the user's assessment history.
    Requires: Authorization: Bearer <token>
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run CPU-bound vision inference in a thread pool (avoids blocking the event loop)
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(None, vision_service.analyze_image, file_path)

        if user_routine:
            try:
                routine_data = json.loads(user_routine)
                from schemas import UserRoutine
                analysis.user_routine = UserRoutine(**routine_data)
            except Exception as e:
                print(f"[API] Failed to parse user_routine: {e}")

        if analysis.localization and analysis.localization.method == "multi_face_rejected":
            raise HTTPException(
                status_code=400, 
                detail="Multiple faces detected. Please upload a photo with only one face clearly visible."
            )

        # Only reject if MediaPipe ran and explicitly found no face (not a fallback)
        if (
            analysis.localization
            and not analysis.localization.face_detected
            and analysis.localization.method == "mediapipe_facemesh"
        ):
            raise HTTPException(
                status_code=400, 
                detail="No face detected. Please upload a correct image containing a clearly visible face."
            )

        explanation = await engine.generate_explanation(analysis)

        # ── Auto-save to MongoDB ──────────────────────────────────────────────
        assessment_id = None
        try:
            col = get_assessments_collection()
            region_scores_dict = None
            if analysis.localization and analysis.localization.region_scores:
                region_scores_dict = analysis.localization.region_scores.model_dump()

            doc = {
                "user_id":       current_user["sub"],
                "user_email":    current_user["email"],
                "severity":      analysis.severity.value,
                "acne_type":     analysis.acne_type.value,
                "location":      analysis.location.value,
                "confidence":    analysis.localization.confidence if analysis.localization else None,
                "indicators":    [i.value for i in analysis.indicators],
                "region_scores": region_scores_dict,
                "user_routine":  analysis.user_routine.model_dump() if getattr(analysis, "user_routine", None) else None,
                "explanation":   {
                    "skin_score":                   explanation.skin_score,
                    "simple_summary":               explanation.simple_summary,
                    "why_it_happened":              explanation.why_it_happened,
                    "diet_suggestions":             explanation.diet_suggestions,
                    "skincare_routine_suggestions": explanation.skincare_routine_suggestions,
                    "lifestyle_adjustments":        explanation.lifestyle_adjustments,
                    "recovery_expectation":         explanation.recovery_expectation,
                    "encouragement_line":           explanation.encouragement_line,
                    "food_card":                    explanation.food_card.model_dump(),
                },
                "feedback":      None,
                "created_at":    datetime.now(timezone.utc),
            }
            result_db = await col.insert_one(doc)
            assessment_id = str(result_db.inserted_id)
        except Exception as db_err:
            # Do not fail the whole request if DB save fails
            print(f"[API] Auto-save failed: {db_err}")

        return AnalysisResponse(
            analysis=analysis,
            explanation=explanation,
            assessment_id=assessment_id,
        )

    except HTTPException:
        raise
    except ValueError as ve:
        # Multi-face rejection or other validation errors from VisionService
        if "MULTI_FACE" in str(ve):
            raise HTTPException(
                status_code=400,
                detail="Multiple faces detected. Please upload a photo with only one face clearly visible."
            )
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/localize", response_model=LocalizeResponse)
async def localize_acne(file: UploadFile = File(...)):
    """Standalone AI acne localization (no auth required — for testing)."""
    file_path = os.path.join(UPLOAD_DIR, f"loc_{file.filename}")
    localizer = AcneLocalizer()
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = localizer.detect(file_path)
        return LocalizeResponse(
            location=result.location,
            region_scores=result.region_scores,
            confidence=round(result.confidence, 4),
            method=result.method,
            face_detected=result.face_detected,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Localization failed: {str(e)}")
    finally:
        localizer.close()
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
