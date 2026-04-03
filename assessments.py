"""
assessments.py — Assessment history CRUD for AcneGuard.

Endpoints:
  POST   /assessments                  → Save a new scan result (JWT required)
  GET    /assessments                  → Get all scans for current user (JWT required)
  DELETE /assessments/{id}             → Delete a specific scan (JWT required)
  GET    /assessments/export           → Export all scans as feedback JSON (JWT required)
  POST   /assessments/{id}/feedback    → Save user correction/feedback for retraining
"""
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any

from auth import get_current_user
from database import get_assessments_collection

router = APIRouter(tags=["Assessments"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SaveAssessmentRequest(BaseModel):
    severity:   str
    acne_type:  str
    location:   str
    confidence: Optional[float] = None
    indicators: Optional[List[str]] = []
    region_scores: Optional[Any] = None
    explanation: Optional[Any] = None
    image_url:  Optional[str] = None   # future: store image URL

class AssessmentOut(BaseModel):
    id:         str
    severity:   str
    acne_type:  str
    location:   str
    confidence: Optional[float]
    indicators: List[str]
    region_scores: Optional[Any]
    explanation: Optional[Any]
    created_at: datetime

class FeedbackRequest(BaseModel):
    user_agrees:        bool
    corrected_severity: Optional[str] = None   # e.g. "mild", "moderate"
    comment:            Optional[str] = None


def _fmt(doc) -> dict:
    """Convert MongoDB doc to serializable dict."""
    doc["id"] = str(doc.pop("_id"))
    return doc


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", status_code=201)
async def save_assessment(
    body: SaveAssessmentRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save an AI scan result linked to the current user."""
    col = get_assessments_collection()
    doc = {
        "user_id":      current_user["sub"],
        "user_email":   current_user["email"],
        "severity":     body.severity,
        "acne_type":    body.acne_type,
        "location":     body.location,
        "confidence":   body.confidence,
        "indicators":   body.indicators or [],
        "region_scores": body.region_scores,
        "explanation":  body.explanation,
        "feedback":     None,
        "created_at":   datetime.now(timezone.utc),
    }
    result = await col.insert_one(doc)
    return {"id": str(result.inserted_id), "message": "Assessment saved."}


@router.get("")
async def get_assessments(current_user: dict = Depends(get_current_user)):
    """Return all assessments for the logged-in user, newest first."""
    col = get_assessments_collection()
    cursor = col.find(
        {"user_id": current_user["sub"]},
        sort=[("created_at", -1)],
    )
    results = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        results.append(doc)
    return results


@router.delete("/{assessment_id}")
async def delete_assessment(
    assessment_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete an assessment — only the owner can delete."""
    col = get_assessments_collection()
    try:
        oid = ObjectId(assessment_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid assessment ID.")

    result = await col.delete_one({"_id": oid, "user_id": current_user["sub"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return {"message": "Deleted successfully."}


@router.get("/export")
async def export_feedback(current_user: dict = Depends(get_current_user)):
    """
    Export all assessments as structured feedback JSON for model retraining.
    Each record includes severity, location, acne_type, region_scores for use
    as labelled training data.
    """
    col = get_assessments_collection()
    cursor = col.find({"user_id": current_user["sub"]}, sort=[("created_at", -1)])
    feedback = []
    async for doc in cursor:
        feedback.append({
            "id":                str(doc["_id"]),
            "severity":          doc.get("severity"),
            "acne_type":         doc.get("acne_type"),
            "location":          doc.get("location"),
            "region_scores":     doc.get("region_scores"),
            "confidence":        doc.get("confidence"),
            "user_feedback":     doc.get("feedback"),
            "created_at":        doc.get("created_at").isoformat() if doc.get("created_at") else None,
        })
    return {"count": len(feedback), "feedback": feedback}


@router.post("/{assessment_id}/feedback")
async def save_feedback(
    assessment_id: str,
    body: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Save user feedback for a specific assessment.
    Used for model retraining — records whether the AI was correct,
    and optionally stores a user-corrected severity label.
    """
    col = get_assessments_collection()
    try:
        oid = ObjectId(assessment_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid assessment ID.")

    feedback_doc = {
        "user_agrees":        body.user_agrees,
        "corrected_severity": body.corrected_severity,
        "comment":            body.comment,
        "submitted_at":       datetime.now(timezone.utc).isoformat(),
    }

    result = await col.update_one(
        {"_id": oid, "user_id": current_user["sub"]},
        {"$set": {"feedback": feedback_doc}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return {"message": "Feedback saved. Thank you for helping us improve!"}
