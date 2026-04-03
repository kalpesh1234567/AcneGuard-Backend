"""
auth.py — Production-level authentication router for AcneGuard.

Endpoints:
  POST /auth/register         → Create account (bcrypt password)
  POST /auth/login            → Email + password → JWT access token
  GET  /auth/me               → Get current user (JWT required)
  POST /auth/forgot-password  → Generate 6-digit OTP (10-min TTL)
  POST /auth/reset-password   → Verify OTP → update password
"""
import os
import random
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field

from database import get_users_collection
from email_service import send_otp_email

load_dotenv()

JWT_SECRET    = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

router = APIRouter(tags=["Authentication"])
bearer_scheme = HTTPBearer(auto_error=False)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name:             str      = Field(..., min_length=2, max_length=80)
    email:            EmailStr
    password:         str      = Field(..., min_length=8)

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user: dict

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email:        EmailStr
    otp:          str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    id:         str
    name:       str
    email:      str
    created_at: datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: str, email: str, name: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub":   user_id,
        "email": email,
        "name":  name,
        "exp":   expire,
        "iat":   datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """JWT dependency — raises 401 if token is missing or invalid."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise ValueError("No sub claim")
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token invalid or expired: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    """Register a new user. Returns a JWT on success."""
    users = get_users_collection()

    existing = await users.find_one({"email": body.email.lower()})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    now = datetime.now(timezone.utc)
    doc = {
        "name":          body.name.strip(),
        "email":         body.email.lower(),
        "password_hash": hash_password(body.password),
        "is_active":     True,
        "created_at":    now,
        "updated_at":    now,
        "reset_otp":     None,
        "otp_expiry":    None,
    }
    result = await users.insert_one(doc)
    user_id = str(result.inserted_id)

    token = create_access_token(user_id, body.email.lower(), body.name.strip())
    return TokenResponse(
        access_token=token,
        user={"id": user_id, "name": body.name.strip(), "email": body.email.lower()},
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Login with email + password. Returns a JWT on success."""
    users = get_users_collection()
    user = await users.find_one({"email": body.email.lower()})

    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated.")

    user_id = str(user["_id"])
    token = create_access_token(user_id, user["email"], user["name"])
    return TokenResponse(
        access_token=token,
        user={"id": user_id, "name": user["name"], "email": user["email"]},
    )


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's info from the JWT payload."""
    return {
        "id":    current_user["sub"],
        "email": current_user["email"],
        "name":  current_user["name"],
    }


@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest):
    """
    Generate a 6-digit OTP valid for 10 minutes.
    In production, send via email. For dev, the OTP is returned in the response.
    """
    users = get_users_collection()
    user = await users.find_one({"email": body.email.lower()})

    # Always return 200 to avoid user enumeration
    if not user:
        return {"message": "If that email is registered, an OTP has been sent."}

    otp = generate_otp()
    otp_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)

    await users.update_one(
        {"email": body.email.lower()},
        {"$set": {"reset_otp": otp, "otp_expiry": otp_expiry}},
    )

    # Send OTP via real email
    try:
        await send_otp_email(body.email.lower(), otp)
    except Exception as e:
        # Roll back OTP if email fails so user can retry
        await users.update_one(
            {"email": body.email.lower()},
            {"$set": {"reset_otp": None, "otp_expiry": None}},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send OTP email. Please try again. ({e})",
        )

    return {
        "message": "OTP sent to your email. It expires in 10 minutes.",
    }


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest):
    """Verify OTP + update password hash."""
    users = get_users_collection()
    user = await users.find_one({"email": body.email.lower()})

    if not user or not user.get("reset_otp"):
        raise HTTPException(status_code=400, detail="No active reset request for this email.")

    if user["reset_otp"] != body.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP.")

    if datetime.now(timezone.utc) > user["otp_expiry"].replace(tzinfo=timezone.utc):
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")

    new_hash = hash_password(body.new_password)
    await users.update_one(
        {"email": body.email.lower()},
        {"$set": {"password_hash": new_hash, "reset_otp": None, "otp_expiry": None}},
    )
    return {"message": "Password updated successfully. You can now log in."}
