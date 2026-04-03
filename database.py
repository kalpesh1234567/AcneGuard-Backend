"""
database.py — MongoDB Atlas connection via Motor (async driver).
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "acneguard")

# Singleton client — created once at import time
_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client


def get_db():
    return get_client()[DB_NAME]


def get_users_collection():
    return get_db()["users"]


def get_assessments_collection():
    return get_db()["assessments"]


async def create_indexes():
    """Call this once on app startup to set up required indexes."""
    users = get_users_collection()
    # Unique index on email
    await users.create_index("email", unique=True)
    # TTL index on OTP — auto-expire reset tokens after 10 minutes
    await users.create_index("otp_expiry", expireAfterSeconds=0)
