import sys
sys.path.append(r"c:\Users\Dell\.gemini\antigravity\scratch\acne_guard_ai\backend")
from fastapi.testclient import TestClient
from api import app
from auth import get_current_user

client = TestClient(app)

app.dependency_overrides[get_current_user] = lambda: {"sub": "test_user", "email": "test@test.com"}

response = client.post("/diet/predict", json={
    "diet_type": "Veg",
    "specific_food_item": "Leafy Greens",
    "sugar_intake": "Low",
    "dairy_intake": "None",
    "processed_food_freq": "Low",
    "spicy_food_freq": "Low",
    "oily_food_level": "Low",
    "water_intake_liters": 3.0,
    "sleep_hours": 8,
    "stress_level": 1,
    "exercise": "Yes",
    "skin_type": "Normal"
})

print("Status:", response.status_code)
print("Response:", response.json())
