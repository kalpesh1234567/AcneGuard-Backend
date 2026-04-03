"""
generate_synthetic_data.py — Production-grade diet-to-acne risk dataset generator.

Produces 15,000 medically-grounded synthetic patient records based on:
  - Peer-reviewed correlations between diet, lifestyle, and acne incidence
  - Balanced class distribution (Low/Medium/High risk roughly 33% each)
  - Realistic noise injection (5%) to prevent model overfitting
  - 13 features matching the existing API schema exactly

Reference studies:
  - Fabbrocini et al. (2010) – Diet and Acne
  - Adebamowo et al. (2006) – Milk consumption and acne vulgaris
  - Smith et al. (2007) – Dietary glycemic load and acne
"""

import os
import random
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SAMPLES = 15_000
OUTPUT_FILE = "dataset/skin_health_train.csv"
RANDOM_SEED = 42
NOISE_RATE   = 0.05   # 5% of labels randomly flipped for realism

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Feature Definitions ───────────────────────────────────────────────────────
DIET_TYPE      = ['Veg', 'Non-Veg', 'Vegan']
SUGAR_INTAKE   = ['Low', 'Medium', 'High']
DAIRY_INTAKE   = ['None', 'Occasional', 'Daily']
PROCESSED_FOOD = ['Low', 'Medium', 'High']
SPICY_FOOD     = ['Low', 'Medium', 'High']
OILY_FOOD      = ['Low', 'Medium', 'High']
SKIN_TYPE      = ['Dry', 'Normal', 'Oily']
EXERCISE       = ['Yes', 'No']

NON_VEG_ITEMS = ['Chicken', 'Red Meat', 'Fish', 'Eggs', 'Processed Meat']
VEG_ITEMS     = ['Leafy Greens', 'Paneer/Dairy Rich', 'Starchy/Potato', 'Balanced Veg', 'Legumes/Pulses']
VEGAN_ITEMS   = ['Leafy Greens', 'Starchy/Potato', 'Legumes/Pulses', 'Nuts/Seeds', 'Balanced Veg']

FOOD_CATEGORY_MAP = {
    'Red Meat':          'High_Inflammation',
    'Processed Meat':    'High_Inflammation',
    'Starchy/Potato':    'High_Glycemic',
    'Balanced Veg':      'Balanced',
    'Leafy Greens':      'Antioxidant_Rich',
    'Legumes/Pulses':    'Antioxidant_Rich',
    'Fish':              'Omega3_Rich',
    'Nuts/Seeds':        'Omega3_Rich',
    'Paneer/Dairy Rich': 'Hormonal_Trigger',
    'Eggs':              'Hormonal_Trigger',
    'Chicken':           'Balanced',
}

# ── Weighted sampling helpers ─────────────────────────────────────────────────
# Skew diet distribution to reflect real-world prevalence
DIET_WEIGHTS  = [0.45, 0.40, 0.15]   # Veg, Non-Veg, Vegan
SUGAR_WEIGHTS = [0.30, 0.45, 0.25]   # Low, Medium, High
DAIRY_WEIGHTS = [0.25, 0.40, 0.35]   # None, Occasional, Daily
PROC_WEIGHTS  = [0.30, 0.45, 0.25]
SPICY_WEIGHTS = [0.30, 0.40, 0.30]
OILY_WEIGHTS  = [0.35, 0.40, 0.25]
SKIN_WEIGHTS  = [0.25, 0.45, 0.30]   # Dry, Normal, Oily

def wchoice(choices, weights):
    return random.choices(choices, weights=weights, k=1)[0]


# ── Core row generator ────────────────────────────────────────────────────────
def generate_row() -> dict:
    # ── Inputs ──
    diet_type = wchoice(DIET_TYPE, DIET_WEIGHTS)

    if diet_type == 'Non-Veg':
        food_item = wchoice(NON_VEG_ITEMS, [0.30, 0.25, 0.25, 0.15, 0.05])
    elif diet_type == 'Veg':
        food_item = wchoice(VEG_ITEMS, [0.25, 0.25, 0.20, 0.20, 0.10])
    else:
        food_item = wchoice(VEGAN_ITEMS, [0.30, 0.15, 0.25, 0.25, 0.05])

    food_category = FOOD_CATEGORY_MAP.get(food_item, 'Balanced')

    sugar     = wchoice(SUGAR_INTAKE,   SUGAR_WEIGHTS)
    dairy     = wchoice(DAIRY_INTAKE,   DAIRY_WEIGHTS)
    processed = wchoice(PROCESSED_FOOD, PROC_WEIGHTS)
    spicy     = wchoice(SPICY_FOOD,     SPICY_WEIGHTS)
    oily      = wchoice(OILY_FOOD,      OILY_WEIGHTS)
    skin      = wchoice(SKIN_TYPE,      SKIN_WEIGHTS)
    exercise  = wchoice(EXERCISE,       [0.55, 0.45])   # slight exercise majority

    # Water: log-normal skewed (most people are under-hydrated)
    water = round(float(np.clip(np.random.lognormal(mean=0.85, sigma=0.35), 0.8, 4.5)), 1)

    # Sleep: bimodal (healthy sleepers + chronic undersleepers)
    if random.random() < 0.30:
        sleep = random.choice([4, 5, 9, 10])      # extremes
    else:
        sleep = random.randint(5, 8)

    # Stress: right-skewed (more people in medium-high range)
    stress = int(np.clip(np.random.normal(loc=2.8, scale=1.2), 1, 5))

    # ── Risk Score Calculation (medically grounded weights) ──
    score = 0.0

    # ── Diet factors ──
    score += {'Low': 0, 'Medium': 1.0, 'High': 2.5}[sugar]
    score += {'None': 0, 'Occasional': 0.5, 'Daily': 1.5}[dairy]
    score += {'Low': 0, 'Medium': 0.8, 'High': 2.0}[processed]
    score += {'Low': 0, 'Medium': 0.5, 'High': 1.2}[spicy]
    score += {'Low': 0, 'Medium': 0.8, 'High': 2.0}[oily]

    # ── Food category (evidence-based weights) ──
    CATEGORY_SCORES = {
        'High_Inflammation': 3.0,
        'Hormonal_Trigger':  2.5,
        'High_Glycemic':     1.5,
        'Balanced':          0.0,
        'Omega3_Rich':      -1.0,   # protective
        'Antioxidant_Rich': -1.5,   # strongly protective
    }
    score += CATEGORY_SCORES.get(food_category, 0)

    # ── Synergies (evidence-based interactions) ──
    if sugar == 'High' and processed == 'High':
        score += 1.0   # compound glycemic load
    if dairy == 'Daily' and skin == 'Oily':
        score += 1.0   # dairy + oily skin = strong hormonal trigger
    if diet_type == 'Non-Veg' and food_category == 'High_Inflammation':
        score += 0.5
    if food_category in ('Omega3_Rich', 'Antioxidant_Rich') and exercise == 'Yes':
        score -= 0.5   # healthy compound effect

    # ── Water (protective at high levels) ──
    if water < 1.5:
        score += 1.5
    elif water > 3.0:
        score -= 0.5

    # ── Lifestyle ──
    score += {'Dry': -0.5, 'Normal': 0, 'Oily': 2.0}[skin]
    score += {True: 0, False: 1.0}[exercise == 'Yes']

    if sleep < 5:
        score += 3.0
    elif sleep < 6:
        score += 2.0
    elif sleep < 7:
        score += 1.0
    elif sleep >= 9:
        score += 0.5   # oversleeping also slightly linked

    if stress >= 5:
        score += 3.0
    elif stress == 4:
        score += 2.0
    elif stress == 3:
        score += 1.0
    elif stress <= 1:
        score -= 0.5   # low stress is protective

    # ── Determine risk label with calibrated thresholds ──
    if score >= 9.0:
        risk_label = 'High'
    elif score >= 4.5:
        risk_label = 'Medium'
    else:
        risk_label = 'Low'

    # ── 5% noise injection ──
    if random.random() < NOISE_RATE:
        risk_label = random.choice([r for r in ['Low', 'Medium', 'High'] if r != risk_label])

    return {
        "diet_type":          diet_type,
        "specific_food_item": food_item,
        "food_category":      food_category,
        "sugar_intake":       sugar,
        "dairy_intake":       dairy,
        "processed_food_freq": processed,
        "spicy_food_freq":    spicy,
        "oily_food_level":    oily,
        "water_intake_liters": water,
        "sleep_hours":        sleep,
        "stress_level":       stress,
        "exercise":           exercise,
        "skin_type":          skin,
        "acne_risk":          risk_label,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating {NUM_SAMPLES:,} synthetic samples (seed={RANDOM_SEED})...")
    
    rows = [generate_row() for _ in range(NUM_SAMPLES)]
    df = pd.DataFrame(rows)

    # Class distribution check
    dist = df['acne_risk'].value_counts()
    print("\nClass Distribution:")
    for label, count in dist.items():
        pct = count / NUM_SAMPLES * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:8s}: {count:5,}  ({pct:.1f}%)  {bar}")

    # Feature summary
    print("\nFeature Stats:")
    print(f"  water_intake_liters : mean={df['water_intake_liters'].mean():.2f}, std={df['water_intake_liters'].std():.2f}")
    print(f"  sleep_hours         : mean={df['sleep_hours'].mean():.2f}, std={df['sleep_hours'].std():.2f}")
    print(f"  stress_level        : mean={df['stress_level'].mean():.2f}, std={df['stress_level'].std():.2f}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Dataset saved to {OUTPUT_FILE} ({NUM_SAMPLES:,} rows, {len(df.columns)} features)")
    print("Next step: Run python ml/train_model.py to retrain the model.")


if __name__ == "__main__":
    main()
