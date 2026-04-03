from typing import Dict, List, Any
from schemas import AcneType, AcneLocation, SkinIndicator

# --- Production Knowledge Base for AcneGuard AI ---
# Disclaimer: This information is for educational purposes only and not a substitute for professional medical advice.

ACNE_TYPE_INFO = {
    AcneType.WHITEHEADS: {
        "description": "Small, flesh-colored bumps caused by clogged pores (closed comedones).",
        "causes": [
            "Excess sebum (oil) production", 
            "Accumulation of dead skin cells", 
            "Hormonal fluctuations",
            "Comedogenic cosmetic products"
        ],
        "diet": {
            "avoid": ["High-glycemic foods (sugary snacks, white bread)", "Dairy products if sensitive"],
            "recommend": ["Vitamin A rich foods (sweet potatoes, carrots)", "Zinc-rich foods (pumpkin seeds, lentils)"]
        },
        "skincare": {
            "key_ingredients": ["Salicylic Acid (BHA)", "Retinoids (Adapalene)", "Glycolic Acid (AHA)"],
            "avoid": ["Thick, pore-clogging creams", "Coconut oil on face"]
        },
        "severity_note": "Usually responds well to over-the-counter topical treatments."
    },
    AcneType.BLACKHEADS: {
        "description": "Open comedones where trapped oil oxidizes and turns dark upon exposure to air.",
        "causes": [
            "Excess oil production in the T-zone", 
            "Improper cleansing/sleeping with makeup", 
            "Heavy, occlusive moisturizers"
        ],
        "diet": {
            "avoid": ["Greasy or deep-fried foods", "Excessive sugar"],
            "recommend": ["Antioxidant-rich berries", "Green tea (EGCG reduces sebum)", "Omega-3s"]
        },
        "skincare": {
            "key_ingredients": ["Salicylic Acid (BHA)", "Niacinamide (oil control)", "Clay masks (Kaolin/Bentonite)"],
            "avoid": ["Physical scrubs that tear skin", "Alcohol-based toners"]
        },
        "severity_note": "Requires consistent exfoliation. Avoid squeezing to prevent scarring."
    },
    AcneType.PAPULES: {
        "description": "Small, tender pink bumps indicating inflammation and bacteria trapped under skin.",
        "causes": [
            "C. acnes bacterial proliferation", 
            "Rupture of pore walls from squeezing", 
            "Stress-induced inflammation"
        ],
        "diet": {
            "avoid": ["Processed sugar and refined carbs", "Spicy foods (can trigger flushing)"],
            "recommend": ["Anti-inflammatory foods (fatty fish, turmeric)", "Probiotics (yogurt, kimchi)"]
        },
        "skincare": {
            "key_ingredients": ["Benzoyl Peroxide (kills bacteria)", "Azelaic Acid", "Tea Tree Oil"],
            "avoid": ["Harsh scrubs", "Fragrance in skincare"]
        },
        "severity_note": "Do not pick. This is an inflammatory stage that can lead to scarring."
    },
    AcneType.PUSTULES: {
        "description": "Pimples containing pus, red at the base (classic 'zits').",
        "causes": [
            "Active bacterial infection", 
            "Systemic inflammation", 
            "Reaction to food sensitivities"
        ],
        "diet": {
            "avoid": ["Dairy (especially skim milk)", "High sugar intake", "Alcohol"],
            "recommend": ["Omega-3 fatty acids (walnuts, salmon)", "Leafy greens", "Zinc supplements"]
        },
        "skincare": {
            "key_ingredients": ["Benzoyl Peroxide", "Sulfur (dries out pus)", "Hydrocolloid patches"],
            "avoid": ["Popping (spreads bacteria)", "Heavy oils"]
        },
        "severity_note": "Highly infectious to surrounding pores if popped incorrectly."
    },
    AcneType.NODULES: {
        "description": "Large, painful, solid lumps deeply embedded in the skin.",
        "causes": [
            "Deep infection in dermis", 
            "Genetic predisposition", 
            "Androgen imbalance (hormonal)"
        ],
        "diet": {
            "avoid": ["High-glycemic diet", "Whey protein supplements", "Dairy"],
            "recommend": ["Low-inflammatory diet (Mediterranean style)", "Spearmint tea (anti-androgenic)", "Water"]
        },
        "skincare": {
            "key_ingredients": ["Professional care recommended", "Topical Retinoids", "Ice (to reduce pain)"],
            "avoid": ["Everything harsh - treat gently"]
        },
        "severity_note": "⚠️ High risk of scarring. Often requires dermatologist intervention (oral medication or cortisone shots)."
    }
}

LOCATION_INFO = {
    AcneLocation.FOREHEAD: {
        "why": [
            "Dandruff or fungal issues on scalp falling on forehead",
            "Irregular sleep schedule causing internal stress",
            "Digestive issues (poor gut health)", 
            "Hair products (Pomade acne) clogging pores"
        ],
        "lifestyle": [
            "Use an anti-dandruff shampoo (Ketoconazole) if scalp is distinct",
            "Establish a consistent 7-8 hour sleep cycle",
            "Wash face immediately after sweating/wearing hats", 
            "Reduce fermented foods for a while"
        ]
    },
    AcneLocation.CHEEKS: {
        "why": [
            "Dirty pillowcases (bacterial transfer)", 
            "Respiratory system stress (traditional mapping)", 
            "Mobile phone bacteria against skin", 
            "High sugar intake impacting collagen"
        ],
        "lifestyle": [
            "Change pillowcases every 2 days (silk/satin is best)", 
            "Wipe phone screen with alcohol daily", 
            "Avoid touching face with hands", 
            "Check for dental problems (gums often map to lower cheeks)"
        ]
    },
    AcneLocation.NOSE: {
        "why": [
            "Blood pressure fluctuations", 
            "Vitamin B deficiency", 
            "Excess oil production (T-Zone active)", 
            "Constipation or bloating"
        ],
        "lifestyle": [
            "Check Vitamin B levels", 
            "Reduce sodium intake and spicy foods", 
            "Eat more fiber to aid digestion", 
            "Use Salicylic Acid (BHA) to unclog nose pores"
        ]
    },
    AcneLocation.CHIN: {
        "why": [
            "Hormonal fluctuations (menstrual cycle or androgens)", 
            "Kidney imbalances (dehydration)", 
            "Consuming dairy products"
        ],
        "lifestyle": [
            "Track cycle to predict flare-ups", 
            "Drink 3L of water daily", 
            "Consider reducing dairy intake", 
            "Drink Spearmint tea (natural anti-androgen)"
        ]
    },
    AcneLocation.JAWLINE: {
        "why": [
            "High Cortisol (Stress) levels", 
            "Hormonal spikes (Testosterone/Androgens)", 
            "Lymphatic drainage congestion"
        ],
        "lifestyle": [
            "Practice lymphatic drainage massage", 
            "Prioritize stress management (meditation/sleep)", 
            "Consult dermatologist for hormonal evaluation", 
            "Avoid resting chin on hands"
        ]
    }
}

SKIN_INDICATOR_INFO = {
    SkinIndicator.OILY_SKIN: {
        "advice": "Use a gentle foaming cleanser and Niacinamide. Do NOT skip moisturizer; use a gel-based one."
    },
    SkinIndicator.INFLAMED_AREAS: {
        "advice": "Focus on barrier repair. Use Centella Asiatica (Cica), Aloe, or Panthenol. Pause actives like retinol until healed."
    },
    SkinIndicator.DRYNESS: {
        "advice": "Layer hydration (toner > serum > cream). Look for Ceramides, Hyaluronic Acid, and Squalane."
    },
    SkinIndicator.UNEVEN_TEXTURE: {
        "advice": "Incorporate gentle chemical exfoliation (Lactic Acid or low % Glycolic Acid) 2-3 times a week."
    }
}

PREVENTION_TIPS = [
    " Consistency is key: Acne treatments take 4-6 weeks to show results.",
    " Double Cleanse: Use an oil cleanser/balm first, then a water-based cleanser at night.",
    " Sun Protection: Sun exposure darkens acne scars (PIH). Wear SPF 30+ daily.",
    " Diet: Your gut health reflects on your skin. Eat colorful plants and fermented foods.",
    " Hygiene: changing towels and pillowcases is the cheapest skincare upgrade."
]
