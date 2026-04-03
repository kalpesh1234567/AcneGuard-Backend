"""
train_model.py — Production-grade Random Forest trainer for diet-to-acne risk.

Improvements over v1:
  - class_weight='balanced' to handle any class imbalance
  - 300 estimators with max_depth tuning for better generalization
  - Stratified train/test split
  - Per-class precision/recall/F1 reporting
  - Feature importance printed for interpretability
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE     = "dataset/skin_health_train.csv"
MODEL_FILE    = "skin_health_backend/ml/model.joblib"
ENCODERS_FILE = "skin_health_backend/ml/encoders.joblib"
RANDOM_SEED   = 42

# ── Feature schema (must match generate_synthetic_data.py) ───────────────────
ORDINAL_COLS = ['sugar_intake', 'dairy_intake', 'processed_food_freq',
                'spicy_food_freq', 'oily_food_level', 'exercise']

ORDINAL_CATEGORIES = [
    ['Low', 'Medium', 'High'],       # sugar
    ['None', 'Occasional', 'Daily'], # dairy
    ['Low', 'Medium', 'High'],       # processed
    ['Low', 'Medium', 'High'],       # spicy
    ['Low', 'Medium', 'High'],       # oily
    ['No', 'Yes'],                   # exercise
]

NOMINAL_COLS = ['diet_type', 'food_category', 'skin_type']
NUMERIC_COLS = ['water_intake_liters', 'sleep_hours', 'stress_level']


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(categories=ORDINAL_CATEGORIES), ORDINAL_COLS),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), NOMINAL_COLS),
        ('num', 'passthrough', NUMERIC_COLS),
    ])


def train():
    # ── Load ──
    print(f"Loading dataset from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Shape: {df.shape}")

    # Drop human-readable column not needed for model
    X = df.drop(columns=['acne_risk', 'specific_food_item'])
    y = df['acne_risk']

    X.fillna('None', inplace=True)

    print("\nClass distribution:")
    for label, count in y.value_counts().items():
        print(f"  {label:8s}: {count:,}  ({count/len(y)*100:.1f}%)")

    # ── Encode features ──
    print("\nBuilding preprocessor pipeline...")
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    feature_names = (
        ORDINAL_COLS
        + list(preprocessor.named_transformers_['nom'].get_feature_names_out(NOMINAL_COLS))
        + NUMERIC_COLS
    )

    # ── Encode target ──
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    print(f"\nTarget classes: {list(target_le.classes_)}")

    # ── Stratified split (ensures class ratio in both train+test) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded,
        test_size=0.20, random_state=RANDOM_SEED, stratify=y_encoded
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Train Random Forest ──
    print("\nTraining Random Forest (300 trees, class_weight=balanced)...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,          # let trees grow fully
        min_samples_split=4,     # prevents tiny splits
        class_weight='balanced', # corrects any residual imbalance
        random_state=RANDOM_SEED,
        n_jobs=-1,               # use all CPU cores
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    # ── Cross-validation ──
    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf, X_processed, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

    # ── Feature importance (top 10) ──
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 Feature Importances:")
    for i in top_idx:
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        print(f"  {name:35s}: {importances[i]:.4f}")

    # ── Save artifacts ──
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    artifacts = {
        'preprocessor':   preprocessor,
        'target_encoder': target_le,
        'feature_names':  feature_names,
    }
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(artifacts, ENCODERS_FILE)
    print(f"\n✅ Model saved   → {MODEL_FILE}")
    print(f"✅ Encoders saved → {ENCODERS_FILE}")


if __name__ == "__main__":
    train()
