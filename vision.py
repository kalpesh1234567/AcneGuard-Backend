"""
AcneGuard AI — Vision Service
Handles ResNet-18-based severity prediction + AI-based acne localization.
"""
from schemas import (
    AcneAnalysis, AcneSeverity, AcneType, AcneLocation,
    SkinIndicator, LocalizationResult, RegionScores
)
import os
import json

try:
    import torch
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[VisionService] Torch not available. Running mock mode.")

from localization import AcneLocalizer


class VisionService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Use absolute path to find model relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "acne_model.pth")
            classes_path = os.path.join(base_dir, "classes.json")
        else:
            classes_path = "classes.json"

        self.model       = None
        self.device      = None
        self.class_names = []

        # Initialize localization engine
        self.localizer = AcneLocalizer()

        if not TORCH_AVAILABLE:
            print("[VisionService] Torch not installed — severity will be mocked.")
            return

        if not os.path.exists(model_path):
            print(f"[VisionService] Model file not found: {model_path}")
            return

        if not os.path.exists(classes_path):
            print(f"[VisionService] classes.json not found: {classes_path}")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(classes_path, "r") as f:
            self.class_names = json.load(f)

        print("[VisionService] Loaded classes:", self.class_names)

        # ─── Build ResNet-18 model ──────────────────────────────────────────────────
        try:
            res_model = models.resnet18(weights=None)
            num_ftrs = res_model.fc.in_features
            res_model.fc = torch.nn.Linear(num_ftrs, len(self.class_names))
            res_model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=False)
            )
            self.model = res_model.to(self.device)
            self.model.eval()
            print("[VisionService] ResNet-18 model loaded successfully.")
        except Exception as e:
            print(f"[VisionService] ⚠ Could not load ResNet-18 weights: {e}")
            print("[VisionService] Running in mock mode.")
            self.model = None
            return

        # ResNet-18 is trained on 224x224 with the same ImageNet stats
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("[VisionService] ResNet-18 transform pipeline ready.")

    def analyze_image(self, image_path: str) -> AcneAnalysis:
        """
        Full analysis pipeline:
        1. CNN → Acne Severity (clear / mild / moderate / severe)
        2. AcneLocalizer → Location (forehead / cheeks / nose / chin / jawline)
        3. Return AcneAnalysis with both results + full localization metadata
        """
        # ─── Default values (used if model not loaded) ───────────────────────
        severity   = AcneSeverity.MODERATE
        acne_type  = AcneType.PUSTULES
        indicators = [SkinIndicator.OILY_SKIN]

        # ─── Step 1: ResNet-18 Severity Prediction ───────────────────────────
        if self.model:
            try:
                image = Image.open(image_path).convert("RGB")
                tensor = self.transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probs, 1)

                idx = pred.item()
                predicted_label = self.class_names[idx].lower()
                print(f"[VisionService] ResNet-18 Prediction: {predicted_label} "
                      f"(confidence={confidence.item():.3f})")

                # Direct severity mapping
                severity_map = {
                    "clear":    AcneSeverity.CLEAR,
                    "mild":     AcneSeverity.MILD,
                    "moderate": AcneSeverity.MODERATE,
                    "severe":   AcneSeverity.SEVERE,
                }
                if predicted_label in severity_map:
                    severity = severity_map[predicted_label]

                # Infer acne type from severity (rule-based heuristic)
                type_from_severity = {
                    AcneSeverity.CLEAR:    AcneType.WHITEHEADS,
                    AcneSeverity.MILD:     AcneType.BLACKHEADS,
                    AcneSeverity.MODERATE: AcneType.PUSTULES,
                    AcneSeverity.SEVERE:   AcneType.NODULES,
                }
                acne_type = type_from_severity.get(severity, AcneType.PUSTULES)

                # Indicators from severity
                if severity in [AcneSeverity.MODERATE, AcneSeverity.SEVERE]:
                    indicators = [SkinIndicator.OILY_SKIN, SkinIndicator.INFLAMED_AREAS]
                elif severity == AcneSeverity.MILD:
                    indicators = [SkinIndicator.OILY_SKIN, SkinIndicator.UNEVEN_TEXTURE]
                else:
                    indicators = []

            except Exception as e:
                print(f"[VisionService] CNN inference failed: {e}")

        print(f"[VisionService] Final Severity: {severity.value}")

        # ─── Step 2: AI Localization ──────────────────────────────────────────
        print("[VisionService] Running AI acne localization...")
        loc_raw = self.localizer.detect(image_path)
        print(f"[VisionService] {loc_raw}")

        # ── Multi-face check — reject before building AcneAnalysis ────────────
        if loc_raw.method == "multi_face_rejected":
            raise ValueError("MULTI_FACE: Multiple faces detected in the image.")

        # Map string location to AcneLocation enum
        try:
            location_enum = AcneLocation(loc_raw.location)
        except ValueError:
            location_enum = AcneLocation.CHEEKS

        # Build Pydantic LocalizationResult
        try:
            localization_result = LocalizationResult(
                location=location_enum,
                region_scores=RegionScores(
                    forehead=loc_raw.region_scores.get("forehead", 0.0),
                    cheeks=loc_raw.region_scores.get("cheeks",   0.0),
                    nose=loc_raw.region_scores.get("nose",     0.0),
                    chin=loc_raw.region_scores.get("chin",     0.0),
                    jawline=loc_raw.region_scores.get("jawline",  0.0),
                ),
                confidence=loc_raw.confidence,
                method=loc_raw.method,
                face_detected=loc_raw.face_detected,
            )
        except Exception as e:
            print(f"[VisionService] Localization result build failed: {e}")
            localization_result = None

        # ─── Step 3: Assemble final AcneAnalysis ─────────────────────────────
        return AcneAnalysis(
            severity=severity,
            acne_type=acne_type,
            location=location_enum,
            indicators=indicators,
            localization=localization_result,
        )

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, "localizer") and self.localizer:
            self.localizer.close()

