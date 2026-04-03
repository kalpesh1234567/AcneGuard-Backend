"""
AcneGuard AI — True Acne Localization Module
Pipeline: MediaPipe Face Mesh (478 landmarks) → Region Polygons →
          OpenCV HSV Red/Inflamed Pixel Density → Ranked Location Detection

Supports MediaPipe 0.10+ Tasks API (mp.tasks.vision.FaceLandmarker).
Falls back to quadrant analysis if MediaPipe/OpenCV not available.
"""
import os
import numpy as np
from typing import Dict, Optional

# Optional imports — graceful fallback if libs not installed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Localizer] OpenCV not available. Install: pip install opencv-python")

MP_AVAILABLE = False
MP_TASKS_AVAILABLE = False
try:
    import mediapipe as mp
    # MediaPipe 0.10+ uses Tasks API
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_TASKS_AVAILABLE = True
    MP_AVAILABLE = True
    print("[Localizer] MediaPipe Tasks API available (v0.10+).")
except Exception as _mp_err:
    # Check if legacy solutions API exists
    try:
        import mediapipe as mp
        _ = mp.solutions.face_mesh
        MP_AVAILABLE = True
        print("[Localizer] MediaPipe legacy solutions API available.")
    except Exception:
        print(f"[Localizer] MediaPipe not usable: {_mp_err}. Install: pip install mediapipe")


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh Landmark Indices per Facial Region
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ─────────────────────────────────────────────────────────────────────────────

REGION_LANDMARKS = {
    "forehead": [
        10, 338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132,
        93, 234, 127, 162, 21, 54, 103, 67, 109,
        10, 338, 297, 332, 284
    ],
    # Actual forehead (top of face above eyebrows)
    "forehead_precise": [10, 67, 109, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],

    "cheeks": [
        # Right cheek
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
        # Left cheek
        454, 323, 361, 288, 397, 365, 379, 378, 400, 377
    ],

    "nose": [
        1, 2, 5, 4, 19, 94, 98, 327, 326, 19,
        49, 279, 48, 278, 64, 294, 168, 6, 197,
        195, 5, 4, 1, 19, 94
    ],

    "chin": [
        152, 377, 400, 378, 379, 365, 397, 288, 361,
        323, 454, 356, 389, 251, 284, 332, 297, 338,
        10, 109, 103, 54, 21, 162, 127, 234, 93,
        132, 172, 136, 150, 149, 176, 148, 152
    ],
    # Actual chin (bottom of face)
    "chin_precise": [152, 377, 400, 378, 379, 365, 397, 435, 288, 216, 206, 426, 423, 147, 213, 192, 176, 149, 150, 136, 172],

    "jawline": [
        # Right jawline
        136, 150, 149, 176, 148, 152,
        # Left jawline
        365, 379, 378, 400, 377, 152,
        # Lower jaw connectors
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10
    ]
}

# Simplified, clean region definitions for polygon extraction
FACE_REGIONS = {
    "forehead": [10, 67, 109, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172],
    "cheeks":   [234, 93, 132, 58, 172, 136, 150, 149, 400, 378, 379, 365, 397, 288, 361, 323, 454],
    "nose":     [168, 6, 197, 195, 5, 4, 45, 275, 220, 440, 48, 278, 64, 294],
    "chin":     [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 152],
    "jawline":  [136, 172, 58, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
                 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 365, 379, 400, 149, 150]
}


class LocalizationResult:
    """Holds the result of acne localization."""
    def __init__(
        self,
        location: str,
        region_scores: Dict[str, float],
        confidence: float,
        method: str,
        face_detected: bool
    ):
        self.location = location
        self.region_scores = region_scores
        self.confidence = confidence
        self.method = method
        self.face_detected = face_detected

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "region_scores": self.region_scores,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "face_detected": self.face_detected
        }

    def __repr__(self):
        scores_str = ", ".join(f"{k}={v:.3f}" for k, v in self.region_scores.items())
        return (
            f"LocalizationResult(location='{self.location}', "
            f"confidence={self.confidence:.2f}, method='{self.method}', "
            f"scores=[{scores_str}])"
        )


class AcneLocalizer:
    """
    Detects acne location using MediaPipe Face Mesh + OpenCV HSV analysis.

    Steps:
    1. Load image via OpenCV
    2. Run MediaPipe FaceMesh to extract 468 landmarks
    3. Define 5 facial region polygons from landmark coords
    4. For each region: count red/yellow inflamed pixels (HSV range)
    5. Normalize scores → pick highest density region
    6. Return LocalizationResult with full region scores + confidence
    """

    # HSV color ranges for detecting red/inflamed/acne-like pixels
    # Red in HSV wraps around (0-10 AND 160-180)
    HSV_RED_LOWER1 = np.array([0,   60,  60])
    HSV_RED_UPPER1 = np.array([10, 255, 255])
    HSV_RED_LOWER2 = np.array([160, 60,  60])
    HSV_RED_UPPER2 = np.array([180, 255, 255])

    # Yellow-ish tones (pustules, inflamed skin)
    HSV_YELLOW_LOWER = np.array([15, 40, 60])
    HSV_YELLOW_UPPER = np.array([35, 255, 255])

    def __init__(self):
        self.face_mesh = None
        self.use_tasks_api = False
        self._init_mediapipe()

    def _init_mediapipe(self):
        if not MP_AVAILABLE:
            print("[Localizer] Skipping MediaPipe init — not installed.")
            return
        try:
            if MP_TASKS_AVAILABLE:
                # MediaPipe 0.10+ Tasks API — use FaceLandmarker model
                # Download model bytes inline via the bundled asset
                import urllib.request, tempfile
                MODEL_URL = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                )
                model_path = os.path.join(
                    tempfile.gettempdir(), "face_landmarker.task"
                )
                if not os.path.exists(model_path):
                    print("[Localizer] Downloading FaceLandmarker model (~7MB)...")
                    urllib.request.urlretrieve(MODEL_URL, model_path)
                    print("[Localizer] Model downloaded.")

                base_options = mp_python.BaseOptions(
                    model_asset_path=model_path
                )
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=2,   # detect up to 2 so we can reject multi-face images
                    min_face_detection_confidence=0.4,
                    min_face_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
                self.face_mesh = mp_vision.FaceLandmarker.create_from_options(options)
                self.use_tasks_api = True
                print("[Localizer] MediaPipe FaceLandmarker (Tasks API) initialized.")
            else:
                # Legacy solutions API
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=2,  # detect up to 2 so we can reject multi-face images
                    refine_landmarks=True,
                    min_detection_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
                self.use_tasks_api = False
                print("[Localizer] MediaPipe FaceMesh (legacy) initialized.")
        except Exception as e:
            print(f"[Localizer] MediaPipe init failed: {e}")
            self.face_mesh = None

    def detect(self, image_path: str) -> LocalizationResult:
        """
        Main entry point. Takes an image path, returns LocalizationResult.
        Falls back to image-quadrant analysis if face mesh fails.
        """
        if not CV2_AVAILABLE:
            return self._fallback_result("no_opencv")

        image = self._load_image(image_path)
        if image is None:
            return self._fallback_result("image_load_failed")

        # Try MediaPipe Face Mesh detection
        if self.face_mesh is not None and MP_AVAILABLE:
            result = self._detect_with_facemesh(image)
            if result is not None:
                return result
            print("[Localizer] Face mesh failed — falling back to quadrant analysis.")

        # Fallback: analyze image quadrants directly
        return self._detect_with_quadrants(image)

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image using OpenCV."""
        if not os.path.exists(image_path):
            print(f"[Localizer] Image not found: {image_path}")
            return None
        try:
            img = cv2.imread(image_path)
            if img is None:
                # Try alternate read for some formats
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print(f"[Localizer] Image load error: {e}")
            return None

    def _detect_with_facemesh(self, image: np.ndarray) -> Optional[LocalizationResult]:
        """
        Full MediaPipe face mesh detection pipeline.
        Supports both Tasks API (0.10+) and legacy solutions API.
        Returns None if no face detected.
        """
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            if self.use_tasks_api:
                # New Tasks API (mediapipe 0.10+)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image
                )
                detection_result = self.face_mesh.detect(mp_image)
                if not detection_result.face_landmarks:
                    print("[Localizer] No face detected in image (Tasks API).")
                    return None
                # ── Multi-face rejection ─────────────────────────────────────
                if len(detection_result.face_landmarks) > 1:
                    print(f"[Localizer] {len(detection_result.face_landmarks)} faces detected — rejecting image.")
                    return self._multi_face_result()
                raw_landmarks = detection_result.face_landmarks[0]
                # Convert NormalizedLandmark to pixel coords
                points = np.array(
                    [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks],
                    dtype=np.int32
                )
            else:
                # Legacy solutions API
                results = self.face_mesh.process(rgb_image)
                if not results.multi_face_landmarks:
                    print("[Localizer] No face detected in image (legacy API).")
                    return None
                # ── Multi-face rejection ─────────────────────────────────────
                if len(results.multi_face_landmarks) > 1:
                    print(f"[Localizer] {len(results.multi_face_landmarks)} faces detected — rejecting image.")
                    return self._multi_face_result()
                raw_landmarks = results.multi_face_landmarks[0].landmark
                points = np.array(
                    [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks],
                    dtype=np.int32
                )
        except Exception as e:
            print(f"[Localizer] Face mesh processing error: {e}")
            return None

        landmarks = None  # not used below — using 'points' directly

        # 'points' is already built above per API branch

        # Build acne mask (red/inflamed pixels)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        acne_mask = self._build_acne_mask(hsv)

        # Score each facial region
        region_scores = {}
        region_areas = {}

        for region_name, landmark_indices in FACE_REGIONS.items():
            # Get valid landmark coordinates
            valid_pts = []
            for idx in landmark_indices:
                if idx < len(points):
                    valid_pts.append(points[idx])

            if len(valid_pts) < 3:
                region_scores[region_name] = 0.0
                region_areas[region_name] = 1
                continue

            # Create region mask from landmark polygon
            region_mask = np.zeros((h, w), dtype=np.uint8)
            polygon_pts = np.array(valid_pts, dtype=np.int32)
            cv2.fillConvexPoly(region_mask, cv2.convexHull(polygon_pts), 255)

            region_area = np.sum(region_mask > 0)
            if region_area == 0:
                region_scores[region_name] = 0.0
                region_areas[region_name] = 1
                continue

            # Count acne pixels within this region
            acne_in_region = np.sum((acne_mask > 0) & (region_mask > 0))
            density = float(acne_in_region) / float(region_area)
            region_scores[region_name] = density
            region_areas[region_name] = region_area

        # Normalize scores to 0–1 relative to each other
        region_scores = self._normalize_scores(region_scores)

        # Pick winner
        best_region = max(region_scores, key=region_scores.get)
        best_score = region_scores[best_region]
        location = self._map_region_to_location(best_region)

        print(f"[Localizer] Region scores: {region_scores}")
        print(f"[Localizer] Detected location: {location} (confidence={best_score:.3f})")

        return LocalizationResult(
            location=location,
            region_scores=region_scores,
            confidence=best_score,
            method="mediapipe_facemesh",
            face_detected=True
        )

    def _build_acne_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Build a binary mask of red/inflamed/acne-colored pixels using HSV."""
        # Red range 1 (lower hue)
        mask_red1 = cv2.inRange(hsv_image, self.HSV_RED_LOWER1, self.HSV_RED_UPPER1)
        # Red range 2 (upper hue — wraps around)
        mask_red2 = cv2.inRange(hsv_image, self.HSV_RED_LOWER2, self.HSV_RED_UPPER2)
        # Yellow / pustule tones
        mask_yellow = cv2.inRange(hsv_image, self.HSV_YELLOW_LOWER, self.HSV_YELLOW_UPPER)

        combined = cv2.bitwise_or(mask_red1, mask_red2)
        combined = cv2.bitwise_or(combined, mask_yellow)

        # Morphological cleanup — remove noise, fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        return combined

    def _detect_with_quadrants(self, image: np.ndarray) -> LocalizationResult:
        """
        Fallback: divide face into vertical/horizontal zones if MediaPipe unavailable.
        Not as accurate as face mesh but still better than hardcoded value.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        acne_mask = self._build_acne_mask(hsv)

        # Define approximate regions as image quadrants
        quadrant_defs = {
            "forehead": acne_mask[0: h // 4, w // 5: 4 * w // 5],
            "nose":     acne_mask[h // 4: h // 2, 2 * w // 5: 3 * w // 5],
            "cheeks":   np.hstack([
                acne_mask[h // 4: 3 * h // 4, 0: 2 * w // 5],
                acne_mask[h // 4: 3 * h // 4, 3 * w // 5: w]
            ]),
            "chin":     acne_mask[5 * h // 8: 3 * h // 4, w // 4: 3 * w // 4],
            "jawline":  acne_mask[3 * h // 4: h, :]
        }

        region_scores = {}
        for name, region in quadrant_defs.items():
            area = region.size
            acne_pixels = int(np.sum(region > 0))
            region_scores[name] = float(acne_pixels) / float(area) if area > 0 else 0.0

        region_scores = self._normalize_scores(region_scores)
        best_region = max(region_scores, key=region_scores.get)
        location = self._map_region_to_location(best_region)
        confidence = region_scores[best_region]

        print(f"[Localizer] Quadrant fallback scores: {region_scores}")
        print(f"[Localizer] Detected location: {location} (confidence={confidence:.3f})")

        return LocalizationResult(
            location=location,
            region_scores=region_scores,
            confidence=confidence,
            method="quadrant_fallback",
            face_detected=False
        )

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize density scores to 0–1 scale relative to max score."""
        total = sum(scores.values())
        if total == 0:
            # No acne detected — return equal distribution
            n = len(scores)
            return {k: round(1.0 / n, 4) for k in scores}

        max_val = max(scores.values())
        if max_val == 0:
            n = len(scores)
            return {k: round(1.0 / n, 4) for k in scores}

        # Normalize by max (so winner = 1.0, others relative)
        # Then scale to proportion (0–1, sums to 1)
        total = sum(scores.values())
        return {k: round(float(v) / total, 4) for k, v in scores.items()}

    def _map_region_to_location(self, region_name: str) -> str:
        """Maps region name string to AcneLocation enum value string."""
        mapping = {
            "forehead": "forehead",
            "cheeks":   "cheeks",
            "nose":     "nose",
            "chin":     "chin",
            "jawline":  "jawline"
        }
        return mapping.get(region_name, "cheeks")

    def _fallback_result(self, reason: str) -> LocalizationResult:
        """Returns a safe fallback when detection cannot run."""
        print(f"[Localizer] Using static fallback. Reason: {reason}")
        equal_score = round(1.0 / 5, 4)
        return LocalizationResult(
            location="cheeks",
            region_scores={
                "forehead": equal_score,
                "cheeks":   equal_score,
                "nose":     equal_score,
                "chin":     equal_score,
                "jawline":  equal_score
            },
            confidence=0.0,
            method=f"static_fallback:{reason}",
            face_detected=False
        )

    def _multi_face_result(self) -> LocalizationResult:
        """Returns a rejection result when multiple faces are detected in the image."""
        equal_score = round(1.0 / 5, 4)
        return LocalizationResult(
            location="cheeks",
            region_scores={
                "forehead": equal_score,
                "cheeks":   equal_score,
                "nose":     equal_score,
                "chin":     equal_score,
                "jawline":  equal_score
            },
            confidence=0.0,
            method="multi_face_rejected",
            face_detected=False
        )

    def close(self):
        """Release MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
