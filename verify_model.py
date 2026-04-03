import os
import torch
from vision import VisionService

def verify():
    print("=== AcneGuard Model Verification ===")
    
    # Initialize service
    service = VisionService()
    
    if not service.model:
        print("[ERROR] Model could not be loaded. Check acne_model.pth existence and architecture.")
        return

    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data", "val")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), "data", "train")
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found at {data_dir}")
        return

    classes = ["clear", "mild", "moderate", "severe"]
    results = {}

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.exists(cls_path):
            print(f"[SKIP] Class folder {cls} not found.")
            continue
        
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"[SKIP] No images found for class {cls}.")
            continue
        
        test_img = os.path.join(cls_path, images[0])
        print(f"\nTesting class: {cls.upper()}")
        print(f"Image: {test_img}")
        
        try:
            analysis = service.analyze_image(test_img)
            results[cls] = analysis.severity.value
            print(f"Result: {analysis.severity.value}")
        except Exception as e:
            print(f"Error analyzing {cls}: {e}")

    print("\n=== Summary ===")
    for cls, pred in results.items():
        status = "✅" if cls.lower() == pred.lower() else "❌"
        print(f"Expected: {cls:8} | Predicted: {pred:8} {status}")

if __name__ == "__main__":
    verify()
