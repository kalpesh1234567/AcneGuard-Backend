"""
AcneGuard — ResNet-18 Resume Training Script
=============================================
Resumes training from an existing acne_model.pth checkpoint.
Used to extend training beyond the initial run (e.g. 25 -> 40 epochs).

USAGE:
    cd backend
    python resume_train_resnet.py --epochs 15
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms

# ─── Argument Parser ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Resume training AcneGuard ResNet-18")
parser.add_argument("--epochs", type=int,   default=15,       help="Additional training epochs")
parser.add_argument("--batch",  type=int,   default=32,       help="Batch size")
parser.add_argument("--lr",     type=float, default=0.0005,   help="Learning rate (lower for fine-tuning)")
parser.add_argument("--data",   type=str,   default="./data", help="Dataset root path")
args = parser.parse_args()

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR  = args.data
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "acne_model.pth")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.json")
LOG_FILE     = os.path.join(BASE_DIR, "training_log.txt")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS   = args.epochs
BATCH_SIZE   = args.batch
LEARNING_RATE = args.lr

# ─── Logging ──────────────────────────────────────────────────────────────────
def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

# Append to existing log
with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write("\n=== RESUMED TRAINING (15 more epochs) ===\n")

log_print(f"Device        : {DEVICE}")
log_print(f"Dataset       : {DATASET_DIR}")
log_print(f"Extra Epochs  : {NUM_EPOCHS}")
log_print(f"Learning Rate : {LEARNING_RATE} (reduced for fine-tuning)")

# ─── Verify checkpoint exists ─────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] No model found at {MODEL_PATH}. Run train_resnet.py first.")
    exit(1)

log_print(f"Loading checkpoint from: {MODEL_PATH}")

# ─── Load classes ─────────────────────────────────────────────────────────────
with open(CLASSES_FILE, "r") as f:
    class_names = json.load(f)
log_print(f"Classes: {class_names}")

# ─── Data Transforms ──────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Load Dataset ─────────────────────────────────────────────────────────────
train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "val")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transform) if os.path.exists(val_dir) else None

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_dataset else None

# ─── Class Weights ────────────────────────────────────────────────────────────
class_counts  = [len(os.listdir(os.path.join(train_dir, c))) for c in class_names]
total_samples = sum(class_counts)
class_weights = torch.tensor(
    [total_samples / (len(class_names) * count) for count in class_counts],
    dtype=torch.float
).to(DEVICE)
log_print(f"Class Weights: {class_weights.tolist()}")

# ─── Rebuild & Load ResNet-18 ─────────────────────────────────────────────────
log_print("Rebuilding ResNet-18 architecture...")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model = model.to(DEVICE)
log_print("Checkpoint loaded successfully! Resuming training...")

# ─── Loss, Optimizer, Scheduler ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights)
# Use lower LR for fine-tuning on top of already-trained weights
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ─── Resume Training Loop ─────────────────────────────────────────────────────
best_val_acc     = 0.0
best_model_state = None
EPOCH_OFFSET     = 25  # We already did 25 epochs

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total * 100

    # Validation
    if val_loader:
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = val_correct / val_total * 100
        log_print(f"Epoch [{EPOCH_OFFSET + epoch + 1:02d}/40] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.1f}% Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            log_print(f"  --> Updated Best Model (Val Acc: {best_val_acc:.1f}%)")
    else:
        log_print(f"Epoch [{EPOCH_OFFSET + epoch + 1:02d}/40] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.1f}%")
        best_model_state = model.state_dict()

    scheduler.step()

# ─── Save Final Model ─────────────────────────────────────────────────────────
state_to_save = best_model_state if best_model_state else model.state_dict()
torch.save(state_to_save, MODEL_PATH)
log_print(f"\nResume Training Complete! Model saved at {MODEL_PATH}")
if val_loader:
    log_print(f"Final Best Validation Accuracy (all 40 epochs): {best_val_acc:.1f}%")
