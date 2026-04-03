"""
AcneGuard — ResNet-18 Training Script
=====================================
Trains a ResNet-18 model (pretrained on ImageNet) on the acne severity dataset.
This script automatically handles class imbalance via inverse frequency weighting.
Logs are written to `training_log.txt` for live user monitoring.
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
parser = argparse.ArgumentParser(description="Train AcneGuard ResNet-18")
parser.add_argument("--epochs", type=int,   default=25,          help="Training epochs")
parser.add_argument("--batch",  type=int,   default=32,          help="Batch size")
parser.add_argument("--lr",     type=float, default=0.001,       help="Learning rate")
parser.add_argument("--data",   type=str,   default="./data",    help="Dataset root path")
args = parser.parse_args()

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR = args.data
OUTPUT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acne_model.pth")
CLASSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.json")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS   = args.epochs
BATCH_SIZE   = args.batch
LEARNING_RATE = args.lr

# --- Logging setup ---
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")

def log_print(msg):
    """Custom printer that also writes to a log file for user visibility."""
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

# Start fresh log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== AcneGuard Training Progress (ResNet-18) ===\n")

log_print(f"Device: {DEVICE}")
log_print(f"Dataset: {DATASET_DIR}")
log_print(f"Epochs: {NUM_EPOCHS}")
log_print(f"Batch Size: {BATCH_SIZE}")
log_print(f"Log File: {LOG_FILE}")
log_print(f"Model will be saved to: {OUTPUT_MODEL}")

# ─── Data Transforms ──────────────────────────────────────────────────────────
# ResNet-18 expects 224×224 input.
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

if not os.path.exists(train_dir):
    log_print(f"[ERROR] Train directory not found: {train_dir}")
    exit(1)

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform) if os.path.exists(val_dir) else None

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_dataset else None

class_names = train_dataset.classes
log_print(f"Classes found: {class_names}")

# --- Class Balance Handling ---
class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in class_names]
total_samples = sum(class_counts)
class_weights = torch.tensor([total_samples / (len(class_names) * count) for count in class_counts], dtype=torch.float).to(DEVICE)

log_print(f"Data Distribution: {dict(zip(class_names, class_counts))}")
log_print(f"Calculated Class Weights: {class_weights.tolist()}")

with open(CLASSES_FILE, "w") as f:
    json.dump(class_names, f)
log_print(f"Saved classes.json -> {CLASSES_FILE}")

# ─── Build ResNet-18 ─────────────────────────────────────────────────────────
log_print("Loading ResNet-18 with ImageNet weights...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the FC layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(DEVICE)

# ─── Loss, Optimizer, Scheduler ───────────────────────────────────────────────
log_print("Initializing Loss with Class Weights for better accuracy on minor categories...")
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ─── Training Loop ────────────────────────────────────────────────────────────
best_val_acc  = 0.0
best_model_state = None

for epoch in range(NUM_EPOCHS):
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
        log_print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.1f}% Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            log_print(f"  --> Updated Best Model (Val Acc: {best_val_acc:.1f}%)")
    else:
        log_print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.1f}%")
        best_model_state = model.state_dict()

    scheduler.step()

state_to_save = best_model_state if best_model_state else model.state_dict()
torch.save(state_to_save, OUTPUT_MODEL)
log_print(f"\nTraining Complete! ResNet-18 Model saved at {OUTPUT_MODEL}")
if val_loader:
    log_print(f"Final Best Validation Accuracy: {best_val_acc:.1f}%")
