
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model import ZeroLabelModel

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

model_path = Path("models/zerolabel_model.pth")
gallery_path = Path("data/gallery")
test_path = Path("data/test")

if not model_path.exists():
    raise FileNotFoundError("Run python train.py first.")

model = ZeroLabelModel(pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

features = []
labels = []

with torch.no_grad():
    for folder in sorted(gallery_path.iterdir()):
        if not folder.is_dir():
            continue

        for file in folder.rglob("*"):
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = Image.open(file).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            features.append(model.extract_features(tensor))
            labels.append(folder.name)

if not features:
    raise ValueError(
        "Add reference images to data/gallery/class_name/."
    )

gallery_features = torch.cat(features)
correct = 0
total = 0

with torch.no_grad():
    for folder in sorted(test_path.iterdir()):
        if not folder.is_dir():
            continue

        for file in folder.rglob("*"):
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = Image.open(file).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            query = model.extract_features(tensor)

            similarities = query @ gallery_features.T
            predicted = labels[similarities.argmax().item()]
            actual = folder.name

            print(
                f"{file.name}: Actual={actual}, "
                f"Predicted={predicted}"
            )

            total += 1
            if predicted == actual:
                correct += 1

if total == 0:
    raise ValueError("Add test images to data/test/class_name/.")

print(f"Top-1 Accuracy: {correct / total * 100:.2f}%")
print(f"Correct predictions: {correct}/{total}")
