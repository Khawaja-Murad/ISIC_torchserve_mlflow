import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

NUM_CLASSES = 9

model = models.vit_b_16(weights=None)
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, NUM_CLASSES)

# 2. Load state dict
model.load_state_dict(
    torch.load(
        "/Volumes/002-350/khawajamurad/Documents/Projects/ISIC_torchserve_mlflow/best_model.pth",
        map_location="cpu",
    )
)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.667, 0.530, 0.552], std=[0.154, 0.178, 0.197]),
    ]
)


scripted_model = torch.jit.script(model)
scripted_model.save("vit_model.pt")

loaded_model = torch.jit.load("vit_model.pt")
loaded_model.eval()

img = Image.open(
    "/Volumes/002-350/khawajamurad/Documents/Projects/ISIC_torchserve_mlflow/ISIC_0025780.jpg"
).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    original_output = model(input_tensor)
    scripted_output = loaded_model(input_tensor)
    print("Output difference:", torch.max(original_output - scripted_output).item())
