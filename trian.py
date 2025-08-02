import torch
from torchvision import models, transforms
from PIL import Image

# --- Model Setup (match training pipeline) ---
num_classes = 2
model = models.efficientnet_v2_s(weights=None)      # No pretrained weights for inference
model.classifier[1] = torch.nn.Linear(1280, num_classes)
model.load_state_dict(torch.load('efficientnetv2_glaucoma.pth', map_location='cpu'))
model.eval()

# --- Preprocessing (must match your training pipeline) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

label_map = {0: "Glaucoma not Present", 1: "Glaucoma Present"}

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        out = model(inp)
        pred = out.argmax(1).item()
    print(f"Prediction: {label_map[pred]}")

if __name__ == '__main__':
    # Example usage; replace 'images/test1.jpg' with your image path
    predict_image("images/test1.jpg")
