import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Ensure CPU inference for Streamlit
device = torch.device("cpu")

# Model definition (must match training setup)
num_classes = 2
model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = torch.nn.Linear(1280, num_classes)
model.load_state_dict(torch.load('efficientnetv2_glaucoma.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
label_map = {0: "Glaucoma not Present", 1: "Glaucoma Present"}

st.title("Glaucoma Detection from Fundus Image")
uploaded_file = st.file_uploader("Upload fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded image', use_container_width=True)  # Use updated Streamlit param
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        pred = output.argmax(1).item()
    st.write(f"**Prediction:** {label_map[pred]}")

