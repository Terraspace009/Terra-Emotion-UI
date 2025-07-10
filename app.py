import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np

# -------------------------------
# Load model
# -------------------------------
from torchvision import models

# Define model structure
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 7)  # 7 emotion classes

# Load weights
model.load_state_dict(torch.load("terra_emotion_mobilenet_optimized.pt", map_location=torch.device("cpu")))
model.eval()

# -------------------------------
# Setup
# -------------------------------
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

ascii_chars = "@%#*+=-:. "

def image_to_ascii(img_tensor):
    img_array = img_tensor.squeeze().numpy()
    img_rescaled = ((img_array - img_array.min()) / (img_array.ptp() + 1e-5) * (len(ascii_chars) - 1)).astype(int)
    ascii_art = "\n".join("".join(ascii_chars[p] for p in row) for row in img_rescaled)
    return ascii_art

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="TERRA Emotion UI", layout="centered")
st.title("🧠 TERRA Emotion UI")
st.caption("Upload a face image to predict emotion and view ASCII art")

uploaded_file = st.file_uploader("📷 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, 1).item()
        emotion = emotion_labels[predicted]

    st.subheader("🎭 Detected Emotion")
    st.success(emotion)

    ascii_art = image_to_ascii(img_tensor)
    st.subheader("🎨 ASCII Art")
    st.text(ascii_art)

