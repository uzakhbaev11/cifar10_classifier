import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import pandas as pd
import os

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    # Определяем абсолютный путь к модели
    current_dir = os.path.dirname(__file__)  # папка, где лежит streamlit_app.py
    model_path = os.path.join(current_dir, "..", "models", "cifar_net.pth")
    model_path = os.path.abspath(model_path)

    # Загружаем модель
    model = Net()
    device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

st.title("CIFAR-10 Image Classifier")
st.write("Загрузите изображение (jpg/png) — модель вернёт предсказанный класс из 10.")

uploaded_file = st.file_uploader("Выберите изображение", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    
    input_tensor = transform(image).unsqueeze(0)  
    with st.spinner("Предсказание..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

    st.subheader(f"Предсказанный класс: **{classes[top_idx]}** ({top_prob*100:.2f}%)")
    df = pd.DataFrame({"class": classes, "probability": probs})
    df = df.sort_values("probability", ascending=False)
    st.bar_chart(df.set_index("class"))
    st.write(df)
