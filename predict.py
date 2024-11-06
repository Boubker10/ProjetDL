import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import EfficientNetClassifier 

class_names = {
    0: "Bean",
    1: "Carrot",
    2: "Cucumber",
    3: "Potato",
    4: "Tomato",
    5: "Broccoli"
}

output_classes = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetClassifier(output_classes=output_classes).to(device)
checkpoint_path = "runs/VegetableClassificationEfficientNet-1/best_checkpoint.pth"  
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    image = test_transforms(image)
    image = image.unsqueeze(0)  
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    predicted_class = torch.argmax(probabilities).item()
    class_name = class_names[predicted_class]
    confidence = probabilities[predicted_class].item()

    return class_name, confidence

st.title("Vegetable Image Classification")
st.write(" By Boubker & Nathan ")
st.write("Upload a vegetable image to receive a model prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image uploaded.', use_column_width=True)
    st.write("Classification in progress...")
    class_name, confidence = predict_image(image)
    st.write(f"**Predicted Class :** {class_name}")
    st.write(f"**Confiance :** {confidence:.2f}")
