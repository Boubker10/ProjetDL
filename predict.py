import torch
from torchvision import transforms
from PIL import Image
from model import EfficientNetClassifier  # Assurez-vous que le modèle est importé correctement
import tkinter as tk
from tkinter import filedialog, Label, Button
import os

# Dictionnaire pour mapper les numéros de classes aux noms
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

# Charger le modèle et le checkpoint
model = EfficientNetClassifier(output_classes=output_classes).to(device)
checkpoint_path = "runs\\VegetableClassificationEfficientNet-1\\best_checkpoint.pth"  # Modifier le chemin si nécessaire
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Transformations pour les images de test
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fonction pour effectuer la prédiction sur une image
def predict_image(image_path):
    # Charger et transformer l'image
    image = Image.open(image_path).convert('RGB')
    image = test_transforms(image)
    image = image.unsqueeze(0)  # Ajouter une dimension pour le batch

    # Déplacer l'image sur le même appareil que le modèle
    image = image.to(device)

    # Effectuer la prédiction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Renvoyer le nom de la classe prédite et sa probabilité
    predicted_class = torch.argmax(probabilities).item()
    class_name = class_names[predicted_class]
    confidence = probabilities[predicted_class].item()

    return class_name, confidence

# Fonction de gestion de l'interface Tkinter pour charger et prédire une image
def upload_and_predict():
    # Ouvrir une boîte de dialogue pour sélectionner une image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        # Afficher le chemin de l'image sélectionnée
        label_file_explorer.config(text="Image sélectionnée: " + file_path)
        
        # Effectuer la prédiction
        class_name, confidence = predict_image(file_path)
        
        # Afficher les résultats
        result_text = f"Classe prédite: {class_name} \nConfiance: {confidence:.2f}"
        label_result.config(text=result_text)

# Interface graphique Tkinter
window = tk.Tk()
window.title("Classification d'Images de Légumes")
window.geometry("500x300")

# Label pour le chemin de l'image sélectionnée
label_file_explorer = Label(window, text="Aucune image sélectionnée", width=50, height=2)
label_file_explorer.pack()

# Bouton pour ouvrir le sélecteur de fichier
button_explore = Button(window, text="Sélectionner une image", command=upload_and_predict)
button_explore.pack()

# Label pour afficher les résultats de la prédiction
label_result = Label(window, text="", width=50, height=4)
label_result.pack()

# Lancer l'interface Tkinter
window.mainloop()
