import torch
import torch.nn as nn
import torchvision as tv

class EfficientNetClassifier(nn.Module):
    def __init__(self, output_classes: int) -> None:
        super().__init__()
        # Chargement de EfficientNet-B0 pré-entraîné sur ImageNet
        self.base = tv.models.efficientnet_b0(weights=tv.models.EfficientNet_B0_Weights.DEFAULT)
        
        # Remplacement de la dernière couche de classification
        self.base.classifier[1] = nn.Linear(self.base.classifier[1].in_features, output_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

if __name__ == "__main__":
    print("EfficientNet-B0 Model for Vegetable Classification")

    # Exemple d'entrée pour tester le modèle
    t = torch.rand(1, 3, 128, 128)  # Les images EfficientNet-B0 attendent une taille de 128x128

    # Instancier le modèle pour 7 classes de légumes (par exemple)
    output_classes = 6
    model = EfficientNetClassifier(output_classes)
    
    # Passer une entrée fictive pour vérifier le modèle
    y = model(t)
    print("Output Shape:", y.shape)
