import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from helper_logger import DataLogger
from helper_tester import ModelTesterMetrics
from dataset import SimpleTorchDataset  
from torchvision import transforms
from model import EfficientNetClassifier  

SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_epochs = 32
batch_size = 32
initial_lr = 0.0001

if __name__ == "__main__":

    print("| Pytorch Model Training! by Boubker & Nathan ")
    print("| Total Epochs:", total_epochs)
    print("| Batch Size:", batch_size)
    print("| Device:", device)

    logger = DataLogger("VegetableClassificationEfficientNet")
    metrics = ModelTesterMetrics()

    metrics.loss = torch.nn.CrossEntropyLoss()
    metrics.activation = torch.nn.Softmax(dim=1)


    model = EfficientNetClassifier(output_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=(3, 3)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_dataset = SimpleTorchDataset('dataset\\Vegetable Images\\validation', aug=val_transforms)
    training_dataset = SimpleTorchDataset('./dataset/Vegetable Images/train', aug=train_transforms)
    testing_dataset = SimpleTorchDataset('./dataset/Vegetable Images/test', aug=val_transforms)

    validation_datasetloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    training_datasetloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_datasetloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)
    for current_epoch in range(total_epochs):
        print("Epoch:", current_epoch)
        
        model.train()
        metrics.reset()

        for (image, label) in tqdm(training_datasetloader, desc="Training:"):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = metrics.compute(output, label)
            loss.backward()
            optimizer.step()

        training_mean_loss = metrics.average_loss()
        training_mean_accuracy = metrics.average_accuracy()

        model.eval()
        metrics.reset()

        with torch.no_grad():
            for (image, label) in tqdm(validation_datasetloader, desc="Validating:"):
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                metrics.compute(output, label)

        evaluation_mean_loss = metrics.average_loss()
        evaluation_mean_accuracy = metrics.average_accuracy()
        scheduler.step(evaluation_mean_loss)
        logger.append(
            current_epoch,
            training_mean_loss,
            training_mean_accuracy,
            evaluation_mean_loss,
            evaluation_mean_accuracy
        )
        if logger.current_epoch_is_best:
            print("> Latest Best Epoch:", logger.best_accuracy())
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            state_dictonary = {
                "model_state": model_state,
                "optimizer_state": optimizer_state
            }
            torch.save(
                state_dictonary, 
                logger.get_filepath("best_checkpoint.pth")
            )

        logger.save()
        print("")

    print("| Training Complete, Loading Best Checkpoint")
    state_dictonary = torch.load(
        logger.get_filepath("best_checkpoint.pth"), 
        map_location=device
    )
    model.load_state_dict(state_dictonary['model_state'])
    model = model.to(device)
    model.eval()
    metrics.reset()

    for (image, label) in tqdm(testing_datasetloader, desc="Testing:"):
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        metrics.compute(output, label)

    testing_mean_loss = metrics.average_loss()
    testing_mean_accuracy = metrics.average_accuracy()

    print("")
    logger.write_text(f"# Final Testing Loss: {testing_mean_loss}")
    logger.write_text(f"# Final Testing Accuracy: {testing_mean_accuracy}")
    logger.write_text(f"# Report:")
    logger.write_text(metrics.report())
    logger.write_text(f"# Confusion Matrix:")
    logger.write_text(metrics.confusion())
    print("")
