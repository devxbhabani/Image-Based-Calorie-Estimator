import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.dataloader import NutritionDataset
from src.classification_model import FoodRegressionModel
from src import config

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and DataLoader
    try:
        dataset = NutritionDataset(config.DISH_INGREDIENTS_CSV, config.IMAGERY_DIR, transform=transform)
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return
        
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Initialize Model, Loss and Optimizer
    model = FoodRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss (MSE): {avg_loss:.4f}")
        
    # Save the trained weights
    os.makedirs(os.path.dirname(config.CLASSIFIER_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.CLASSIFIER_PATH)
    print(f"Model saved to {config.CLASSIFIER_PATH}")

if __name__ == "__main__":
    import os
    train()