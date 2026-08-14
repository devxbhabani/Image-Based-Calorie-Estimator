import torch
import torchvision.transforms as transforms
from PIL import Image
from src.classification_model import FoodRegressionModel
from src import config

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FoodRegressionModel()
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(config.CLASSIFIER_PATH))
        else:
            self.model.load_state_dict(torch.load(config.CLASSIFIER_PATH, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_calories_and_mass(self, img_path):
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            
        calories, mass = output[0][0].item(), output[0][1].item()
        return round(calories, 2), round(mass, 2)