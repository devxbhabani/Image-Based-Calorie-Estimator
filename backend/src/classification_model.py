import torch
import torch.nn as nn
import torchvision.models as models

class FoodRegressionModel(nn.Module):
    """
    Model designed to predict numerical targets (Calories, Mass)
    directly from an image using a ResNet50 backbone.
    """

    def __init__(self):
        super(FoodRegressionModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        # Replace classification layer with a regression layer outputting:
        # [0] Estimated Calories, [1] Estimated Mass (grams)
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.backbone(x)
