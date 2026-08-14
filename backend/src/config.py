import os

#backenf root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#dataset directories
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGERY_DIR = os.path.join(DATA_DIR, "imagery")

#metadata files
DISH_INGREDIENTS_CSV = os.path.join(BASE_DIR, "dish_ingredients.csv")

## Models weights dir
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classification", "best_classifier.pth")

#parametres
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 15