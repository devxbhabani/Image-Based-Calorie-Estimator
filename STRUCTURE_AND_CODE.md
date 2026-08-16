# 📁 Professional React + Flask Project Structure & Boilerplate Code

This document outlines a professional full-stack architecture for your **Image-Based Food Calorie Estimator** project. It uses a **Python Flask** backend for depth estimation/machine learning inference and a modern **React (Vite)** frontend with custom glassmorphism styling.

---

## 🏃 Step-by-Step Execution Guide

To run this React + Flask project, follow this order of operations:

### **Step 1: Setup the Python Backend**
```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create and activate a virtual environment
python -m venv env
.\env\Scripts\activate

# 3. Install backend dependencies
pip install -r requirements.txt

# 4. Place your "dish_ingredients.csv" inside the "backend" directory

# 5. Download the image subset
python download_subset.py

# 6. Train the regression model (Optional - generates best_classifier.pth)
python -m src.train

# 7. Start the Flask API server
python app.py
```
*The backend API will run on `http://127.0.0.1:5000`.*

### **Step 2: Setup the React Frontend**
Open a new terminal window:
```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install node packages
npm install

# 3. Start the Vite dev server
npm run dev
```
*The frontend user interface will run on `http://localhost:5173`.*

---

## 📌 Directory Layout

```text
idp-food-calorie-estimator/
│
├── backend/                    # Python Flask Server
│   ├── app.py                  # API Server entry point
│   ├── requirements.txt        # Backend dependencies
│   ├── download_subset.py      # Dataset image subset downloader
│   ├── dish_ingredients.csv    # Place your Kaggle CSV file here
│   │
│   └── src/                    # Machine Learning Pipeline
│       ├── __init__.py
│       ├── config.py
│       ├── dataloader.py
│       ├── classification_model.py
│       ├── depth_model.py
│       ├── train.py
│       ├── inference.py
│       └── utils.py
│
└── frontend/                   # React Frontend (Vite)
    ├── package.json            # NPM dependencies
    ├── vite.config.js          # Vite config (configured with proxy)
    ├── index.html              # HTML shell (Google Fonts included)
    └── src/
        ├── main.jsx            # React root launcher
        ├── App.jsx             # Main Dashboard UI React logic
        └── App.css             # Premium CSS styling (Dark mode, glassmorphism)
```

---

## 🛠️ Backend Boilerplate Code

### 1. [`backend/requirements.txt`](file:///d:/IDP_Project/backend/requirements.txt)
```text
flask>=2.3.0
flask-cors>=3.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=9.5.0
werkzeug>=2.3.0
```

---

### 2. [`backend/app.py`](file:///d:/IDP_Project/backend/app.py)
```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename

# Set paths to import src directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.depth_model import DepthEstimator
from src.inference import Predictor

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for local React server
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model loaders
depth_estimator = DepthEstimator()
predictor = None
try:
    predictor = Predictor()
except Exception as e:
    print(f"Warning: Predictor weights could not be loaded. Running with depth fallback. Error: {e}")

@app.route('/api/estimate', methods=['POST'])
def estimate_nutrition():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Save the file locally
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # 1. Run Structural Depth Map Estimation
        depth_map, normalized_depth_visual = depth_estimator.get_depth_map(filepath)
        
        # Calculate volume using voxel analysis
        background_val = np.min(depth_map)
        foreground_depth = depth_map - background_val
        foreground_depth[foreground_depth < 0.0] = 0.0
        # Reference pixel area conversion metric
        estimated_volume = float(np.sum(foreground_depth) * 0.01)
        
        # 2. Run Nutritional Regression Model
        if predictor is not None:
            calories, weight_g = predictor.predict_calories_and_mass(filepath)
        else:
            # Fallback estimation values using density mapping
            weight_g = round(estimated_volume * 0.95, 1)
            calories = round(weight_g * 1.6, 1)
            
        # Convert visual depth output to Base64 to serve React image source
        _, buffer = cv2.imencode('.png', normalized_depth_visual)
        depth_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Cleanup file
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'success': True,
            'volume_cm3': round(estimated_volume, 1),
            'weight_g': round(weight_g, 1),
            'calories_kcal': round(calories, 1),
            'depth_map': f"data:image/png;base64,{depth_base64}"
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
```

---

### 3. [`backend/src/config.py`](file:///d:/IDP_Project/backend/src/config.py)
```python
import os

# Backend root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset directories
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGERY_DIR = os.path.join(DATA_DIR, "imagery")

# Metadata file (Local Kaggle file in backend root)
DISH_INGREDIENTS_CSV = os.path.join(BASE_DIR, "dish_ingredients.csv")

# Models weights dir
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classification", "best_classifier.pth")

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 15
```

---

### 4. [`backend/src/dataloader.py`](file:///d:/IDP_Project/backend/src/dataloader.py)
```python
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from src import config

class NutritionDataset(Dataset):
    def __init__(self, metadata_path, imagery_dir, transform=None):
        self.imagery_dir = imagery_dir
        self.transform = transform
        
        # Check if CSV path exists
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")

        # Load the CSV file
        raw_df = pd.read_csv(metadata_path)
        
        # Check if dataset contains ingredient-level records (requires aggregation)
        if 'ingr_name' in raw_df.columns:
            # Group by dish_id to get total calories and mass (grams)
            self.df = raw_df.groupby('dish_id').agg({
                'calories': 'sum',
                'grams': 'sum'
            }).reset_index()
        else:
            self.df = raw_df
            
        # Filter valid records with available images
        self.valid_indices = []
        for idx in range(len(self.df)):
            dish_id = self.df.iloc[idx, 0]
            img_path = os.path.join(self.imagery_dir, f"{dish_id}.png")
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
                
        print(f"Loaded {len(self.valid_indices)} valid records out of {len(self.df)} total dishes.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]
        
        # Target column parsing
        dish_id = row[0]
        calories = float(row[1]) if isinstance(row[1], (int, float)) else float(row['calories'])
        mass = float(row[2]) if isinstance(row[2], (int, float)) else float(row['grams'])
        
        img_path = os.path.join(self.imagery_dir, f"{dish_id}.png")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        targets = torch.tensor([calories, mass], dtype=torch.float32)
        return image, targets
```

---

### 5. [`backend/src/classification_model.py`](file:///d:/IDP_Project/backend/src/classification_model.py)
```python
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
```

---

### 6. [`backend/src/depth_model.py`](file:///d:/IDP_Project/backend/src/depth_model.py)
```python
import cv2
import torch
import numpy as np

class DepthEstimator:
    """
    Uses the pre-trained MiDaS model to estimate structural depth from a single image.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def get_depth_map(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        # Normalize to 0-255 range for visualization
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_map, normalized_depth
```

---

### 7. [`backend/src/utils.py`](file:///d:/IDP_Project/backend/src/utils.py)
```python
import numpy as np

def estimate_volume_from_depth(depth_map, reference_pixel_area=0.01):
    """
    Estimates food volume from a depth map using structural voxel estimation.
    Args:
        depth_map (np.ndarray): Output from depth model.
        reference_pixel_area (float): Approximate real-world area of one pixel (in cm^2).
    """
    # Background thresholding
    background_val = np.min(depth_map)
    foreground_depth = depth_map - background_val
    foreground_depth[foreground_depth < 0.0] = 0.0
    
    # Volume calculation = Area * Depth Sum
    estimated_volume = np.sum(foreground_depth) * reference_pixel_area
    return round(estimated_volume, 2)

def calculate_calories(food_type, weight_grams, calorie_database):
    """
    Looks up density in standard database and maps weight to calories.
    """
    density_key = food_type.lower()
    if density_key in calorie_database:
        cal_per_gram = calorie_database[density_key]
        return round(cal_per_gram * weight_grams, 2)
    return 0.0
```

---

### 8. [`backend/src/train.py`](file:///d:/IDP_Project/backend/src/train.py)
```python
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
```

---

### 9. [`backend/src/inference.py`](file:///d:/IDP_Project/backend/src/inference.py)
```python
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
```

---

## 🎨 React Frontend Boilerplate Code

### 1. [`frontend/package.json`](file:///d:/IDP_Project/frontend/package.json)
```json
{
  "name": "food-calorie-estimator-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/vite-plugin-react": "^4.0.0",
    "vite": "^4.3.0"
  }
}
```

---

### 2. [`frontend/vite.config.js`](file:///d:/IDP_Project/frontend/vite.config.js)
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/vite-plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
```

---

### 3. [`frontend/index.html`](file:///d:/IDP_Project/frontend/index.html)
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Calorie Estimator Dashboard</title>
    <!-- Premium Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

---

### 4. [`frontend/src/main.jsx`](file:///d:/IDP_Project/frontend/src/main.jsx)
```javascript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

---

### 5. [`frontend/src/App.jsx`](file:///d:/IDP_Project/frontend/src/App.jsx)
```javascript
import React, { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/api/estimate', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok && data.success) {
        setResults(data);
      } else {
        setError(data.error || 'Failed to estimate nutrition.');
      }
    } catch (err) {
      setError('Connection to backend server failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="navbar">
        <div className="logo-container">
          <span className="logo-emoji">🥗</span>
          <span className="logo-text">AuraCal</span>
        </div>
        <div className="badge">CV & ML Engine</div>
      </header>

      <main className="dashboard-grid">
        {/* Left Control Card */}
        <section className="glass-card upload-section">
          <h2>Analyze Plate</h2>
          <p className="subtitle">Upload a top-down food photo to compute structural parameters.</p>

          <form onSubmit={handleUpload} className="upload-form">
            <div className={`dropzone ${previewUrl ? 'has-preview' : ''}`}>
              {previewUrl ? (
                <img src={previewUrl} alt="Meal Preview" className="image-preview" />
              ) : (
                <div className="dropzone-text">
                  <span className="upload-icon">📷</span>
                  <p>Drag & drop or click to choose photo</p>
                </div>
              )}
              <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
            </div>

            <button type="submit" className="btn-primary" disabled={!selectedFile || loading}>
              {loading ? (
                <span className="spinner"></span>
              ) : (
                'Run Volumetric AI'
              )}
            </button>
          </form>

          {error && <div className="error-alert">{error}</div>}
        </section>

        {/* Right Dashboard Results Card */}
        <section className="glass-card results-section">
          <h2>Estimation Metrics</h2>
          {!results && !loading && (
            <div className="empty-state">
              <p>Please upload a plate photo to load metrics analysis</p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="loading-bar"></div>
              <p>Analyzing depth structure...</p>
            </div>
          )}

          {results && (
            <div className="results-content">
              {/* Metrics Grid */}
              <div className="metrics-grid">
                <div className="metric-box">
                  <span className="metric-label">Estimated Volume</span>
                  <span className="metric-value">{results.volume_cm3} <span className="metric-unit">cm³</span></span>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Estimated Weight</span>
                  <span className="metric-value">{results.weight_g} <span className="metric-unit">g</span></span>
                </div>
                <div className="metric-box highlighted">
                  <span className="metric-label">Energy Load</span>
                  <span className="metric-value">{results.calories_kcal} <span className="metric-unit">kcal</span></span>
                </div>
              </div>

              {/* Depth Visual Grid */}
              <div className="visuals-grid">
                <div>
                  <h4>Input Plate</h4>
                  <img src={previewUrl} alt="Raw Input" className="visual-img" />
                </div>
                <div>
                  <h4>Depth Estimation Output</h4>
                  <img src={results.depth_map} alt="Depth Render" className="visual-img depth-img" />
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
```

---

### 6. [`frontend/src/App.css`](file:///d:/IDP_Project/frontend/src/App.css)
```css
/* Styling Design System */
:root {
  --bg-primary: #0a0b10;
  --bg-card: rgba(255, 255, 255, 0.03);
  --border-card: rgba(255, 255, 255, 0.08);
  --accent: #2dd4bf;
  --accent-glow: rgba(45, 212, 191, 0.25);
  --text-main: #f3f4f6;
  --text-muted: #9ca3af;
  --font-family: 'Outfit', system-ui, -apple-system, sans-serif;
  --grad-primary: linear-gradient(135deg, #2dd4bf 0%, #3b82f6 100%);
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  background-color: var(--bg-primary);
  color: var(--text-main);
  font-family: var(--font-family);
  min-height: 100vh;
  overflow-x: hidden;
}

.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* Navbar */
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 3rem;
  border-bottom: 1px solid var(--border-card);
  backdrop-filter: blur(8px);
}

.logo-container {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.logo-emoji {
  font-size: 1.8rem;
}

.logo-text {
  font-size: 1.4rem;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: var(--grad-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.badge {
  background: var(--border-card);
  padding: 0.4rem 0.8rem;
  border-radius: 50px;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--accent);
  border: 1px solid rgba(45, 212, 191, 0.2);
}

/* Layout Grid */
.dashboard-grid {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 2rem;
  padding: 2.5rem 3rem;
  flex-grow: 1;
}

@media (max-width: 1024px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}

/* Glass Card */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 24px;
  padding: 2.5rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.glass-card h2 {
  font-size: 1.8rem;
  font-weight: 600;
}

.subtitle {
  color: var(--text-muted);
  font-size: 0.95rem;
}

/* File Upload Component */
.upload-form {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  flex-grow: 1;
}

.dropzone {
  border: 2px dashed var(--border-card);
  border-radius: 16px;
  height: 320px;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  cursor: pointer;
  overflow: hidden;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.dropzone:hover {
  border-color: var(--accent);
  box-shadow: 0 0 15px var(--accent-glow);
}

.dropzone.has-preview {
  border-style: solid;
  border-color: var(--border-card);
}

.file-input {
  position: absolute;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.dropzone-text {
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  color: var(--text-muted);
}

.upload-icon {
  font-size: 2.5rem;
}

.image-preview {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* Button & Spinner */
.btn-primary {
  background: var(--grad-primary);
  color: white;
  border: none;
  padding: 1rem;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 52px;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Error Alert */
.error-alert {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.2);
  padding: 1rem;
  border-radius: 12px;
  font-size: 0.9rem;
}

/* Metrics Dashboard Display */
.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  flex-grow: 1;
  color: var(--text-muted);
  border: 1px dashed var(--border-card);
  border-radius: 16px;
  font-size: 0.95rem;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex-grow: 1;
  gap: 1rem;
  color: var(--text-muted);
}

.loading-bar {
  width: 200px;
  height: 4px;
  background: var(--border-card);
  border-radius: 10px;
  overflow: hidden;
  position: relative;
}

.loading-bar::after {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  width: 50%;
  background: var(--grad-primary);
  animation: loading 1.5s ease infinite;
}

@keyframes loading {
  0% { left: -50%; }
  100% { left: 100%; }
}

/* Results Content */
.results-content {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  flex-grow: 1;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.25rem;
}

.metric-box {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-card);
  padding: 1.5rem;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.metric-box.highlighted {
  border-color: var(--accent);
  box-shadow: 0 0 15px var(--accent-glow);
}

.metric-label {
  font-size: 0.85rem;
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-value {
  font-size: 2.2rem;
  font-weight: 700;
}

.metric-unit {
  font-size: 1rem;
  color: var(--text-muted);
  font-weight: 400;
}

/* Images visual comparison split */
.visuals-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

.visuals-grid h4 {
  font-size: 0.95rem;
  margin-bottom: 0.75rem;
  color: var(--text-muted);
  font-weight: 600;
}

.visual-img {
  width: 100%;
  height: 250px;
  object-fit: cover;
  border-radius: 12px;
  border: 1px solid var(--border-card);
}

.depth-img {
  filter: hue-rotate(90deg) saturate(1.5);
}
```
