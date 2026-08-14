# 🥗 AuraCal: Image-Based Food Calorie & Nutrient Estimator

AuraCal is a full-stack, machine-learning-powered application designed to estimate the mass, volume, and caloric value of meals from a single overhead food photograph. 

By combining monocular depth estimation (to compute 3D volumetric parameters) with deep learning regression models (to predict density and calories), AuraCal bridges the gap between visual food recognition and precise diet tracking.

---

## ⚡ Key Features

*   **📐 Volumetric 3D Estimation:** Integrates the pre-trained **MiDaS** depth model to calculate structural plate dimensions from a flat 2D image.
*   **🧠 Deep Learning Calorie Regression:** Uses a modified **ResNet-50** regression network to predict food mass (grams) and calorie count (kcal) directly from visual inputs.
*   **📊 Integrated Local Dataset:** Pre-configured to ingest and aggregate the **Nutrition5k** dataset (dish_ingredients.csv), converting individual ingredient lines into total dish-level targets.
*   **👾 Cyberpunk Neo-Brutalist UI:** A modern web interface built in **React (Vite)** featuring high-impact flat colors, stark black borders, monospace layouts, and live depth map comparisons.

---

## 📌 Project Architecture

`mermaid
graph TD
    A[React Client UI] -->|Upload Image| B[Flask API Server]
    B --> C[Depth Estimator Module]
    B --> D[ML Predictor Module]
    C -->|Estimated 3D Voxel Volume| E[Response Aggregator]
    D -->|Predicted Calories & Mass| E
    E -->|JSON Response with Base64 Depth Map| A
`

---

## 📂 Project Structure

`	ext
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
│       ├── config.py           # Paths and hyperparameters
│       ├── dataloader.py       # Custom PyTorch Dataset (Nutrition5k parser)
│       ├── classification_model.py # ResNet50 Regression architecture
│       ├── depth_model.py      # MiDaS wrapper
│       ├── train.py            # Model training script
│       ├── inference.py        # Model prediction script
│       └── utils.py            # Geometric & calorie math helpers
│
└── frontend/                   # React Frontend (Vite)
    ├── package.json            # Frontend package config
    ├── vite.config.js          # Vite config (pre-configured proxy to port 5000)
    ├── index.html              # HTML launcher shell
    └── src/
        ├── main.jsx            # React root script
        ├── App.jsx             # Main dashboard logic
        └── App.css             # Cyberpunk Neo-Brutalist stylesheet
`

---

## 🏃 Step-by-Step Running Guide

### **Step 1: Place Your Dataset**
Place your Kaggle-downloaded dish_ingredients.csv file inside the ackend/ directory.

### **Step 2: Setup the Python Backend**
Open a terminal in the ackend/ directory:
`ash
cd backend

# 1. Create a virtual environment
python -m venv env

# 2. Activate the virtual environment
# Windows (PowerShell):
.\env\Scripts\Activate.ps1
# Windows (CMD):
.\env\Scripts\activate.bat
# Linux/macOS:
source env/bin/activate

# 3. Install packages
pip install -r requirements.txt

# 4. Download dataset visual imagery subset (Fetches 50 images to start)
python download_subset.py

# 5. Train the Regression Model (Generates best_classifier.pth)
python -m src.train

# 6. Start the Flask API
python app.py
`
*The backend API server will run on http://127.0.0.1:5000.*

### **Step 3: Setup the React Frontend**
Open a **new** terminal window in the rontend/Food-Calorie-Estimator/ directory:
`ash
cd frontend/Food-Calorie-Estimator

# 1. Install Node modules
npm install

# 2. Start the Vite server
npm run dev
`
*The UI dashboard will run on http://localhost:5173. Open this URL in your web browser to upload images and analyze meals.*

---

## 🛠️ Tech Stack

*   **Frontend:** React, Vite, CSS Grid/Flexbox, Outfit/Space Mono Fonts.
*   **Backend:** Flask, Flask-CORS, Werkzeug.
*   **Deep Learning & Vision:** PyTorch, Torchvision, OpenCV, Timm, NumPy, Pandas.
*   **Data Models:** ResNet50 (pre-trained feature extractor), MiDaS Small (depth estimation model).
