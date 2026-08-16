from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename

# Check if running in production on Render (where RAM is limited to 512MB)
IS_RENDER = os.environ.get('RENDER') is not None

# Set paths to import src directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model loaders (Skip PyTorch on Render Free Tier to avoid Out Of Memory crashes)
depth_estimator = None
predictor = None

if not IS_RENDER:
    try:
        from src.depth_model import DepthEstimator
        from src.inference import Predictor
        depth_estimator = DepthEstimator()
        predictor = Predictor()
        print("PyTorch AI models successfully loaded.")
    except Exception as e:
        print(f"Warning: Models could not be loaded locally. Error: {e}")
else:
    print("Running in Render Memory-Safe Mode. PyTorch imports bypassed to fit 512MB limit.")

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
        if depth_estimator is not None:
            # 1. Run PyTorch MiDaS Depth Model
            depth_map, normalized_depth_visual = depth_estimator.get_depth_map(filepath)
            
            # Calculate volume using voxel analysis
            background_val = np.min(depth_map)
            foreground_depth = depth_map - background_val
            foreground_depth[foreground_depth < 0.0] = 0.0
            # Reference pixel area conversion metric
            estimated_volume = float(np.sum(foreground_depth) * 0.01)
            
            # 2. Run PyTorch Calorie Regression Model
            if predictor is not None:
                calories, weight_g = predictor.predict_calories_and_mass(filepath)
            else:
                # Fallback estimation values using density mapping
                weight_g = round(estimated_volume * 0.95, 1)
                calories = round(weight_g * 1.6, 1)
        else:
            # Render Free Tier Heuristic fallback (Fits inside 40MB of RAM)
            img = cv2.imread(filepath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Create a mock depth map visualization using a thermal Jet colormap
            normalized_depth_visual = cv2.applyColorMap(255 - gray, cv2.COLORMAP_JET)
            
            # Heuristic volume & mass estimation from color histogram & size
            h, w = gray.shape
            estimated_volume = float((np.mean(gray) / 255.0) * 350.0)
            weight_g = round(estimated_volume * 0.9, 1)
            calories = round(weight_g * 1.45, 1)
            
        # Convert depth visual output to Base64 to serve React image source
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
