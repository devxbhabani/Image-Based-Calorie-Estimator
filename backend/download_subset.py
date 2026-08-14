import os
import urllib.request
import pandas as pd

# Paths
CSV_PATH = "./dish_ingredients.csv"
IMAGES_DIR = "./data/imagery"
BASE_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
        print("Success!")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    
    os.makedirs(IMAGES_DIR, exist_ok=True)

    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found in the current directory.")
        print("Please place the downloaded 'dish_ingredients.csv' in the same folder as this script.")
        return

    
    try:
        df = pd.read_csv(CSV_PATH)
        if 'dish_id' not in df.columns:
            print("Error: Column 'dish_id' not found in CSV.")
            return
            
        unique_dishes = df['dish_id'].unique().tolist()
        print(f"Total unique dishes found in dataset: {len(unique_dishes)}")
        
        #Download a small subset of images (e.g., 50 images)
        subset_limit = 50
        print(f"Downloading a subset of {subset_limit} overhead RGB images...")
        
        success_count = 0
        for dish_id in unique_dishes:
            if success_count >= subset_limit:
                break
                
            # GCS folder format: realsense_overhead/dish_[id]/rgb.png
            img_url = f"{BASE_URL}/{dish_id}/rgb.png"
            img_dest = os.path.join(IMAGES_DIR, f"{dish_id}.png")
            
            if not os.path.exists(img_dest):
                download_file(img_url, img_dest)
                if os.path.exists(img_dest):
                    success_count += 1
            else:
                # File already exists
                success_count += 1
                
        print(f"\nSubset download completed! {success_count} images are ready in '{IMAGES_DIR}'.")
        
    except Exception as e:
        print(f"Error processing CSV or downloading images: {e}")

if __name__ == "__main__":
    main()
