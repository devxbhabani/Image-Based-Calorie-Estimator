import cv2
import torch
import numpy

class DepthEstimator:
    """
    Uses the pre-trained MiDaS model to estimate structural depth from a single image.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
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