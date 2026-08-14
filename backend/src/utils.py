import numpy as np

def estimate_volume_from_depth(depth_map, reference_pixel_area=0.01):

    """
    Estimates food volume from a depth map using structural voxel estimation.
    Args:
        depth_map (np.ndarray): Output from depth model.
        reference_pixel_area (float): Approximate real-world area of one pixel (in cm^2).
    """

    background_val = np.min(depth_map)
    foreground_depth = depth_map - background_val
    foreground_depth[foreground_depth < 0.0] = 0.0
    
    # Volume calculation = Area * Depth Sum
    estimated_volume = np.sum(foreground_depth) * reference_pixel_area
    return round(estimated_volume, 2)

def calculate_calories(food_type, weight_grams, calorie_database):
    
    density_key = food_type.lower()
    if density_key in calorie_database:
        cal_per_gram = calorie_database[density_key]
        return round(cal_per_gram * weight_grams, 2)
    return 0.0