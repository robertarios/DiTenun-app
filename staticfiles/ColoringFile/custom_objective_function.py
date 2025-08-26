import numpy as np

def calculate_user_color_preferences(image_hsv):
    user_preferences = [[0, 0, 0], [348, 74, 73], [45, 100, 69]]
    
    hue_img = image_hsv[:, :, 0].astype(np.int32)
    sat_img = image_hsv[:, :, 1]
    val_img = image_hsv[:, :, 2]
    
    hue_scaled = (hue_img * 2).astype(np.int32)
    
    unique_colors = set()
    height, width = image_hsv.shape[:2]
    
    for i in range(height):
        for j in range(width):
            h = hue_scaled[i, j]
            s = sat_img[i, j]
            v = val_img[i, j]
            unique_colors.add((h, s, v))
    
    extracted_colors = sorted(list(unique_colors))
    
    if len(extracted_colors) != len(user_preferences):
        return 0.0
    
    for i in range(len(extracted_colors)):
        h_ext, s_ext, v_ext = extracted_colors[i]
        h_pref, s_pref, v_pref = user_preferences[i]
        
        h_diff = min(abs(h_ext - h_pref), 360 - abs(h_ext - h_pref))
        s_diff = abs(s_ext - s_pref)
        v_diff = abs(v_ext - v_pref)
        
        if h_diff > 10 or s_diff > 25 or v_diff > 25:
            return 0.0
    
    return 1.0