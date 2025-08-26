import numpy as np

def calculate_user_color_preferences(image_hsv):
    user_preferences = np.array([[56, 100, 100], [132, 100, 31], [225, 61, 70], [205, 60, 100], [331, 97, 92]])
    
    hue_img = image_hsv[:, :, 0].astype(np.int32) * 2
    sat_img = image_hsv[:, :, 1].astype(np.int32)
    val_img = image_hsv[:, :, 2].astype(np.int32)
    
    unique_colors = set()
    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            h = hue_img[i, j] % 360
            s = sat_img[i, j]
            v = val_img[i, j]
            unique_colors.add((h, s, v))
    
    image_colors = np.array(list(unique_colors))
    
    if len(image_colors) != len(user_preferences):
        return 0.0
    
    sorted_image = image_colors[np.lexsort((image_colors[:, 2], image_colors[:, 1], image_colors[:, 0]))]
    sorted_user = user_preferences[np.lexsort((user_preferences[:, 2], user_preferences[:, 1], user_preferences[:, 0]))]
    
    hue_diff = np.abs(sorted_image[:, 0] - sorted_user[:, 0])
    sat_diff = np.abs(sorted_image[:, 1] - sorted_user[:, 1])
    val_diff = np.abs(sorted_image[:, 2] - sorted_user[:, 2])
    
    hue_penalty = np.sum(np.minimum(hue_diff, 360 - hue_diff)) / 360.0
    sat_penalty = np.sum(sat_diff) / 255.0
    val_penalty = np.sum(val_diff) / 255.0
    
    total_penalty = (hue_penalty + sat_penalty + val_penalty) / (3 * len(user_preferences))
    fitness = max(0.0, 1.0 - total_penalty)
    
    return fitness