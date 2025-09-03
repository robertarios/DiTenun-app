import numpy as np

def calculate_user_color_preferences(image_hsv):
    user_preferences = np.array([[331, 97, 92], [56, 100, 100], [138, 100, 93], [205, 60, 100]], dtype=np.int32)
    
    h, s, v = image_hsv[:,:,0], image_hsv[:,:,1], image_hsv[:,:,2]
    hue_img = h.astype(np.int32) * 2
    sat_img = s.astype(np.int32) * 100 // 255
    val_img = v.astype(np.int32) * 100 // 255
    
    unique_colors = set()
    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            h_val = hue_img[i,j] % 360
            s_val = sat_img[i,j]
            v_val = val_img[i,j]
            unique_colors.add((h_val, s_val, v_val))
    
    extracted_colors = np.array(list(unique_colors), dtype=np.int32)
    
    if len(extracted_colors) != len(user_preferences):
        return 0.0
    
    sorted_extracted = np.sort(extracted_colors, axis=0)
    sorted_user = np.sort(user_preferences, axis=0)
    
    if np.array_equal(sorted_extracted, sorted_user):
        return 1.0
    
    hue_diff = np.abs(sorted_extracted[:,0] - sorted_user[:,0])
    sat_diff = np.abs(sorted_extracted[:,1] - sorted_user[:,1])
    val_diff = np.abs(sorted_extracted[:,2] - sorted_user[:,2])
    
    hue_penalty = np.sum(np.minimum(hue_diff, 360 - hue_diff)) / 360.0
    sat_penalty = np.sum(sat_diff) / 400.0
    val_penalty = np.sum(val_diff) / 400.0
    
    total_penalty = (hue_penalty + sat_penalty + val_penalty) / 3.0
    fitness = max(0.0, 1.0 - total_penalty)
    
    return fitness