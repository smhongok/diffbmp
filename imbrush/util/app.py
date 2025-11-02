def half_blur(x, y, r, v, theta, c,
                      I_target,
                      target_binary_mask,
                      remove_num = 700):
    """
    Remove primitives on the right half of the image.
    Primitives with smaller radius are removed first.
    More primitives are removed as we move further right.
    
    Args:
        x, y, r, v, theta, c: Primitive parameters (numpy arrays or torch tensors)
        I_target: Target image (to get dimensions)
        target_binary_mask: Binary mask
    
    Returns:
        Filtered x, y, r, v, theta, c arrays with some primitives removed
    """
    print("Half blur function called...")
    import numpy as np
    import torch
    
    # Check if inputs are torch tensors
    is_torch = torch.is_tensor(x)
    device = x.device if is_torch else None
    
    # Convert to numpy for processing
    if is_torch:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        c_np = c.detach().cpu().numpy()
    else:
        x_np = x
        y_np = y
        r_np = r
        v_np = v
        theta_np = theta
        c_np = c
    
    # Get image width
    img_width = I_target.shape[1]
    half_width = img_width / 2 + 20
    
    # Create mask for primitives to keep
    keep_mask = np.ones(len(x_np), dtype=bool)
    
    # Process only primitives on the right half
    right_half_mask = x_np > half_width
    right_indices = np.where(right_half_mask)[0]
    
    if len(right_indices) == 0:
        # No primitives on right half, return all
        return x, y, r, v, theta, c
    
    # Get parameters for right half primitives
    right_x = x_np[right_indices]
    right_r = r_np[right_indices]
    
    # Calculate removal ratio based on distance from center
    # Distance from half_width normalized to [0, 1]
    distance_ratio = (right_x - half_width) / half_width
    distance_ratio = np.clip(distance_ratio, 0, 1)
    
    # Removal ratio: 0% at center, up to 100% at far right
    removal_ratio = distance_ratio * 1.0
    
    # Calculate removal probability for each primitive (based on radius and distance)
    removal_probs = []
    for idx, orig_idx in enumerate(right_indices):
        # Calculate radius percentile (smaller radius = lower percentile)
        radius_percentile = np.sum(right_r < right_r[idx]) / len(right_r)
        
        # Probability increases for smaller radius and larger distance
        # Base probability from distance, modified by radius
        base_prob = removal_ratio[idx]
        # Smaller radius = higher removal probability
        radius_factor = 1.0 - radius_percentile  # 0 (largest) to 1 (smallest)
        prob = base_prob * (0.3 + 0.7 * radius_factor)  # Weight radius factor
        removal_probs.append((orig_idx, prob))
    
    # Sort by probability (highest first) and remove until we reach remove_num
    removal_probs.sort(key=lambda x: x[1], reverse=True)
    
    removed_count = 0
    attempt = 0
    max_attempts = len(removal_probs) * 3  # Allow multiple passes if needed
    
    while removed_count < remove_num and attempt < max_attempts:
        for orig_idx, prob in removal_probs:
            if removed_count >= remove_num:
                break
            
            # Skip if already marked for removal
            if not keep_mask[orig_idx]:
                continue
            
            # Add randomness: use probability as threshold
            # Increase probability on subsequent attempts
            adjusted_prob = min(prob + 0.2 + (attempt * 0.1), 1.0)
            if np.random.random() < adjusted_prob:
                # Only set to False if it's currently True
                if keep_mask[orig_idx]:
                    keep_mask[orig_idx] = False
                    removed_count += 1
        
        attempt += 1
        
        # If we've gone through all candidates and still haven't removed enough,
        # break to avoid infinite loop
        if removed_count < remove_num and np.sum(keep_mask[right_indices]) == 0:
            # No more right-half primitives to remove
            break
    
    # Apply mask to filter primitives
    x_filtered = x_np[keep_mask]
    y_filtered = y_np[keep_mask]
    r_filtered = r_np[keep_mask]
    v_filtered = v_np[keep_mask]
    theta_filtered = theta_np[keep_mask]
    c_filtered = c_np[keep_mask]
    
    # Print statistics
    original_count = len(x_np)
    filtered_count = len(x_filtered)
    removed_count = original_count - filtered_count
    print(f"Half blur: Removed {removed_count} / {original_count} primitives ({removed_count/original_count*100:.1f}%)")
    
    # Convert back to torch tensors if input was torch
    if is_torch:
        x_filtered = torch.from_numpy(x_filtered).to(device)
        y_filtered = torch.from_numpy(y_filtered).to(device)
        r_filtered = torch.from_numpy(r_filtered).to(device)
        v_filtered = torch.from_numpy(v_filtered).to(device)
        theta_filtered = torch.from_numpy(theta_filtered).to(device)
        c_filtered = torch.from_numpy(c_filtered).to(device)
    
    return x_filtered, y_filtered, r_filtered, v_filtered, theta_filtered, c_filtered

def mask_blur(x, y, r, v, theta, c,
                      I_target,
                      target_binary_mask,
                      dense_mask,
                      remove_num = 700):
    """
    dense_mask : 0 ~ 255
    255 means keep all primitives on that pixel, 0 means remove all primitives on that pixel
    """
    print("Mask blur function called...")
    import numpy as np
    import torch
    
    # Check if inputs are torch tensors
    is_torch = torch.is_tensor(x)
    device = x.device if is_torch else None
    
    # Convert to numpy for processing
    if is_torch:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        c_np = c.detach().cpu().numpy()
    else:
        x_np = x
        y_np = y
        r_np = r
        v_np = v
        theta_np = theta
        c_np = c
    
    # Ensure dense_mask is numpy array
    if torch.is_tensor(dense_mask):
        dense_mask_np = dense_mask.detach().cpu().numpy()
    else:
        dense_mask_np = dense_mask
    
    # Create mask for primitives to keep
    keep_mask = np.ones(len(x_np), dtype=bool)
    
    # Get image dimensions
    img_height, img_width = I_target.shape[:2]
    
    # Calculate removal probability for each primitive
    removal_probs = []
    for i in range(len(x_np)):
        # Get primitive position (clamp to image bounds)
        px = int(np.clip(x_np[i], 0, img_width - 1))
        py = int(np.clip(y_np[i], 0, img_height - 1))
        
        # Get mask value at primitive position (0-255)
        mask_value = dense_mask_np[py, px]
        
        # Convert mask value to removal probability
        # 255 -> 0% removal, 0 -> 100% removal
        base_removal_prob = 1.0 - (mask_value / 255.0)
        
        # Calculate radius percentile (smaller radius = higher removal probability)
        radius_percentile = np.sum(r_np < r_np[i]) / len(r_np)
        radius_factor = 1.0 - radius_percentile  # 0 (largest) to 1 (smallest)
        
        # Combine mask-based probability with radius factor
        # Higher removal probability for smaller radius
        final_prob = base_removal_prob * (0.3 + 0.7 * radius_factor)
        
        removal_probs.append((i, final_prob))
    
    # Sort by probability (highest first) and remove until we reach remove_num
    removal_probs.sort(key=lambda x: x[1], reverse=True)
    
    removed_count = 0
    attempt = 0
    max_attempts = len(removal_probs) * 3  # Allow multiple passes if needed
    
    while removed_count < remove_num and attempt < max_attempts:
        for orig_idx, prob in removal_probs:
            if removed_count >= remove_num:
                break
            
            # Skip if already marked for removal
            if not keep_mask[orig_idx]:
                continue
            
            # Add randomness: use probability as threshold
            # Increase probability on subsequent attempts
            adjusted_prob = min(prob + 0.1 + (attempt * 0.05), 1.0)
            if np.random.random() < adjusted_prob:
                # Only set to False if it's currently True
                if keep_mask[orig_idx]:
                    keep_mask[orig_idx] = False
                    removed_count += 1
        
        attempt += 1
        
        # If no more primitives can be removed, break
        if removed_count < remove_num and np.sum(keep_mask) == 0:
            break
    
    # Apply mask to filter primitives
    x_filtered = x_np[keep_mask]
    y_filtered = y_np[keep_mask]
    r_filtered = r_np[keep_mask]
    v_filtered = v_np[keep_mask]
    theta_filtered = theta_np[keep_mask]
    c_filtered = c_np[keep_mask]
    
    # Print statistics
    original_count = len(x_np)
    filtered_count = len(x_filtered)
    removed_count = original_count - filtered_count
    print(f"Mask blur: Removed {removed_count} / {original_count} primitives ({removed_count/original_count*100:.1f}%)")
    
    # Convert back to torch tensors if input was torch
    if is_torch:
        x_filtered = torch.from_numpy(x_filtered).to(device)
        y_filtered = torch.from_numpy(y_filtered).to(device)
        r_filtered = torch.from_numpy(r_filtered).to(device)
        v_filtered = torch.from_numpy(v_filtered).to(device)
        theta_filtered = torch.from_numpy(theta_filtered).to(device)
        c_filtered = torch.from_numpy(c_filtered).to(device)
    
    return x_filtered, y_filtered, r_filtered, v_filtered, theta_filtered, c_filtered