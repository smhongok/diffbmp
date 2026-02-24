"""Utility functions for primitive file handling."""
import os
import glob


def expand_primitive_wildcards(primitive_file_config):
    """
    Expand wildcard patterns in primitive_file config.
    
    Args:
        primitive_file_config: str or list of strings, may contain wildcards like "CelebA/*.png"
    
    Returns:
        Expanded config (str or list) with wildcards resolved to actual filenames
    """
    def _expand_single(item):
        if not isinstance(item, str):
            return [item]
        
        if not any(ch in item for ch in ["*", "?", "["]):
            return [item]
        
        # Determine base directory based on extension

        IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
        SVG_EXTS = (".svg",)
        
        ext = os.path.splitext(item)[1].lower()
        
        if ext in IMAGE_EXTS or ext == "": 
            base_dir = os.path.join("assets", "primitives")
            target_exts = IMAGE_EXTS
        elif ext in SVG_EXTS:
            base_dir = os.path.join("assets", "svg")
            target_exts = SVG_EXTS
        else:
            return [item]
        
        # Use recursive=True if pattern contains **
        matches = []
        recursive = "**" in item
        
        search_patterns = [item] if ext != "" else [item + e for e in target_exts]
        
        for pattern in search_patterns:
            full_pattern = os.path.join(base_dir, pattern)
            matches.extend(glob.glob(full_pattern, recursive=recursive))
        
        matches = sorted(list(set(matches))) 
        
        rels = []
        prefix = base_dir + os.sep
        for p in matches:
            if p.startswith(prefix):
                rels.append(p[len(prefix):])
            else:
                rels.append(p)
        
        return rels if rels else [item]
    
    # Handle list input
    if isinstance(primitive_file_config, list):
        expanded_list = []
        for it in primitive_file_config:
            expanded_list.extend(_expand_single(it))
        return expanded_list
    elif isinstance(primitive_file_config, str):
        expanded = _expand_single(primitive_file_config)
        return expanded if len(expanded) > 1 or (len(expanded) == 1 and expanded[0] != primitive_file_config) else primitive_file_config
    
    return primitive_file_config
