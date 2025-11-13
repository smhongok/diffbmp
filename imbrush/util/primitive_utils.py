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
        """Expand a single item (string or non-string)."""
        if not isinstance(item, str):
            return [item]
        
        # Check if contains wildcard characters
        if not any(ch in item for ch in ["*", "?", "["]):
            return [item]
        
        # Determine base directory based on extension
        ext = os.path.splitext(item)[1].lower()
        if ext == ".svg":
            base_dir = os.path.join("assets", "svg")
        elif ext in (".png", ".jpg", ".jpeg"):
            base_dir = os.path.join("assets", "primitives")
        else:
            return [item]
        
        # Expand glob pattern
        full_pattern = os.path.join(base_dir, item)
        # Use recursive=True if pattern contains **
        recursive = "**" in item
        matches = sorted(glob.glob(full_pattern, recursive=recursive))
        
        # Convert back to relative paths
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
    
    # Handle string input
    elif isinstance(primitive_file_config, str):
        expanded = _expand_single(primitive_file_config)
        # Return list if expanded, otherwise keep as string
        if len(expanded) > 1:
            return expanded
        elif len(expanded) == 1 and expanded[0] != primitive_file_config:
            return expanded
        else:
            return primitive_file_config
    
    return primitive_file_config
