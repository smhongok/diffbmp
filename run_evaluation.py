import subprocess
import re
import pandas as pd
from datetime import datetime
import os

def run_and_capture_excel_paths(commands):
    """
    Runs a list of shell commands, captures stdout, and extracts Excel file paths.
    """
    excel_paths = []
    for cmd in commands:
        # Execute the command, capture stdout
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(result.stdout)
        # Extract lines containing .xlsx file paths
        for line in reversed(result.stdout.splitlines()):
            match = re.search(r'(/[^ ]+\.xlsx)', line)
            if match:
                excel_paths.append('outputs' + match.group(1))
                break
    return excel_paths

def merge_excels(excel_paths, output_path):
    """
    Reads all Excel files in excel_paths, concatenates them, and saves to output_path.
    """
    dfs = [pd.read_excel(path) for path in excel_paths]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_excel(output_path, index=False)
    return combined

if __name__ == "__main__":
    # List your commands here
    base_command = "python compare_methods.py --config configs/default.json"
    commands = []
    img_paths = [
        "images/CelebA/182340.png",
        "images/CelebA/182341.png",
        "images/CelebA/182342.png"
    ]
    initializers = [
        "StructureAwareInitializer"
    ]
    renderers = [
        "MseRenderer",
    ]
    svg_texts = [
        "A",
        "B",
        "M",
        "X",
        "Y"
    ]

    #svg_texts = [""] # for svg mode
    #svg_paths = ["siggraph_logo.svg"]
    
    for svg_text in svg_texts:
        for svg_path in svg_paths:
            for initializer in initializers:
                for renderer in renderers:
                    for img_path in img_paths:
                        #commands.append(f"{base_command} --initializer {initializer} --renderer {renderer} --svg_path {svg_path} --img_path {img_path}")
                        commands.append(f"{base_command} --initializer {initializer} --renderer {renderer} --svg_text {svg_text} --img_path {img_path}")
    
    # Execute commands and collect generated Excel file paths
    excel_paths = run_and_capture_excel_paths(commands)
    print("Found Excel files:", excel_paths)
    
    # Merge collected Excel files into one
    os.makedirs("outputs/merged_results", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = "outputs/merged_results/" + timestamp + ".xlsx"
    merged_df = merge_excels(excel_paths, output_file)
    print(f"Merged Excel saved to: {output_file}")