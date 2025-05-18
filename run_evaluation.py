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
                excel_paths.append(os.path.join('outputs', match.group(1)))
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
    base_command = "python compare_methods.py --config configs/smhong-dev-font.json"
    commands = []
    img_paths = [
        "images/artwork/Cafe_Terrace_at_Night.jpg",
        "images/artwork/Gustav_Klimt.jpg",
        "images/artwork/The_Great_Wave_of_Kanagawa.jpg",
        "images/artwork/The_Scream.jpg",
        "images/BSDS500/102061.jpg",
        "images/MoviePosters/1.6_245943.jpg",
        "images/MoviePosters/1.9_270846.jpg",
        "images/nature/puppy1.jpg",
        "images/nature/puppy2.jpg",
        "images/nature/melon.jpg"
    ]
    initializers = [
        #"RandomInitializer",
        "StructureAwareInitializer"
    ]
    renderers = [
        "MseRenderer",
        #"FreqRenderer_1",
        #"FreqRenderer_2"
    ]
    svg_texts = [
        "B",
        "G",
        "M",
        #"S",
        #"T",
    ]
    
    for svg_text in svg_texts:
        for initializer in initializers:
            for renderer in renderers:
                for img_path in img_paths:
                    commands.append(f"{base_command} --initializer {initializer} --renderer {renderer} --svg_text {svg_text} --img_path {img_path}")
    
    # 명령 실행 및 생성된 Excel 파일 경로 수집
    excel_paths = run_and_capture_excel_paths(commands)
    print("Found Excel files:", excel_paths)
    
    # 수집된 Excel 파일을 하나로 병합
    os.makedirs("outputs/merged_results", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = "outputs/merged_results/" + timestamp + ".xlsx"
    merged_df = merge_excels(excel_paths, output_file)
    print(f"Merged Excel saved to: {output_file}")