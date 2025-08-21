import os
from typing import List, Dict

from qwen_agent.agents import Assistant

from qwen_agent.gui import WebUI

def get_files_from_folder(folder_path: str) -> List [str]:
    supporter_extensions = ['.pdf', '.docx', '.pptx', '.txt', '.csv', '.tsv', '.xlsx', '.xls', '.html']
    files = []

    # 转换为绝对路径
    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist.")
    
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in supporter_extensions:
                file_path = os.path.join(root, filename)
                file_path = os.path.abspath(file_path)
                files.append(file_path)
    return files

