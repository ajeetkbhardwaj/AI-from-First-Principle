import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "llmProject"



list_of_files = [
    "README.md",
    f"src/{project_name}/app.py",
    "requirements.txt",
    "setup.py",
    "research/lab1-attension.ipynb",
    "research/lab2-encoding.ipynb",
    "research/lab3-tokenization.ipynb",
    "research/lab4-feedforward-neural-network.ipynb",
    "research/lab5-layer-normalization.ipynb",
    "research/lab6-encoder-architecture.ipynb",
    "research/lab7-decoder-architecture.ipynb",
    "research/lab8-transformer-architecture.ipynb",

]



for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")