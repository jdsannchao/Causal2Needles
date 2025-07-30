import os
import time
import io
import random
import statistics
import json
import pickle
import base64
from tqdm import tqdm
import requests
import numpy as np
from PIL import Image, ImageFile
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import logging
from evaluate import eval

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


api_key = ''

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def create_folder_if_not_exists(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建.")
    else:
        print(f"文件夹 '{folder_path}' 已存在.")

def get_logger(root_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(root_dir + "log2.txt")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def rephrase_cause(prompt, cause, full_texts):
    start_node, end_node = cause[0], cause[1]
    prompt += f"\nCause: {full_texts[start_node]}"
    prompt += f"Effect: {full_texts[end_node]}"
    prompt += "Output:\n"

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="o1-preview-2024-09-12",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "seed": 1
            }
        ]
    )

    print(response.choices[0].message.content)

    return response.choices[0].message.content

def main(causes_dir, output_dir):
    logger = get_logger(output_dir)

    anns_all = json.load(open('', 'r'))

    yms_anns = anns_all['yms']
    movies = yms_anns.keys()

    with open("", "r", encoding="utf-8") as prompt_file:
        prompt = prompt_file.read()

    for movie_id in tqdm(movies):
        logger.info(f"movie_id: {movie_id}\n")
        questions = {}
        full_texts = yms_anns[movie_id]['texts']
        with open(causes_dir + f'{movie_id}.json', 'r', encoding='utf-8') as file:
            causes = json.load(file)

        for cause in causes:
            new_cause = rephrase_cause(prompt, cause, full_texts)
            logger.info(f"Cause: {full_texts[cause[0]]}Effect: {full_texts[cause[1]]}New: {new_cause}\n")
        with open(output_dir + f'{movie_id}.json', 'w', encoding='utf-8') as json_file:
            json.dump(questions, json_file)



if __name__ == "__main__":
    causes_dir = ''
    output_dir = ''
    create_folder_if_not_exists(output_dir)
    main(causes_dir, output_dir)
