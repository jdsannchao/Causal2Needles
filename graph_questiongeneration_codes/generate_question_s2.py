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

    handler = logging.FileHandler(root_dir + "log.txt")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def generate_question(prompt, rephrase):
    prompt += f"Cause: {rephrase[0]}\n"
    prompt += f"Effect: {rephrase[1]}\n"
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

    output_text = response.choices[0].message.content
    lines = output_text.strip().split('\n')
    last_line = lines[-1]
    if last_line.startswith("Output: "):
        question = last_line[len("Output: "):].strip()
        return question, lines[2]
    else:
        return None

def main(causes_dir, rephrase_dir, output_dir):
    logger = get_logger(output_dir)

    anns_all = json.load(open('/home/limiaoyu/M-SyMoN/AnnotatedSet/annotations.json', 'r'))

    symon_anns = anns_all['en']
    movies = symon_anns.keys()

    with open("", "r", encoding="utf-8") as prompt_file:
        prompt = prompt_file.read()

    for movie_id in tqdm(movies):
        logger.info(f"movie_id: {movie_id}\n")
        questions = {}
        with open(causes_dir + f'{movie_id}.json', 'r', encoding='utf-8') as file:
            causes = json.load(file)
        with open(rephrase_dir + f'{movie_id}.json', 'r', encoding='utf-8') as file:
            rephrases = json.load(file)

        for cause, rephrase in zip(causes, rephrases):
            question, bridge_entity = generate_question(prompt, rephrase)
            if not question:
                logger.info(f"Cause: {rephrase[0]}\nEffect: {rephrase[1]}\nQuestion: ERROR\n")
            else:
                questions[question] = [cause, rephrase]
                logger.info(f"Cause: {rephrase[0]}\nEffect: {rephrase[1]}\nBridge: {bridge_entity}\nQuestion: {question}\n")
        with open(output_dir + f'{movie_id}.json', 'w', encoding='utf-8') as json_file:
            json.dump(questions, json_file)



if __name__ == "__main__":
    causes_dir = ''
    rephrase_dir = ''
    output_dir = ''
    create_folder_if_not_exists(output_dir)
    main(causes_dir, rephrase_dir, output_dir)
