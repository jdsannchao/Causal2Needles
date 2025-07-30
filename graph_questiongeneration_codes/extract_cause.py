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

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


client = OpenAI(api_key=api_key)
class Edge(BaseModel):
    start_node: int
    end_node: int

class Event_Graph(BaseModel):
    characters: List[Edge]
def Extract_Graph_Local(prompt, window):
    nodes_prompt = "\n\nNow, it is your turn to construct the event graph for the following event list.\nEvent List:\n"
    indices = window.keys()
    for index in indices:
        nodes_prompt += f"Node {index}: {window[index]}"
    full_prompt = prompt + nodes_prompt + "\nOutput:\n"
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": f"{full_prompt}"}
        ],
        response_format=Event_Graph,
        temperature=0,
        seed=1
    )

    response = completion.choices[0].message.parsed

    return response.characters

def Extract_Graph_Global(prompt, full_texts):
    nodes_prompt = "\n\nNow, it is your turn to construct the event graph for the following event list.\nEvent List:\n"
    index = 0
    for sent in full_texts:
        nodes_prompt += f"Node {index}: {sent}"
        index += 1
    full_prompt = prompt + nodes_prompt + "\nOutput:\n"
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": f"{full_prompt}"}
        ],
        response_format=Event_Graph,
        temperature=0,
        seed=1
    )

    response = completion.choices[0].message.parsed

    return response.characters

def get_windows(full_texts, window_length=15, stride=5):
    windows = []
    for i in range(0, len(full_texts), stride):
        if i + window_length > len(full_texts):
            subdict = {i + j: full_texts[i + j] for j in range(len(full_texts) - i)}
            windows.append(subdict)
            break
        subdict = {i + j: full_texts[i + j] for j in range(window_length)}
        windows.append(subdict)
    return windows

def get_causes(full_texts, event_graph, causes, logger):
    for edge in event_graph:
        if edge.end_node - edge.start_node > 2 and (edge.start_node, edge.end_node) not in causes:
            causes.append((edge.start_node, edge.end_node))
            logger.info(f"start_node: {full_texts[edge.start_node]}end_node: {full_texts[edge.end_node]}\n")


def main(output_dir):
    logger = get_logger(output_dir)

    anns_all = json.load(open('', 'r'))

    yms_anns = anns_all['yms']
    movies = yms_anns.keys()

    with open("", "r", encoding="utf-8") as prompt_file:
        prompt = prompt_file.read()

    for movie_id in tqdm(movies):
        logger.info(f"movie_id: {movie_id}\n")
        full_texts = anns_all['yms'][movie_id]['texts']

        windows = get_windows(full_texts)
        causes = []
        event_graph_global = Extract_Graph_Global(prompt, full_texts)
        get_causes(full_texts, event_graph_global, causes, logger)
        for window in windows:
            event_graph_local = Extract_Graph_Local(prompt, window)
            get_causes(full_texts, event_graph_local, causes, logger)
        print(causes)
        with open(output_dir + f'{movie_id}.json', 'w', encoding='utf-8') as json_file:
            json.dump(causes, json_file)

if __name__ == "__main__":
    output_dir = ''
    create_folder_if_not_exists(output_dir)
    main(output_dir)
