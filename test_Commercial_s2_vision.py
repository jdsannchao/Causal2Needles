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
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import logging
import random
random.seed(42)
import re
import base64
from io import BytesIO

# for API
import google.generativeai as genai
from google.api_core import exceptions
import anthropic

import argparse

parser = argparse.ArgumentParser(description="A command-line example with early argument access.")
parser.add_argument('model_id', type=str, help='gemini-pro-1.5/claude-sonnet/gpt4o....')
parser.add_argument('Gemini_api_key', type=str, help='API key for gemini-pro-1.5/flash-001....')
parser.add_argument('dataset', type=str, help='yms/symon')
parser.add_argument('video_path', type=str, help='videos/')
parser.add_argument('prompt_path', type=str, help='prompts/')
parser.add_argument('questions_path', type=str, help='questions/')
parser.add_argument('expansion', type=int, default=5, help='extra padding scenes')
parser.add_argument('output_dir', type=str, help='experiments/')

parser.add_argument('--DEBUG', type=int, default=0, help='test on 2 movies')

args = parser.parse_args()



def extract_abcd(input_string):
    """
    """
    matches = re.findall(r'\b[A-D]\b|\([A-D]\)|[A-D]\.', input_string)
    cleaned_matches = [re.sub(r'[\(\)\.]', '', match) for match in matches]

    if cleaned_matches:
        return cleaned_matches[0]
    else:
        return None

def contains_single_digit(string, digit):
    pattern = fr'\b{digit}\b'
    return bool(re.search(pattern, string))


def create_folder_if_not_exists(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print(f"folder '{folder_path}' exist .")

def get_logger(root_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)  
    handler = logging.FileHandler(root_dir + "log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def encode_image(image):
    ndarray_rgb = image[:, :, ::-1]
    image = Image.fromarray(ndarray_rgb)
    return image


def get_image_from_text(text_id, anns, video_cap, new_size=1024):
    begin_time = anns[text_id]['begin_time'] + 0.5
    end_time = anns[text_id]['end_time'] - 0.5
    middle_time = begin_time + (end_time - begin_time) / 2
    mid_left = begin_time + (middle_time - begin_time) / 2
    mid_right = middle_time + (end_time - middle_time) / 2
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    frames = []
    times = [begin_time, mid_left, middle_time, mid_right, end_time]
    for t in times:
        frame_number = int(t * fps)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video_cap.read()
        if ret:
            frame = cv2.resize(frame, (int(new_size/2), int(new_size/5)), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        else:
            print(f'cannot read time {t} frame ')
            return None

    concatenated_image = cv2.vconcat(frames)
    return concatenated_image


def generate_answer_gemini(images, context, question, prompt, args):
    full_content = images
    text_content = prompt.replace('<CONTEXT>', context)
    text_content = text_content.replace('<QUESTION>', question)
    text_content += f"\n\n\Answer:\n"

    if args.DEBUG:
        print(text_content)

    if images:
        full_content = images
        full_content.insert(0, text_content)
    else:
        full_content=text_content

    
    model = genai.GenerativeModel(model_name=args.model_id) # "gemini-1.5-pro-002", "gemini-1.5-flash-002"

    max_retries = 5  
    retry_count = 0  

    while retry_count < max_retries:
        try:
            response = model.generate_content(full_content,     
                                        generation_config=genai.types.GenerationConfig(
                                                                            # Only one candidate for now.
                                                                            candidate_count=1,
                                                                            temperature=0,
                                                                            max_output_tokens=100
                                                                        )
            )
            response = response.text
            break
        
        except exceptions.ResourceExhausted as e:
            retry_count += 1
            print(f"Error: {e} try after 10s... ({retry_count}/{max_retries})")
            time.sleep(10)
        except ValueError:
            retry_count += 1
            model = genai.GenerativeModel(model_name='gemini-2.0-flash-001') 
            print(f"Error: ValueError, change to flash-001, try after 10s... ({retry_count}/{max_retries})")
            time.sleep(10)        

    else:
        print("Reach Maximum trials, question skipped")
        return None
    return response






def main(args):


    if 'gemini' in args.model_id:
        genai.configure(api_key=args.Gemini_api_key)


    logger = get_logger(args.output_dir)
    logger.info(f"Arguments: {vars(args)}")


    anns_all = json.load(open('./dataset/annotations.json', 'r'))
    if args.dataset =='symon':
        args.dataset = 'en'
    dataset_anns = anns_all[args.dataset]
    movies = dataset_anns.keys()

    with open(args.prompt_path, "r", encoding="utf-8") as prompt_file:
        prompt = prompt_file.read()
    
    logger.info(f"Prompt:\n{prompt}")

    counter_all = 0  # number of all questions
    counter_correct= 0


    if args.DEBUG:
        if args.dataset=='yms':
            movies=['0002', '0013']
        elif args.dataset=='en':
            movies=['__yID2Chs7s','_fHuuL01ikc']


    for movie_id in tqdm(movies):

        if os.path.exists(args.output_dir + f'{movie_id}_answer.json'):
            continue

        anns = dataset_anns[movie_id]['annotations']
        max_idx = len(anns)-1

        video_path = args.video_path + movie_id + '.mp4'
        video_cap = cv2.VideoCapture(video_path)
        if video_cap is None:
            print('read video fails')
        logger.info(f"movie_id: {movie_id}")


        with open(args.questions_path + f'{movie_id}.json', 'r', encoding='utf-8') as file:
            questions = json.load(file)            

        for question, values in questions.items():
            cause_idx = values[0][0]
            effect_idx = values[0][1]
            full_texts = dataset_anns[movie_id]['texts']
            full_texts[cause_idx] = values[1][0]   
            full_texts[effect_idx] = values[1][1]  

            v_question = values[2][0] 
            correct_answer = values[2][1] 

            context = ' '.join(s.strip() for s in full_texts)

            left_expansion = min(random.randint(0, args.expansion), cause_idx)
            right_end = min (effect_idx + args.expansion - left_expansion, max_idx)

            if 'blind' in args.prompt_path:
                images=None
            else:
                images = [encode_image(get_image_from_text(idx, anns, video_cap)) for idx in
                        range(cause_idx-left_expansion, right_end + 1)]

            if 'gemini' in args.model_id:
                answer = generate_answer_gemini(images, context, v_question, prompt, args)

            logger.info(f"Question: {v_question}")
            logger.info(f"Correct Answer: {correct_answer}")
            logger.info(f"LLM_Answer: {answer}")
            counter_all += 1


            if answer:
                answer = answer.rstrip("\n")

                result = extract_abcd(answer)

                if result==correct_answer:
                    counter_correct += 1
                    logger.info(f"Correct!\n")
                    correct_or_not = 'correct'
                else:
                    logger.info(f"Wrong!\n")
                    correct_or_not = 'wrong'
                    

                questions[question]=[cause_idx, effect_idx, cause_idx-left_expansion, right_end, correct_or_not]

        with open(args.output_dir + f'{movie_id}_answer.json', 'w', encoding='utf-8') as save_json_file:
            json.dump(questions, save_json_file, indent=4)

    logger.info(f"num_all: {counter_all}   num_correct_effect: {counter_correct}   precision: {counter_correct/counter_all}")


if __name__ == "__main__":
    create_folder_if_not_exists(args.output_dir)
    main(args)
