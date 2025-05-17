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


def extract_numbers(string):
    numbers = re.findall(r'\d+', string)
    numbers = [int(num) for num in numbers]
    return numbers

def extract_scene_numbers(string):
    part1_scene = None
    part2_scene = None

    match_part1 = re.search(r"Scene (\d+) for Part 1", string)
    match_part2 = re.search(r"Scene (\d+) for Part 2", string)

    if match_part1:
        part1_scene = int(match_part1.group(1))

    if match_part2:
        part2_scene = int(match_part2.group(1))

    return part1_scene, part2_scene

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


def generate_answer_gemini(images, context, questions, prompt, args):
    full_content = images
    text_content = prompt.replace('<CONTEXT>', context)
    text_content = text_content.replace('<PART1>', questions[0])
    text_content = text_content.replace('<PART2>', questions[1])
    text_content += f"\n\nIndex number of the scene:\n"

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

    if args.DEBUG:
        if args.dataset=='yms':
            movies=['0002', '0013']
        elif args.dataset=='en':
            movies=['__yID2Chs7s','_fHuuL01ikc']


    counter_all = 0  # number of all questions
    counter_correct_cause,  counter_correct_effect, counter_correct_both= 0, 0, 0  # number of questions with correct answer


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

        for question, nested_list in questions.items():
            cause_idx = nested_list[0][0]
            effect_idx = nested_list[0][1]
            
            # replace to rephrased sentences
            full_texts = dataset_anns[movie_id]['texts']
            full_texts[cause_idx] = nested_list[1][0]   
            full_texts[effect_idx] = nested_list[1][1]  

            s2_question = nested_list[3]

            context = ' '.join(s.strip() for s in full_texts)

            left_expansion = min(random.randint(0, args.expansion), cause_idx)
            right_end = min (effect_idx + args.expansion - left_expansion, max_idx)

            gt_answer_cause, gt_answer_effect = left_expansion+1,  effect_idx-cause_idx+left_expansion+1
            
            if 'blind' in args.prompt_path:
                print('no image')
                images =None
            else:
                images = [encode_image(get_image_from_text(idx, anns, video_cap)) for idx in
                        range(cause_idx-left_expansion, right_end + 1)]

            if 'rev' in prompt:
                images = images[::-1]
                total_length = len(images)
                gt_answer_cause, gt_answer_effect = total_length - gt_answer_cause + 1, total_length - gt_answer_effect + 1

            logger.info(
                f"Cause: ({cause_idx}) {full_texts[cause_idx]}Effect: ({effect_idx}) {full_texts[effect_idx]}"
                f"\nCause_padding: ({left_expansion})  Effect_padding: ({right_end-effect_idx})")
            

            if 'gemini' in args.model_id:
                answer = generate_answer_gemini(images, context, s2_question, prompt, args)


            counter_all += 1
            logger.info(
                    f"Question: {question}\nGT: {gt_answer_cause}, {gt_answer_effect}\nAnswer: {answer}")

            if answer:
                answer = answer.rstrip("\n")

                result = extract_scene_numbers(answer)

                if result[0]==gt_answer_effect:
                    counter_correct_effect += 1
                    logger.info(f"Correct Effect!\n")
                    correct_or_not_effect = 'correct'
                else:
                    logger.info(f"Wrong Effect!\n")
                    correct_or_not_effect = 'wrong'
                    
                if result[1]==gt_answer_cause:
                    counter_correct_cause += 1
                    logger.info(f"Correct Cause!\n")
                    correct_or_not_cause = 'correct'
                else:
                    logger.info(f"Wrong Cause!\n")
                    correct_or_not_cause = 'wrong'
                
                if correct_or_not_effect == correct_or_not_cause == 'correct':
                    counter_correct_both +=1


                questions[question]=[cause_idx, effect_idx, cause_idx-left_expansion, right_end, correct_or_not_effect, correct_or_not_cause]

        with open(args.output_dir + f'{movie_id}_answer.json', 'w', encoding='utf-8') as save_json_file:
            json.dump(questions, save_json_file, indent=4)

    logger.info(f"num_all: {counter_all}   num_correct_effect: {counter_correct_effect}   precision: {counter_correct_effect/counter_all}")
    logger.info(f"num_all: {counter_all}   num_correct_cause: {counter_correct_cause}   precision: {counter_correct_cause/counter_all}")
    logger.info(f"num_all: {counter_all}   num_correct_both: {counter_correct_both}   precision: {counter_correct_both/counter_all}")


if __name__ == "__main__":
    create_folder_if_not_exists(args.output_dir)
    main(args)
