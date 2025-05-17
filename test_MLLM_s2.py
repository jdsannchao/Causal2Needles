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

from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

import argparse

parser = argparse.ArgumentParser(description="A command-line example with early argument access.")
parser.add_argument('model_id', type=str, help='gemini-pro-1.5/claude-sonnet/gpt4o....')
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


def generate_answer(images, context, question, prompt):

    text_content = prompt + f"\n\nContext: {context}"
    text_content += f"\n\nQuestion: {question}"
    text_content += f"\n\nIndex number of the scene:\n"

    # define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video")
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": text_content},
                {"type": "video"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_video = processor(text=prompt, videos=images, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    answer = processor.decode(output[0][2:], skip_special_tokens=True)
    answer = answer.split("Index number of the scene:\n")[1]

    return answer



def main(args):

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
            

            answer = generate_answer(images, context, question, prompt)

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
