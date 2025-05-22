import os
import time
import random
import json
import re
import logging
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# import google.generativeai as genai
from google.api_core import exceptions
from google import genai
from google.genai import types

parser = argparse.ArgumentParser()
parser.add_argument('model_id', type=str)
parser.add_argument('api_key', type=str)
parser.add_argument('task', type=str)
parser.add_argument('video_path', type=str)
parser.add_argument('prompt_path', type=str)
parser.add_argument('questions_path', type=str)
parser.add_argument('expansion', type=int)
parser.add_argument('output_dir', type=str)
parser.add_argument('--DEBUG', type=int, default=0)
args = parser.parse_args()


def extract_abcd(input_string):
    matches = re.findall(r'\b[A-D]\b|\([A-D]\)|[A-D]\.', input_string)
    cleaned_matches = [re.sub(r'[\(\)\.]', '', match) for match in matches]
    if cleaned_matches:
        return cleaned_matches[0]
    else:
        return None


def contains_single_digit(string, digit):
    return bool(re.search(fr'\b{digit}\b', string))

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_logger(root_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(root_dir, "log.txt"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def encode_image(image):
    return Image.fromarray(image[:, :, ::-1])

def get_image_from_text(text_id, anns, video_cap, new_size=490):
    begin_time = anns[text_id]['begin_time'] + 0.5
    end_time = anns[text_id]['end_time'] - 0.5
    mid = begin_time + (end_time - begin_time) / 2
    mid_l = begin_time + (mid - begin_time) / 2
    mid_r = mid + (end_time - mid) / 2
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for t in [begin_time, mid_l, mid, mid_r, end_time]:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = video_cap.read()
        if ret:
            frame = cv2.resize(frame, (int(new_size/2), int(new_size/5)))
            frames.append(frame)
        else:
            print(f"Error reading frame at time {t} for text ID {text_id}, check the video.")
            return None
    return cv2.vconcat(frames)

def generate_answer_qwen(processor, model, images, context, question, prompt):
    contents = []
    for i, img in enumerate(images):
        contents += [{"type": "text", "text": f"Scene {i+1}: "}, {"type": "image", "image": img}]
    prompt_text = prompt.replace('<CONTEXT>', context).replace('<QUESTION>', question) + "\n\nAnswer:\n"
    messages = [{"role": "user", "content": contents + [{"type": "text", "text": prompt_text}]}]
    with torch.no_grad():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt")
        for k, v in inputs.items():
            v.to(model.device) 
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, outputs)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

def generate_answer_gemini(images, context, question, prompt, client, args):
    prompt_text = prompt.replace('<CONTEXT>', context).replace('<QUESTION>', question) + "\n\nAnswer:\n"
    contents = [prompt_text] + images if images else prompt_text
    retry = 0
    while retry < 5:
        try:
            resp = client.models.generate_content(model=args.model_id, contents=contents,
                        config=types.GenerateContentConfig(candidate_count=1,temperature=0,max_output_tokens=100,))
            return resp.text
        except Exception as e:
            print(f"Error generating answer: {e}")
            retry += 1
            time.sleep(10)
    return None


def main(args):
    create_folder_if_not_exists(args.output_dir)
    logger = get_logger(args.output_dir)

    logger.info(f"Args: {args}")

    if 'gemini' in args.model_id:
        client = genai.Client(api_key=args.api_key)

    elif args.model_id == 'qwen2.5vl-7b-instruct':
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="../HuggingFace/", trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="../HuggingFace/", torch_dtype="auto", device_map="auto")
        model.eval()

    anns_all = json.load(open("./datasets/annotations.json", 'r'))

    with open(args.prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    for dataset_name in ['yms', 'symon']:
        logger.info(f"Testing dataset: {dataset_name}")
        dataset_key = 'en' if dataset_name == "symon" else dataset_name
        dataset_anns = anns_all[dataset_key]
        movies = list(dataset_anns.keys())

        if args.DEBUG:
            logger.info("Debug mode")
            movies = ['0002', '0013'] if dataset_name == 'yms' else ['__yID2Chs7s', '_fHuuL01ikc']

        counter_all = 0
        counter_correct = 0
        for movie_id in tqdm(movies):
            out_path = os.path.join(args.output_dir, f"{movie_id}_answer.json")
            if os.path.exists(out_path):
                continue

            anns = dataset_anns[movie_id]['annotations']
            max_idx = len(anns) - 1
            video_cap = cv2.VideoCapture(os.path.join(args.video_path, f"{dataset_name}/{movie_id}.mp4"))
            if not video_cap:
                print(f"Video not found for {movie_id}, check the path.")
                continue

            full_texts = dataset_anns[movie_id]['texts']
            task_name = str(args.task).split('_')[0]
            with open(os.path.join(args.questions_path, f"{dataset_name}/{task_name}/{movie_id}.json"), 'r', encoding='utf-8') as f:
                questions = json.load(f)

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

                logger.info(f"{movie_id} | Cause: {cause_idx} | Effect: {effect_idx}")

                if 'gemini' in args.model_id:
                    images = [encode_image(get_image_from_text(i, anns, video_cap, 1024)) for i in range(cause_idx-left_expansion, right_end+1)]
                else:
                    images = [encode_image(get_image_from_text(i, anns, video_cap, 490)) for i in range(cause_idx-left_expansion, right_end+1)]


                if 'gemini' in args.model_id:
                    answer = generate_answer_gemini(images, context, v_question, prompt, client, args)
                elif 'qwen' in args.model_id:
                    answer = generate_answer_qwen(processor, model, images, context, v_question, prompt)

                logger.info(f"Q: {question}\nGT: {correct_answer}\nAnswer: {answer}")
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
