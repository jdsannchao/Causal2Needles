#!/bin/bash

# Usage:
# bash run.sh gemini-pro-1.5-002 your_api_key
# bash run.sh llava none

model_id=$1
api_key=$2
dataset="yms"
video_path="./dataset/videos/${dataset}/"
expansion=5
DEBUG=0

echo "Running evaluation for model: $model_id"
echo "Dataset: $dataset"

# Shared question paths
s2_questions_path="questions/${dataset}/s2/"
s1_questions_path="questions/${dataset}/s1/"

# --- GEMINI Mode ---
if [[ $model_id == gemini* ]]; then
    echo "Using Gemini API Key..."

    ## For 2-needle image description task
    task="s2_vision"
    prompt_path="prompts/test/test_prompt_Commercial_s2_vision.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_Commercial_s2_vision.py "$model_id" "$api_key" "$dataset" "$video_path" "$prompt_path" "$s2_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

    ## For 2-needle visual grounding task (forward)
    task="s2_normal"
    prompt_path="prompts/test/test_prompt_Commercial_s2_normal.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_Commercial_s2.py "$model_id" "$api_key" "$dataset" "$video_path" "$prompt_path" "$s2_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

    ## For 2-needle visual grounding task (reverse)
    task="s2_rev"
    prompt_path="prompts/test/test_prompt_Commercial_s2_rev.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_Commercial_s2.py "$model_id" "$api_key" "$dataset" "$video_path" "$prompt_path" "$s2_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

    ## For 1-needle visual grounding task
    task="s1"
    prompt_path="prompts/test/test_prompt_Commercial_s1.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_Commercial_s2.py "$model_id" "$api_key" "$dataset" "$video_path" "$prompt_path" "$s1_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

# --- LLAVA Mode ---
elif [[ $model_id == llava* ]]; then
    echo "Using local LLaVA model..."

    ## For 2-needle image description task
    task="s2_vision"
    prompt_path="prompts/test/test_prompt_Commercial_s2_vision.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_MLLM_s2_vision.py "$model_id" "none" "$dataset" "$video_path" "$prompt_path" "$s2_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

    ## For 2-needle visual grounding task (forward)
    task="s2_normal"
    prompt_path="prompts/test/test_prompt_Commercial_s2_normal.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_MLLM_s2.py "$model_id" "none" "$dataset" "$video_path" "$prompt_path" "$s2_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

    ## For 2-needle visual grounding task (reverse)
    task="s2_rev"
    prompt_path="prompts/test/test_prompt_Commercial_s2_rev.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_MLLM_s2.py "$model_id" "none" "$dataset" "$video_path" "$prompt_path" "$s2_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

    ## For 1-needle visual grounding task
    task="s1"
    prompt_path="prompts/test/test_prompt_Commercial_s1.txt"
    output_dir="experiments/${dataset}/${model_id}/${task}/"
    python test_MLLM_s2.py "$model_id" "none" "$dataset" "$video_path" "$prompt_path" "$s1_questions_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"

else
    echo "Unsupported model_id: $model_id"
    echo "Please use 'gemini-pro-1.5-002' or 'llava-next-7b'."
    exit 1
fi

