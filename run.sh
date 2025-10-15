#!/bin/bash

# Usage:
# bash run.sh <model_id> <api_key> [debug]
# Example:
#   bash run.sh gemini-1.5-pro your_api_key       # runs without debug
#   bash run.sh gemini-1.5-pro your_api_key debug # runs with debug mode
#   bash run.sh qwen2.5vl-7b-instruct none

model_id=$1
api_key=$2
debug_flag=$3

# Set DEBUG=1 only if third argument is 'debug'
if [ "$debug_flag" == "debug" ]; then
    DEBUG=1
else
    DEBUG=0
fi

video_path="datasets/videos/"
question_path="datasets/questions/"
expansion=5

echo "Running evaluation for model: $model_id"
echo "DEBUG mode: $DEBUG"

declare -a tasks=("2needle_vision" "2needle_normal" "2needle_rev" "1needle" "1needle_nc")
declare -A prompts=(
  ["2needle_vision"]="prompts/test/test_prompt_2needle_vision.txt"
  ["2needle_normal"]="prompts/test/test_prompt_2needle_normal.txt"
  ["2needle_rev"]="prompts/test/test_prompt_2needle_rev.txt"
  ["1needle"]="prompts/test/test_prompt_1needle.txt"
  ["1needle_nc"]="prompts/test/test_prompt_1needle.txt"
)
declare -A scripts=(
  ["2needle_vision"]="test_VLM_2needle_ID.py"
  ["2needle_normal"]="test_VLM_2needle_VG.py"
  ["2needle_rev"]="test_VLM_2needle_VG.py"
  ["1needle"]="test_VLM_1needle.py"
  ["1needle_nc"]="test_VLM_1needle_NC.py"
)

for task in "${tasks[@]}"; do
    prompt_path="${prompts[$task]}"
    script="${scripts[$task]}"
    output_dir="experiments/${model_id}/${task}/"
    echo "Launching: $task"
    python $script "$model_id" "$api_key" "$task" "$video_path" "$prompt_path" "$question_path" "$expansion" "$output_dir" --DEBUG "$DEBUG"
done
