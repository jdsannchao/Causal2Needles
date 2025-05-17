# Causal2Needles

## Overview
**Causal2Needles** is a benchmark dataset and evaluation toolkit designed to assess the capabilities of both proprietary and open-source multimodal large language models in long-video understanding.

This repository provides:  
- Instructions for downloading and setting up the dataset  
- Example scripts for evaluating both commercial and open-source models (e.g., Gemini-Pro-002, LLaVA-Next-7B) 
- Automated evaluation of model performance across three types of questions 

## Dataset Setup

1. Download the **Causal2Needles** dataset from [Hugging Face](https://huggingface.co/datasets/causal2needles/Causal2Needles).
2. After downloading, place the dataset folder under the `dataset/` directory. The structure should look like:

```
Causal2Needles/
  ├── dataset/
  │     ├── videos/                # Folder containing video files
  │     ├── annotations.json       # File containing scene annotations
  │     └── questions/             # Folder containing generated questions
  ├── run.sh                       # Main script to start evaluation
  ├── test_Commercial_s1.py        # Script for evaluating 1-Needle questions on proprietary models
  ├── test_Commercial_s2.py        # Script for evaluating Visual Grounding 2-Needle questions
  ├── test_Commercial_vision.py    # Script for evaluating Image Description 2-Needle questions
  ├── test_MLLM_s1.py              # Script for evaluating 1-Needle questions on open-sourced models
  ├── test_MLLM_s2.py              # Script for evaluating Visual Grounding 2-Needle questions
  ├── test_MLLM_vision.py          # Script for evaluating Image Description 2-Needle questions
  ├── requirements.txt             # Required dependencies for local model execution
```

## How to Run

1. **Install Dependencies**

To ensure compatibility, install all required packages:

```bash
pip install -r requirements.txt
````

2. **Run Evaluation**

We provide example scripts for evaluating two models:

* For **Gemini-Pro-002** (requires API key):

```bash
bash run.sh gemini-pro-1.5-002 your_api_key
```

* For **LLaVA-Next-7B** (runs locally, no API key required):

```bash
bash run.sh llava-next-7b none
```

> Make sure your environment supports running LLaVA-Next-7B locally. Refer to `requirements.txt` for necessary dependencies.

The script will automatically run the selected model on all three evaluation tasks.

## Output

After execution, you will obtain the model's accuracy on the following three types of questions:

* **1-Needle Questions**
* **Visual Grounding 2-Needle Questions**
* **Image Description 2-Needle Questions**

## License

This project is released for academic research purposes only. For commercial usage, please contact the authors.

