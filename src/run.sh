#!/bin/bash

GQA_FOLDER="<input your dataset folder>"
MME_FOLDER="<input your dataset folder>"
COCO_FOLDER="<input your dataset folder>"
AMBER_FOLDER="<input your dataset folder>"

# run Qwen-VL-7B
cd VaLiD-Qwen-VL
python run.py --model_path=./Qwen-VL \
    --use_valid \
    --valid_alpha=1.0 \
    --valid_beta=0.2 \
    --eval_data=coco

# run LLaVA-v1.5
cd ../VaLiD-LLaVA-v1.5
python run.py --model_path=./LLaVA-v1.5 \
    --use_valid \
    --valid_alpha=1.0 \
    --valid_beta=0.2 \
    --eval_data=aokvqa

# run Qwen-VL-7B
cd ../VaLiD-InstructBLIP
python run.py --model_path=vicuna7b \
    --use_valid \
    --valid_alpha=1.0 \
    --valid_beta=0.2 \
    --eval_data=gqa
