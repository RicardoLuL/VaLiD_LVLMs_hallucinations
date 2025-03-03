import os
import re
import sys
import json
import torch
import argparse
import random as rd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
from cog import Input
from transformers import AutoTokenizer
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from utils import *
from sampling.entropy_sample import evolve_valid_sampling

evolve_valid_sampling()

GQA_FOLDER = os.environ.get("GQA_FOLDER", "<your-gqa-folder>")
MME_FOLDER = os.environ.get("MME_FOLDER", "<your-mme-folder>")
COCO_FOLDER = os.environ.get("COCO_FOLDER", "<your-coco-folder>")
AMBER_FOLDER = os.environ.get("AMBER_FOLDER", "<your-amber-folder>")

cache_dir = "./cache"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

def prepare_pope_for_evaluation(is_sample=False, data_name='coco'):
    question_folder = "../data/POPE"
    if data_name in ['coco', 'aokvqa']:
        dataset_folder = f"{COCO_FOLDER}/images/val2014"
    elif data_name == 'gqa':
        dataset_folder = GQA_FOLDER
    data = []
    for name in ['adversarial', 'popular', 'random']:
        subdata = []
        with open(os.path.join(question_folder, f"{data_name}/{data_name}_pope_{name}.json"), 'r') as file:
            for line_data in file:
                item = json.loads(line_data)
                image_name, query, label = item['image'], item['text'], item['label']
                subdata.append({
                    'image': os.path.join(dataset_folder, image_name), 
                    'text': query + " Please answer this question with one word.", 
                    'label': label, 'cate': name
                })
        if is_sample:
            subdata = rd.sample(subdata, k=100)
        data += subdata
    return data

def prepare_mme_for_evaluation(is_sample=False):
    data = []
    with open(f'../data/mme.json', 'r') as file:
        for line_data in file:
            meta_data = json.loads(line_data)
            data.append({
                'image': os.path.join(MME_FOLDER, meta_data['image']), 
                'text': meta_data['text'] + " Please answer this question with one word.", 
                'label': meta_data['label'], 'cate': meta_data['category']
            })
    if is_sample:
        data = rd.sample(data, k=200)
    return data

def prepare_amber_for_evaluation(is_sample=False):
    question_file = "../data/amber-discriminative.json"
    data = []
    with open(question_file, 'r') as file:
        for line_data in file:
            item = json.loads(line_data)
            image_name, query, label = item['image'], item['text'], item['label']
            data.append({
                'image': os.path.join(AMBER_FOLDER, image_name), 
                'text': query + " Please answer this question with one word.", 
                'label': label, 'cate': item['type']
            })
    if is_sample:
        data = rd.sample(data, k=500)
    return data

def load_model(model_path):
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True
    ).eval()
    return model, tokenizer

def func_process_image(image_dir):
    image_tensor = Image.open(image_dir).convert("RGB")
    image_tensor = qwen.model.transformer.visual.image_transform(image_tensor).unsqueeze(0).to(qwen.model.device)
    return image_tensor


class Predictor:
    def __init__(self, args) -> None:
        self.model, self.tokenizer = load_model(model_path=args.model_path)
        self.config = args
        
    def run(
        self,
        images_tensor: torch.Tensor = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        ):
        input_ids = self.tokenizer([prompt], return_tensors='pt', padding='longest')
        
        pred = self.model.generate(
            input_ids=input_ids.input_ids.cuda(),
            attention_mask=input_ids.attention_mask.cuda(),
            do_sample=True,
            max_new_tokens=20,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=self.tokenizer.eod_id,
            eos_token_id=self.tokenizer.eod_id,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            images=images_tensor,
            layer_index=self.config.layer_index,
            valid_alpha=self.config.valid_alpha, 
            valid_beta=self.config.valid_beta, 
            use_valid=self.config.use_valid, 
        )
        outputs = [
            self.tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ][0]
        outputs = outputs.strip()
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=float, default=1)
    parser.add_argument("--layer_index", type=int, default=-1)
    parser.add_argument("--use_valid", action='store_true', default=False)
    parser.add_argument("--valid_alpha", type=float, default=0.5)
    parser.add_argument("--valid_beta", type=float, default=0.1)
    parser.add_argument("--is_sample", action='store_true', default=False, help='eval a subset')
    parser.add_argument("--eval_data", default='coco', help='data name')
    parser.add_argument("--model_path", default='./Qwen-VL')
    args = parser.parse_args()
    
    if args.use_valid:
        file_name = f'{args.eval_data}_valid-alpha:{args.valid_alpha}-beta:{args.valid_beta}'
    else:
        file_name = f'{args.eval_data}_original'
    
    if args.eval_data in ['coco', 'aokvqa', 'gqa']:
        dataset = prepare_pope_for_evaluation(args.is_sample, data_name=args.eval_data)
    elif args.eval_data == 'amber':
        dataset = prepare_amber_for_evaluation(args.is_sample)
    elif args.eval_data in 'mme':
        dataset = prepare_mme_for_evaluation(args.is_sample)   
    print('=> dataset is ok...')
        
    qwen = Predictor(args)
    print('=> model is ok...')
    
    savings = []
    for index, item in enumerate(dataset):
        label = item['label']
        text = '<img>{}</img>{} Answer:'.format(item['image'], item['text'])
        images_tensor = func_process_image(item['image'])
        outputs = qwen.run(images_tensor=images_tensor, prompt=text)
        print(f"[Ask]-[{index+1}/{len(dataset)}]:{item['text']} \t [Ans]:{outputs} \t [LABEL]:{label}")
        item['pred'] = outputs
        savings.append(item)
    
    with open(f'{cache_dir}/{file_name}.json', 'w', encoding='utf-8') as file:
        json.dump(savings, file, ensure_ascii=False, indent=2)
    
    if args.eval_data in ['coco', 'aokvqa', 'gqa']:
        eval_pope(savings, file_dir=file_name)
    elif args.eval_data == 'amber':
        eval_amber(savings, file_dir=file_name)
    elif args.eval_data in'mme':
        processing_output(savings, exp_folder_dir=file_name)