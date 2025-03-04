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
from lavis.models import load_model_and_preprocess
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

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

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

def load_model(model_type="vicuna7b"):
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct", model_type=model_type, is_eval=True, device=device
    )
    return model, vis_processors

def func_process_image(image_file):
    raw_image = Image.open(image_file).convert("RGB")
    # prepare the image
    image_tensor = InstructBLIP.vis_processors["eval"](raw_image).unsqueeze(0).to(InstructBLIP.model.device)
    return image_tensor

class Predictor:
    def __init__(self, args) -> None:
        self.model, self.vis_processors = load_model(model_type=args.model_path)
        self.config = args
    
    def run(
        self,
        images_tensor: torch.Tensor = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        ):
        
        with torch.inference_mode():
            outputs = self.model.generate(
                {"image": images_tensor, "prompt": prompt},
                min_length=1,
                max_length=256,
                length_penalty=1,
                repetition_penalty=1,
                use_nucleus_sampling=True,
                top_p=self.config.top_p, 
                num_beams=self.config.num_beams,
                temperature=self.config.temperature,
                use_valid=self.config.use_valid, 
                valid_alpha=self.config.valid_alpha, 
                valid_beta=self.config.valid_beta, 
                layer_index=self.config.layer_index
            )
        outputs = outputs[0].strip()
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
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--is_sample", action='store_true', default=False, help='eval a subset')
    parser.add_argument("--eval_data", default='number', help='data name')
    parser.add_argument("--model_path", default='vicuna7b')
    args = parser.parse_args()
    
    if args.use_valid:
        file_name = f'{args.eval_data}_valid_entropy_fusion-alpha:{args.valid_alpha}-beta:{args.valid_beta}'
    else:
        file_name = f'{args.eval_data}_original'
    
    if args.eval_data in ['coco', 'aokvqa', 'gqa']:
        dataset = prepare_pope_for_evaluation(args.is_sample, data_name=args.eval_data)
    elif args.eval_data == 'amber':
        dataset = prepare_amber_for_evaluation(args.is_sample)
    elif args.eval_data in 'mme':
        dataset = prepare_mme_for_evaluation(args.is_sample)   
    print('=> dataset is ok...')
        
    InstructBLIP = Predictor(args)
    print('=> model is ok...')
    
    savings = []
    for index, item in enumerate(dataset):
        label = item['label']
        image_path, prompt = item['image'], item['text']
        
        images_tensor = func_process_image(item['image'])
        outputs = InstructBLIP.run(images_tensor=images_tensor, prompt=prompt)
        
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