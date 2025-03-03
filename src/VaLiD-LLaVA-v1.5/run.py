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
from utils import *
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_model_name_from_path, process_images
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

def load_model(model_path, load_4bit=True, load_8bit=False, device="cuda"):
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        load_4bit=load_4bit, load_8bit=load_8bit, 
        device=device
    )
    return tokenizer, model, image_processor, context_len


class Predictor:
    def __init__(self, args) -> None:
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(
            model_path=args.model_path,
            load_4bit=False, load_8bit=False, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = args
        
    def run(
        self,
        images_tensor: torch.Tensor = Input(description="Input image"),
        do_sample: bool = Input(description="Whether use sampling strategy"), 
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
        num_beams: int = Input(description='num_beams')
    ):
        input_ids = (tokenizer_image_token(
            prompt=prompt, 
            tokenizer=self.tokenizer, 
            image_token_index=IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).cuda())

        stop_str = " " if SeparatorStyle.TWO != SeparatorStyle.TWO else "</s>"
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_beams=self.config.num_beams,
                max_new_tokens=max_tokens,
                use_cache=True,
                layer_index=self.config.layer_index,
                valid_alpha=self.config.valid_alpha, 
                valid_beta=self.config.valid_beta, 
                use_valid=self.config.use_valid, 
                stopping_criteria=[stopping_criteria], 
                return_dict_in_generate=True,
                output_attentions=True,
                output_scores=True, 
                output_hidden_states=True,
            )
        
        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

def func_process_image(image_dir):
    image = process_images(
        images=[Image.open(image_dir).convert('RGB').resize((224, 224))], 
        image_processor=llava.image_processor, 
        model_cfg=llava.model.config
    ).to(llava.model.device, dtype=torch.float16)
    return image

def func_process_prompt(query):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        if llava.model.config.mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        if llava.model.config.mm_use_im_start_end:
            query = image_token_se + "\n" + query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + query
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--num_beams", type=float, default=1)
    parser.add_argument("--layer_index", type=int, default=-2)
    parser.add_argument("--use_valid", action='store_true', default=False)
    parser.add_argument("--valid_alpha", type=float, default=0.5)
    parser.add_argument("--valid_beta", type=float, default=0.1)
    parser.add_argument("--is_sample", action='store_true', default=False, help='eval a subset')
    parser.add_argument("--eval_data", default='number', help='data name')
    parser.add_argument("--model_path", default='./LLaVA-v1.5')
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
        
    llava = Predictor(args)
    generate_kwargs = {'top_p': args.top_p, 'temperature': args.temperature, 'max_tokens': 1024, 'num_beams': args.num_beams}
    print('=> model is ok...')
    
    savings = []
    for index, item in enumerate(dataset):
        label = item['label']
        image = func_process_image(item['image'])
        prompt = func_process_prompt(item['text'])
        outputs = llava.run(images_tensor=image, prompt=prompt, do_sample=True, **generate_kwargs)
        print(f"[Ask]-[{index+1}/{len(dataset)}]:{item['text']} \t [Ans]:{outputs} \t [LABEL]:{label}")
        item['pred'] = outputs
        savings.append(item)
    
    with open(f'{cache_dir}/{file_name}.json', 'w', encoding='utf-8') as file:
        json.dump(savings, file, ensure_ascii=False, indent=2)
    
    with open(f'{cache_dir}/{file_name}.json', 'w', encoding='utf-8') as file:
        json.dump(savings, file, ensure_ascii=False, indent=2)
    
    if args.eval_data in ['coco', 'aokvqa', 'gqa']:
        eval_pope(savings, file_dir=file_name)
    elif args.eval_data == 'amber':
        eval_amber(savings, file_dir=file_name)
    elif args.eval_data in'mme':
        processing_output(savings, exp_folder_dir=file_name)