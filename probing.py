# Author: Qin Xiaotong
# CreateTime: 2024/5/10
# FileName: probing
# Description:


import argparse
import json
import time
import random
from random import sample
import time
from openai import OpenAI

import torch
import os
from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM, GPTNeoXTokenizerFast,
    T5Tokenizer, T5ForConditionalGeneration,
    LlamaForCausalLM, LlamaTokenizer,
    AutoTokenizer, AutoModelForCausalLM
)
import math
import unicodedata


def initialize_model_and_tokenizer(model_name: str):
    if model_name == "EleutherAI/gpt-neo-2.7B":
        tokenizer = GPT2Tokenizer.from_pretrained('/netcache/huggingface/hub/models--EleutherAI--gpt-neo-2.7B/snapshots/e24fa291132763e59f4a5422741b424fb5d59056')
        model = GPTNeoForCausalLM.from_pretrained('/netcache/huggingface/hub/models--EleutherAI--gpt-neo-2.7B/snapshots/e24fa291132763e59f4a5422741b424fb5d59056')
        model.to(device)
    elif model_name == "EleutherAI/gpt-neox-20b":
        tokenizer = GPTNeoXTokenizerFast.from_pretrained('/netcache/huggingface/gpt-neox-20b')
        model = GPTNeoXForCausalLM.from_pretrained('/netcache/huggingface/gpt-neox-20b', device_map="auto")
    elif model_name == "EleutherAI/gpt-neo-1.3B":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
        model.to(device)
    elif model_name == "EleutherAI/gpt-neo-125M":
        tokenizer = GPT2Tokenizer.from_pretrained('/netcache/huggingface/gpt-neo-125m')
        model = GPTNeoForCausalLM.from_pretrained('/netcache/huggingface/gpt-neo-125m')
        model.to(device)
    elif model_name == "meta-llama/Llama-2-7b-hf":
        tokenizer = LlamaTokenizer.from_pretrained('/netcache/huggingface/llama-2-7b-hf')
        model = LlamaForCausalLM.from_pretrained('/netcache/huggingface/llama-2-7b-hf', device_map="auto")
    elif model_name == "meta-llama/Llama-2-7b-chat-hf":
        tokenizer = LlamaTokenizer.from_pretrained('/netcache/huggingface/Llama-2-7b-chat-hf')
        model = LlamaForCausalLM.from_pretrained('/netcache/huggingface/Llama-2-7b-chat-hf', device_map="auto")
    elif model_name == "meta-llama/Llama-2-13b-hf":
        tokenizer = LlamaTokenizer.from_pretrained('/netcache/huggingface/Llama-2-13b-hf')
        model = LlamaForCausalLM.from_pretrained('/netcache/huggingface/Llama-2-13b-hf', device_map="auto")
    elif model_name == "meta-llama/Llama-2-13b-chat-hf":
        tokenizer = LlamaTokenizer.from_pretrained('/netcache/huggingface/Llama-2-13b-chat-hf')
        model = LlamaForCausalLM.from_pretrained('/netcache/huggingface/Llama-2-13b-chat-hf', device_map="auto")
    elif model_name == "mistralai/Mistral-7B-v0.1":
        tokenizer = AutoTokenizer.from_pretrained('/netcache/huggingface/Mistral-7B-v0.1')
        model = AutoModelForCausalLM.from_pretrained('/netcache/huggingface/Mistral-7B-v0.1', device_map="auto")
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        tokenizer = AutoTokenizer.from_pretrained('/netcache/huggingface/Mistral-7B-Instruct-v0.2')
        model = AutoModelForCausalLM.from_pretrained('/netcache/huggingface/Mistral-7B-Instruct-v0.2', device_map="auto")
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval()

    return model, tokenizer


def get_prediction(input_text, ground_truth, top_k):
    gt_idx = tokenizer.encode(ground_truth, return_tensors="pt")
    gt_idx = gt_idx.to(device)
    gt_idx = gt_idx.flatten().tolist()
    print(f"ground truth:{tokenizer.decode(gt_idx)}")
    print(f"ground truth id:{gt_idx}")

    n_targets = len(gt_idx)
    if args.model_name in ['meta-llama/Llama-2-7b-hf', "meta-llama/Llama-2-13b-hf", "mistralai/Mistral-7B-v0.1",
                                           "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"]:
        n_targets = n_targets - 1
    all_target_prob = []

    preds = ''
    for i in range(n_targets):
        input_idx = tokenizer.encode(input_text, return_tensors='pt')
        input_idx = input_idx.to(device)
        with torch.no_grad():
            logits = model(input_idx).logits
        prob = torch.softmax(logits[:, -1, :], dim=1)
        top_indice = prob.argsort(descending=True)[:, 0]
        # 将索引转换为词汇
        pred = tokenizer.decode(top_indice.item())

        target_prob = prob[:, gt_idx[i]].item()
        all_target_prob.append(target_prob)
        preds += pred
        print(f'prediction:{preds}')
        # print(f'all target prob: {all_target_prob}')

        input_text = input_text + tokenizer.decode(gt_idx[i])

    target_probs = math.prod(all_target_prob)
    print(f'target_probs: {target_probs}')


    return preds, target_probs


def text_generate(input_text, max_len, temperature):
    tokenized_input = tokenizer(input_text, return_tensors='pt').to(device)
    input_idx, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]

    gen_token = model.generate(
        input_ids=input_idx,
        attention_mask=attention_mask,
        temperature=temperature,
        max_new_tokens=max_len,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_txt = [tokenizer.decode(x, skip_special_tokens=True) for x in gen_token.detach().cpu().numpy().tolist()]
    gen_txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in gen_txt
    ]
    print(gen_txt)

    return gen_txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default='EleutherAI/gpt-neo-2.7B',
                        type=str,
                        required=True,
                        help="Choose a model to evaluate.")
    # parser.add_argument("--data_path",
    #                     default='./data/ProbingData/probing_data.json',
    #                     type=str,
    #                     required=True,
    #                     help="The data to evaluate.")
    # parser.add_argument("--rlt_dir",
    #                     default='./result/',
    #                     type=str,
    #                     required=True,
    #                     help="The output directory.")

    args = parser.parse_args()

    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup model & tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    print('Model loaded!')


    for i in [6, 7]:
        data_dir = f'./data/prompt/task{i+1}'
        rlt_dir = f'./result/20240614/{args.model_name}/task{i+1}'
        if not os.path.exists(rlt_dir):
            os.makedirs(rlt_dir)

        for filename in os.listdir(data_dir):
            print(f'Processing task{i+1}, {filename}')
            with open(os.path.join(data_dir, filename), 'r') as fr:
                data_bag = json.load(fr)
            rlt = {}
            for ins, prompt_list in data_bag.items():
                rlt[ins] = []
                for prompt in prompt_list:
                    rlt_dict = {}
                    rlt_dict['prompt id'] = prompt[0]
                    input_text = prompt[1]
                    ground_truth = prompt[2]
                    if args.model_name in ['meta-llama/Llama-2-7b-hf', "meta-llama/Llama-2-13b-hf", "mistralai/Mistral-7B-v0.1",
                                           "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"]:
                        ground_truth = ground_truth.strip(' ')
                    if i == 2:
                        max_tokens = 6
                        temperature = 0.2
                        preds = text_generate(input_text, max_tokens, temperature)
                    elif i == 5:
                        max_tokens = 100
                        temperature = 0.8
                        preds = text_generate(input_text, max_tokens, temperature)
                    else:
                        preds, target_probs = get_prediction(input_text, ground_truth, 1)
                        if preds.strip(' ') == ground_truth.strip(' '):
                            rlt_dict['accuracy'] = 1
                        else:
                            rlt_dict['accuracy'] = 0
                    rlt_dict['prediction'] = preds
                    rlt_dict['ground truth'] = prompt[2]
                    rlt[ins].append(rlt_dict)


            with open(os.path.join(rlt_dir, filename), 'w') as fw:
                json.dump(rlt, fw, indent=2)




