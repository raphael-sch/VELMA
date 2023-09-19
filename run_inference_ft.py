import os
import argparse
import json

import torch

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

from vln.clip import PanoCLIP
from vln.env import ClipEnv
from vln.dataset import load_dataset
from utils import run_navigation

parser = argparse.ArgumentParser(description='Define inference parameters')
parser.add_argument('--weights_dir', default='weights/VELMA-FT-touchdown', type=str)
parser.add_argument('--model_name', default='decapoda-research/llama-7b-hf', type=str)
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='map2seq', choices=['touchdown', 'map2seq'], type=str)
parser.add_argument('--scenario', default='unseen', choices=['seen', 'unseen'], type=str)
parser.add_argument('--splits', nargs='+', help='list of strings')
parser.add_argument('--image', default='openclip', choices=['none', 'clip', 'openclip'], type=str)
parser.add_argument('--image_prompt', default='picture of {}', type=str)
parser.add_argument('--image_threshold', default=3.5, type=float)
parser.add_argument('--landmarks_name', default='gpt3_5shot', type=str)
parser.add_argument('--clip_cache_dir', default='./features', type=str)
parser.add_argument('--output_dir', default='./outputs_ft_inference/', type=str)
parser.add_argument('--max_steps', default=55, type=int)
parser.add_argument('--cutoff_len', default=2048, type=int)
opts = parser.parse_args()

print('splits', opts.splits)

dataset_dir = os.path.join(opts.datasets_dir, opts.dataset_name + '_' + opts.scenario)
graph_dir = os.path.join(dataset_dir, 'graph')
landmarks_dir = os.path.join(opts.datasets_dir, 'landmarks')
landmarks_file = os.path.join(landmarks_dir, opts.dataset_name, f'{opts.landmarks_name}_unfiltered.json')
is_map2seq = opts.dataset_name == 'map2seq'
weights_dir = os.path.normpath(opts.weights_dir)

panoCLIP = None
if opts.image != 'none':
    panoCLIP = PanoCLIP(model_name=opts.image, device="cpu", cache_dir=opts.clip_cache_dir)
env = ClipEnv(graph_dir, panoCLIP, image_threshold=opts.image_threshold, image_prompt=opts.image_prompt)

def main():

    for split in opts.splits:
        inference_instances = load_dataset(split, env, dataset_dir, opts.dataset_name, landmarks_file)

        results, kpa, spd, tc = run_inference(inference_instances, weights_dir)
        results['opts'] = vars(opts)

        results_dir = os.path.join(opts.output_dir, os.path.basename(weights_dir))
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'{split}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)


def run_inference(inference_instances, weights_dir):
    print('load weights from: ', weights_dir)

    tokenizer = LlamaTokenizer.from_pretrained(opts.model_name)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = LlamaForCausalLM.from_pretrained(opts.model_name,
                                             torch_dtype=torch.float16,
                                             device_map="auto")
    model.config.pad_token_id = 0

    model = PeftModel.from_pretrained(model, weights_dir, torch_dtype=torch.float16)
    model.half()
    model.config.pad_token_id = 0

    tc, spd, kpa, results = run_navigation(model, tokenizer, inference_instances, env, opts.max_steps)
    print('kpa', kpa)
    print('spd', spd)
    print('tc', tc)

    return results, kpa, spd, tc


if __name__ == '__main__':
    main()
