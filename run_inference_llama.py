import os
import argparse
import json
import time
import random
import gc

import torch
from tqdm import tqdm

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from vln.clip import PanoCLIP
from vln.env import ClipEnv
from vln.dataset import load_dataset
from vln.agent import Agent
from vln.env import get_gold_nav
from vln.prompt_builder import get_navigation_lines
from vln.evaluate import get_metrics_from_results

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--model_name', default='meta-llama/Llama-2-7b-hf', type=str)  # decapoda-research/llama-7b-hf, facebook/opt-1.3b
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='map2seq', choices=['touchdown', 'map2seq'], type=str)
parser.add_argument('--scenario', default='unseen', choices=['seen', 'unseen'], type=str)
parser.add_argument('--split', default='dev', choices=['dev', 'test'], type=str)
parser.add_argument('--image', default='openclip', choices=['openclip', 'clip', 'none'], type=str)
parser.add_argument('--clip_cache_dir', default='./features', type=str)
parser.add_argument('--output_dir', default='./outputs_llm_inference/', type=str)
parser.add_argument('--prompt_file', default='2shot.txt', type=str)  # filename in llm/prompts/{dataset_name}/navigation/
parser.add_argument('--num_instances', default=-1, type=int)  # -1 for all instances
parser.add_argument('--max_steps', default=55, type=int)
parser.add_argument('--hf_auth_token', default='', type=str)
parser.add_argument('--seed', default=1, type=int)
opts = parser.parse_args()

random.seed(opts.seed)

split = opts.split
image_threshold = 3.5
image_prompt = 'picture of {}'
landmarks_name = 'gpt3_5shot'
image = opts.image


model_name = opts.model_name
exp_name = model_name.split('/')[-1]

exp_name = exp_name + '_' + str(opts.seed)
print('exp_name', exp_name)

dataset_dir = os.path.join(opts.datasets_dir, opts.dataset_name + '_' + opts.scenario)
graph_dir = os.path.join(dataset_dir, 'graph')
landmarks_dir = os.path.join(opts.datasets_dir, 'landmarks')
landmarks_file = os.path.join(landmarks_dir, opts.dataset_name, f'{landmarks_name}_unfiltered.json')
is_map2seq = opts.dataset_name == 'map2seq'

panoCLIP = None
if image != 'none':
    panoCLIP = PanoCLIP(model_name=image, device="cpu", cache_dir=opts.clip_cache_dir)
env = ClipEnv(graph_dir, panoCLIP, image_threshold=image_threshold, image_prompt=image_prompt)

with open(os.path.join('llm', 'prompts', opts.dataset_name, 'navigation', opts.prompt_file)) as f:
    prompt_template = ''.join(f.readlines())


def main():

    print('load instances')
    inference_instances = load_dataset(split, env, dataset_dir, opts.dataset_name, landmarks_file)
    train_instances = load_dataset('train', env, dataset_dir, opts.dataset_name, landmarks_file)
    print('instances loaded')

    if opts.num_instances > 0:
        inference_instances = inference_instances[:opts.num_instances]

    icl_shots = list()
    for _ in inference_instances:
        icl_shots.append(random.sample(train_instances, 2))

    results, kpa, spd, tc = run_inference(inference_instances, icl_shots)

    results['opts'] = vars(opts)

    output_dir = os.path.join(opts.output_dir, opts.dataset_name + '_' + opts.scenario, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'{split}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def run_inference(inference_instances, icl_shots):
    print('load model')
    print(model_name)

    if 'Llama-2' in model_name:
        hf_auth_token = opts.hf_auth_token
        assert hf_auth_token != ''

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_auth_token)
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=hf_auth_token)

        print(model.hf_device_map)
        print(model.dtype)
        print(model.device)
        print(tokenizer.pad_token_id)

    elif 'decapoda-research/llama' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        print('llama loaded')

    elif 'mosaicml' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.unk_token_id

        config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'torch'
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.to(device='cuda:0')
    elif 'falcon' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        print(model.hf_device_map)
    print('model loaded')
    print(model.device)
    print(model.dtype)

    tc, spd, kpa, results = run_navigation(model, tokenizer, inference_instances, icl_shots, env, opts.max_steps)
    print('kpa', kpa)
    print('spd', spd)
    print('tc', tc)

    return results, kpa, spd, tc


def run_navigation(model, tokenizer, instances, icl_shots, env, max_steps):
    results = dict()
    results['prompt_template'] = prompt_template
    results['time'] = int(time.time())
    results['num_novel'] = 0
    results['instances'] = dict()

    assert len(instances) == len(icl_shots)
    for instance, _icl_shots in tqdm(list(zip(instances, icl_shots))):
        torch.cuda.empty_cache()
        gc.collect()

        icl_instance_1 = _icl_shots[0]
        gold_nav_shot_1 = get_gold_nav(icl_instance_1, env)
        gold_navigation_lines_shot_1, _ = get_navigation_lines(gold_nav_shot_1, env, icl_instance_1['landmarks'],
                                                               icl_instance_1.get('traffic_flow'))
        icl_instance_2 = _icl_shots[1]
        gold_nav_shot_2 = get_gold_nav(icl_instance_2, env)
        gold_navigation_lines_shot_2, _ = get_navigation_lines(gold_nav_shot_2, env, icl_instance_2['landmarks'],
                                                               icl_instance_2.get('traffic_flow'))

        _prompt_template = prompt_template.format(icl_instance_1['navigation_text'],
                                                  '\n'.join(gold_navigation_lines_shot_1),
                                                  icl_instance_2['navigation_text'],
                                                  '\n'.join(gold_navigation_lines_shot_2),
                                                  '{}')

        nav, navigation_lines, is_actions = run_navigation_instance(model,
                                                                    tokenizer,
                                                                    env,
                                                                    max_steps,
                                                                    instance,
                                                                    _prompt_template
                                                                    )

        target_panoid = instance['target_panoid']
        target_list = env.graph.get_target_neighbors(target_panoid) + [target_panoid]
        is_novel = False
        if nav.pano_path[-1] in target_list and len(nav.pano_path) - len(instance['route_panoids']) >= 2:
            is_novel = True
            results['num_novel'] += 1

        gold_nav = get_gold_nav(instance, env)
        gold_navigation_lines, gold_is_actions = get_navigation_lines(gold_nav,
                                                                      env,
                                                                      instance['landmarks'],
                                                                      instance.get('traffic_flow'))
        result = dict(idx=instance['idx'],
                      start_heading=instance['start_heading'],
                      gold_actions=gold_nav.actions,
                      gold_states=gold_nav.states,
                      gold_pano_path=instance['route_panoids'],
                      gold_navigation_lines=gold_navigation_lines,
                      gold_is_actions=gold_is_actions,
                      agent_actions=nav.actions,
                      agent_states=nav.states,
                      agent_pano_path=nav.pano_path,
                      agent_navigation_lines=navigation_lines,
                      agent_is_actions=is_actions,
                      is_novel=is_novel)

        results['instances'][result['idx']] = result

    correct, tc, spd, kpa, results = get_metrics_from_results(results, env.graph)
    return tc, spd, kpa, results


def run_navigation_instance(model, tokenizer, env, max_steps, instance, prompt_template):

    def query_func(prompt, hints):
        if 'mosaic' in model_name:
            try:
                prompt, query, new_hints = _query_func(prompt, hints)
            except RuntimeError as e:
                print('catched error')
                print(e)
                return prompt + ' forward', 0, hints
        else:
            prompt, query, new_hints = _query_func(prompt, hints)
        return prompt, query, new_hints

    def _query_func(prompt, hints):
        with torch.autocast("cuda"):
            if 'mosaicml/mpt' in model_name:
                inputs = tokenizer([prompt], padding=True, return_tensors="pt", truncation=True).to(model.device)
            elif 'falcon' in model_name:
                inputs = tokenizer([prompt + ' '], return_tensors="pt", return_token_type_ids=False).to(model.device)
            else:
                inputs = tokenizer([prompt], padding=True, return_token_type_ids=False, return_tensors="pt").to(model.device)

            new_hints = dict(input_ids=inputs['input_ids'])

            past_key_values = None
            if hints:
                past_key_values = hints['past']
                past_input_ids = hints['input_ids']

                new_input_ids = inputs['input_ids'][0][len(past_input_ids[0]):]
                new_input_ids = torch.unsqueeze(new_input_ids, dim=0)

                inputs['input_ids'] = new_input_ids.to(model.device)

            with torch.no_grad():
                raw_outputs = model(**inputs,
                                    return_dict=True,
                                    output_hidden_states=False,
                                    output_attentions=False,
                                    use_cache=True,
                                    past_key_values=past_key_values
                                    )
                past = raw_outputs.past_key_values
                new_hints['past'] = past

            generated_logits = raw_outputs.logits.detach()[:, -1, :]
            generated_id = torch.argmax(generated_logits, dim=-1)[0].item()

            if 'decapoda-research/llama' in model_name:
                token = tokenizer.sp_model.IdToPiece(int(generated_id))
                output = tokenizer.sp_model.decode(token)
            else:
                outputs = tokenizer.batch_decode([generated_id], skip_special_tokens=True)
                output = outputs[0]

            if output[0] != ' ':
                output = ' ' + output

            if output == ' turn':
                output = ' turn_around'

            return prompt + output, 0, new_hints

    agent = Agent(query_func, env, instance, prompt_template)
    nav, navigation_lines, is_actions, _ = agent.run(max_steps, verbatim=False)
    return nav, navigation_lines, is_actions


if __name__ == '__main__':
    main()
