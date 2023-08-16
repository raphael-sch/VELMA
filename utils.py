import gc
import math
import time
import random

import torch
from vln.evaluate import get_metrics_from_results
from vln.agent import Agent
from vln.env import get_gold_nav
from vln.prompt_builder import get_navigation_lines

from tqdm import tqdm
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_prompt_template():
    text = 'Navigate to the described target location!\n'
    text += 'Action Space: forward, left, right, turn_around, stop\n'
    text += 'Navigation Instructions: "{}"\n'
    instructions_prompt = text + 'Action Sequence:\n'
    return instructions_prompt


def run_navigation(model, tokenizer, instances, env, max_steps):
    model.eval()

    prompt_template = get_prompt_template()

    results = dict()
    results['prompt_template'] = prompt_template
    results['time'] = int(time.time())
    results['num_novel'] = 0
    results['instances'] = dict()
    for instance in tqdm(instances):
        torch.cuda.empty_cache()
        gc.collect()
        nav, navigation_lines, is_actions = run_navigation_instance(model,
                                                                    tokenizer,
                                                                    env,
                                                                    max_steps,
                                                                    instance,
                                                                    prompt_template,
                                                                    sample=False)

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


def run_navigation_instance(model, tokenizer, env, max_steps, instance, prompt_template, sample=False, sample_token_ids=None):

    def query_func(prompt, hints):
        with torch.autocast("cuda"):
            inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
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
            generated_id_argmax = torch.argmax(generated_logits, dim=-1)[0].item()
            if sample:
                logits_sample_token_ids = generated_logits[0][sample_token_ids]
                m = torch.distributions.Categorical(logits=logits_sample_token_ids)
                sampled_action_id = m.sample()
                generated_id = sample_token_ids[sampled_action_id]
            else:
                generated_id = generated_id_argmax
            token = tokenizer.sp_model.IdToPiece(int(generated_id))
            output = tokenizer.sp_model.decode(token)

            if len(output) == 0:
                print('empty token generated')
                output = ' forward'

            if output[0] != ' ':
                output = ' ' + output

            if output == ' turn':
                output = ' turn_around'

            return prompt + output, 0, new_hints

    agent = Agent(query_func, env, instance, prompt_template)
    nav, navigation_lines, is_actions, _ = agent.run(max_steps, verbatim=False)
    return nav, navigation_lines, is_actions


def rl_ratio_decay(current_step, max_steps, start, end, strategy='linear'):
    start_step = start * max_steps
    end_step = end * max_steps

    if current_step <= start_step:
        return 0
    elif current_step >= end_step:
        return 1
    else:
        decay_range = end_step - start_step
        decay_step = current_step - start_step
        decay_ratio = decay_step / decay_range

        if strategy == 'cosine':
            return 1 - (0.5 * (1 + math.cos(math.pi * decay_ratio)))
        else:
            return decay_ratio
