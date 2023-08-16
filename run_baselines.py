import json
import os
import argparse
import time
import random

import tqdm

from vln.dataset import load_dataset
from vln.prompt_builder import get_navigation_lines

from vln.env import ClipEnv, get_gold_nav

from vln.evaluate import get_metrics_from_results
from vln.agent import Agent


parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='map2seq', type=str)
parser.add_argument('--baseline', default='forward', type=str, choices=['forward', 'random'])
parser.add_argument('--split', default='dev', type=str)
parser.add_argument('--scenario', default='unseen', type=str)
parser.add_argument('--num_instances', default=-1, type=int)  # -1 for all instances
parser.add_argument('--max_steps', default=55, type=int)  # maximum number of agent steps before run is canceled
parser.add_argument('--landmarks_name', default='gpt3_5shot', choices=['gpt3_0shot', 'gpt3_5shot'], type=str)
parser.add_argument('--output_dir', default='./outputs', type=str)
parser.add_argument('--seed', default=1, type=int)
opts = parser.parse_args()

random.seed(opts.seed)

split = opts.split
num_instances = opts.num_instances
max_steps = opts.max_steps
dataset_name = opts.dataset_name
scenario = opts.scenario
is_map2seq = dataset_name == 'map2seq'

data_dir = opts.datasets_dir
dataset_dir = os.path.join(data_dir, dataset_name + '_' + scenario)
graph_dir = os.path.join(dataset_dir, 'graph')
landmarks_dir = os.path.join(data_dir, 'landmarks')
landmarks_file = os.path.join(landmarks_dir, opts.dataset_name, f'{opts.landmarks_name}_unfiltered.json')

counter = 0


def main():
    env = ClipEnv(graph_dir, panoCLIP=None)

    output_dir = os.path.join(opts.output_dir, dataset_name + '_' + scenario, opts.baseline + '_baseline')
    os.makedirs(output_dir, exist_ok=True)

    instances = load_dataset(split, env, dataset_dir, dataset_name, landmarks_file)

    results = dict()
    results['opts'] = vars(opts)
    results['time'] = int(time.time())
    results['instances'] = dict()

    if num_instances != -1:
        instances = instances[:num_instances]
    print('instances: ', len(instances))

    for i, instance in tqdm.tqdm(list(enumerate(instances))):

        print(i, 'number of instances processed')
        print('idx', instance['idx'])

        result = process_instance(instance, env)
        results['instances'][result['idx']] = result

    correct, tc, spd, sed, results = get_metrics_from_results(results, env.graph)
    print('')
    print('correct', correct)
    print('tc', tc)
    print('spd', spd)
    print('sed', sed)
    print('')

    results_file = os.path.join(output_dir, f'{split}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote results to: ', results_file)


def process_instance(instance, env):

    # main computation
    agent = BaselineAgent(env, instance, opts.baseline)
    nav, navigation_lines, is_actions, query_count = agent.run(max_steps)

    gold_nav = get_gold_nav(instance, env)
    gold_navigation_lines, gold_is_actions = get_navigation_lines(gold_nav, env, agent.landmarks, None)

    global counter
    counter += 1

    print('instance id', instance["id"])
    print('result:')
    print(instance['navigation_text'])
    print(instance['landmarks'])
    print('\n'.join(navigation_lines))
    print('actions', nav.actions)
    print('query_count', query_count)
    print('processed instances', counter)

    result = dict(idx=instance['idx'],
                  navigation_text=instance['navigation_text'],
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
                  query_count=query_count)
    return result


class BaselineAgent(Agent):

    def __init__(self, env, instance, strategy='forward'):
        self.strategy = strategy
        self.pano_length = int(is_map2seq) + 1

        def query_func(prompt, hints=None):
            #step = int(prompt.split('\n')[-1].strip().rstrip('.'))
            if self.pano_length == 40:
                action = 'stop'
            else:
                action = 'forward'

            if action != 'stop' and self.strategy == 'random':
                if prompt.split('\n')[-2].endswith('intersection.'):
                    action = random.choice(['forward', 'left', 'right'])
                if not is_map2seq and self.pano_length == 0:
                    action = random.choice(['forward', 'turn_around'])

            if action == 'forward':
                self.pano_length += 1

            output = prompt + ' ' + action
            return output, 0, dict()

        super().__init__(query_func, env, instance, '')


if __name__ == '__main__':
    main()
