import os
import argparse
import random
import copy
import json
import dataclasses

import torch
import transformers

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import LlamaForCausalLM, LlamaTokenizer

from vln.clip import PanoCLIP
from vln.env import ClipEnv, get_gold_nav, get_gt_action
from vln.base_navigator import BaseNavigator
from vln.prompt_builder import get_navigation_lines
from vln.dataset import load_dataset
from utils import get_prompt_template, setup_seed
from utils import run_navigation, run_navigation_instance
from utils import rl_ratio_decay

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--model_name', default='decapoda-research/llama-7b-hf', type=str)
parser.add_argument('--exp_name', default='default', type=str)
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='map2seq', choices=['touchdown', 'map2seq'], type=str)
parser.add_argument('--scenario', default='unseen', choices=['seen', 'unseen'], type=str)
parser.add_argument('--image', default='none', choices=['none', 'clip', 'openclip'], type=str)
parser.add_argument('--image_prompt', default='picture of {}', type=str)
parser.add_argument('--image_threshold', default=3.5, type=float)
parser.add_argument('--landmarks_name', default='gpt3_5shot', choices=['gpt3_0shot', 'gpt3_5shot'], type=str)
parser.add_argument('--clip_cache_dir', default='./features', type=str)
parser.add_argument('--output_dir', default='./outputs_finetuned_rl/', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_steps', default=55, type=int)
parser.add_argument('--micro_batch_size', default=1, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--lr_scheduler_type', default='linear', type=str)
parser.add_argument('--rl_ratio', default=0.5, type=float)
parser.add_argument('--rl_start', default=0.0, type=float)
parser.add_argument('--rl_end', default=1.0, type=float)
parser.add_argument('--rl_decay', default='constant', choices=['linear', 'cosine', 'constant'], type=str)
parser.add_argument('--cutoff_len', default=2048, type=int)
parser.add_argument('--train_set_size', default=-1, type=int)
parser.add_argument('--val_set_size', default=-1, type=int)
parser.add_argument('--lora_r', default=8, type=int)
parser.add_argument('--lora_alpha', default=16, type=int)
parser.add_argument('--lora_dropout', default=0.05, type=float)
parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj", "v_proj"])  # query_key_value for bloom
parser.add_argument('--seed', default=1, type=int)
opts = parser.parse_args()

setup_seed(opts.seed)

group = 'rl_github'
group = dict(touchdown='td', map2seq='m2s')[opts.dataset_name] + f'_{group}_' + opts.model_name.split('/')[-1]
print('args', opts)

dataset_dir = os.path.join(opts.datasets_dir, opts.dataset_name + '_' + opts.scenario)
graph_dir = os.path.join(dataset_dir, 'graph')
landmarks_dir = os.path.join(opts.datasets_dir, 'landmarks')
landmarks_file = os.path.join(landmarks_dir, opts.dataset_name, f'{opts.landmarks_name}_unfiltered.json')
output_dir = os.path.join(opts.output_dir, group, opts.exp_name)
is_map2seq = opts.dataset_name == 'map2seq'


panoCLIP = None
if opts.image != 'none':
    panoCLIP = PanoCLIP(model_name=opts.image, device="cpu", cache_dir=opts.clip_cache_dir)
env = ClipEnv(graph_dir, panoCLIP, image_threshold=opts.image_threshold, image_prompt=opts.image_prompt)

train_instances = load_dataset('train', env, dataset_dir, opts.dataset_name, landmarks_file)
dev_instances = load_dataset('dev', env, dataset_dir, opts.dataset_name, landmarks_file)
all_dev_instances = dev_instances

random.Random(123).shuffle(train_instances)
random.Random(123).shuffle(dev_instances)

if opts.train_set_size > 0:
    train_instances = train_instances[:opts.train_set_size]
if opts.val_set_size > 0:
    dev_instances = dev_instances[:opts.val_set_size]


def main():

    gradient_accumulation_steps = opts.batch_size // opts.micro_batch_size

    tokenizer = LlamaTokenizer.from_pretrained(opts.model_name)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    device_map = "auto"
    print('device_map', device_map)
    model = LlamaForCausalLM.from_pretrained(opts.model_name, torch_dtype=torch.float16, device_map=device_map)
    model.config.pad_token_id = 0
    print(model.hf_device_map)
    print(model.device)

    config = LoraConfig(
        r=opts.lora_r,
        lora_alpha=opts.lora_alpha,
        target_modules=opts.lora_target_modules,
        lora_dropout=opts.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=opts.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        num_train_epochs=opts.epochs,
        learning_rate=opts.learning_rate,
        lr_scheduler_type=opts.lr_scheduler_type,
        fp16=True,
        fp16_full_eval=True,
        logging_steps=4,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir=output_dir,
        load_best_model_at_end=True,
        group_by_length=False,
        metric_for_best_model='tc',
        greater_is_better=True,
        seed=opts.seed,
        data_seed=opts.seed
    )

    prompt_template = get_prompt_template()
    train_dataset = VLNDataset(train_instances, tokenizer, model, prompt_template, 'train', rl_ratio=opts.rl_ratio)
    dev_dataset = VLNDataset(dev_instances, tokenizer, model, prompt_template, 'dev')

    print('num train instances', len(train_dataset))
    print('num val instances', len(dev_dataset))

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=train_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[MyCallback(train_dataset)]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    model = torch.compile(model)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    if panoCLIP:
        panoCLIP.save_cache()


class CustomTrainer(transformers.Trainer):

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_loop_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        instances = dict(eval=dev_instances, dev=all_dev_instances)[metric_key_prefix]

        epoch = 'finished'
        if metric_key_prefix == 'eval':
            epoch = round(self.state.epoch)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        tokenizer = self.data_collator.tokenizer

        tc, spd, kpa, results = run_navigation(model, tokenizer, instances, env, opts.max_steps)
        results['opts'] = vars(opts)
        results['trainer_state'] = dataclasses.asdict(self.state)

        results_file = os.path.join(self.args.output_dir, f'{metric_key_prefix}_results_{epoch}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        eval_loop_output.metrics[metric_key_prefix + '_tc'] = tc
        eval_loop_output.metrics[metric_key_prefix + '_spd'] = spd
        eval_loop_output.metrics[metric_key_prefix + '_kpa'] = kpa
        return eval_loop_output


def generate_prompt(prompt_template, navigation_text, simulation_text):
    instructions_prompt = prompt_template.format(navigation_text)
    full_prompt = instructions_prompt + '{}\n'.format(simulation_text)

    return full_prompt, instructions_prompt


class MyCallback(transformers.TrainerCallback):

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_step_end(self, args, state, control, **kwargs):
        if opts.rl_decay == 'constant':
            return
        max_steps = state.max_steps
        current_step = state.global_step

        rl_ratio = rl_ratio_decay(current_step, max_steps, opts.rl_start, opts.rl_end, strategy=opts.rl_decay)
        self.train_dataset.set_rl_ratio(rl_ratio)

    def on_epoch_end(self, args, state, control, **kwargs):
        random.shuffle(self.train_dataset.instances)

        self.train_dataset.num_train_on_novel = 0
        for key, values in self.train_dataset.stats.items():
            if values:
                self.train_dataset.stats[key] = list()


class VLNDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer, model, prompt_template, split, rl_ratio=0.0):
        self.instances = instances
        self.num_instances = len(self.instances)
        self._prepare_instances()

        self.tokenizer = tokenizer
        self.model = model
        self.actions_token_ids = self._get_actions_token_ids()
        self.split = split
        self.prompt_template = prompt_template
        self.rl_ratio = rl_ratio

        self.trajectory_cache = {ins['navigation_text'] + str(ins['route_panoids']) for ins in instances}

        self.stats = dict()
        for key in ['novel_trajectory', 'spd', 'tc']:
            self.stats[key] = list()
        self.num_train_on_novel = 0

        self.max_input_length = 0

    def set_rl_ratio(self, ratio):
        self.rl_ratio = ratio

    def _prepare_instances(self):
        for instance in self.instances:
            gold_nav = get_gold_nav(instance, env)
            gold_navigation_lines, _ = get_navigation_lines(gold_nav, env, instance['landmarks'], instance.get('traffic_flow'))
            instance['navigation_lines'] = gold_navigation_lines

    def _get_actions_token_ids(self):
        actions_token_ids = list()
        for action in ['forward\n', 'left\n', 'right\n', 'turn_around\n', 'stop\n']:
            _action_token_ids = self.tokenizer(action)['input_ids'][1:]
            actions_token_ids.append(_action_token_ids)
        print('actions_token_ids', actions_token_ids)
        return actions_token_ids

    def _get_rl_navigation_lines(self, instance):
        target_panoid = instance['target_panoid']
        target_list = env.graph.get_target_neighbors(target_panoid) + [target_panoid]

        agent_nav, agent_navigation_lines, _ = run_navigation_instance(self.model,
                                                                       self.tokenizer,
                                                                       env,
                                                                       opts.max_steps,
                                                                       instance,
                                                                       self.prompt_template)
        agent_pano_path = agent_nav.pano_path
        agent_actions = agent_nav.actions
        spd = env.graph.get_shortest_path_length(agent_pano_path[-1], target_panoid)

        is_novel_trajectory = 0
        if agent_pano_path[-1] in target_list and agent_actions[-1] == 'stop':
            desired_pano_path = agent_pano_path

            length_diff_orig = len(agent_pano_path) - len(instance['orig_route_panoids'])
            if length_diff_orig >= 2:
                is_novel_trajectory = 1

        else:
            desired_pano_path = instance['route_panoids']

        gt_action_map = self._get_gt_action_map(agent_actions,
                                                desired_pano_path,
                                                instance['start_heading'])

        self.stats['novel_trajectory'].append(is_novel_trajectory)
        self.stats['spd'].append(spd)
        self.stats['tc'].append(int(agent_pano_path[-1] in target_list) * 100)
        return agent_navigation_lines, agent_pano_path, gt_action_map, is_novel_trajectory

    def _get_gt_action_map(self, agent_actions, desired_pano_path, start_heading):
        nav = BaseNavigator(env)
        start_panoid = desired_pano_path[0]
        nav.init_state(panoid=start_panoid,
                       heading=start_heading)

        gt_actions = list()
        for action in agent_actions[1:]:
            gt_action = get_gt_action(nav, desired_pano_path)
            gt_actions.append(gt_action)
            nav.step(action)

        assert len(agent_actions[1:]) == len(gt_actions)

        gt_action_map = list(zip(agent_actions[1:], gt_actions))
        return gt_action_map

    def _add_novel_trajectory(self, idx, novel_pano_path):
        instance = self.instances[idx]
        target_panoid = instance['target_panoid']

        if novel_pano_path[-2] == target_panoid:
            novel_pano_path.pop()
        elif novel_pano_path[-1] != target_panoid:
            novel_pano_path.append(target_panoid)
        assert novel_pano_path[-1] == target_panoid

        current_gold_path = instance['route_panoids']
        if current_gold_path == novel_pano_path:
            return

        novel_nav = get_gold_nav(dict(start_heading=instance['start_heading'], route_panoids=novel_pano_path), env)
        novel_navigation_lines, _ = get_navigation_lines(novel_nav, env, instance['landmarks'], instance.get('traffic_flow'))

        cache_key = instance['navigation_text'] + str(novel_pano_path)
        if cache_key not in self.trajectory_cache:
            new_instance = copy.deepcopy(instance)
            new_instance['route_panoids'] = novel_pano_path
            new_instance['is_novel'] = True
            new_instance['navigation_lines'] = novel_navigation_lines
            self.instances.append(new_instance)
        self.trajectory_cache.add(cache_key)

    def __getitem__(self, idx, force_no_rl=False):
        instance = self.instances[idx]
        if instance['is_novel']:
            self.num_train_on_novel += 1

        if random.random() < self.rl_ratio and not force_no_rl:
            self.model.eval()
            navigation_lines, agent_pano_path, gt_action_map, is_novel = self._get_rl_navigation_lines(instance)
            self.model.train()
        else:
            navigation_lines = instance['navigation_lines']
            gt_action_map = None

        navigation_text = instance['navigation_text']
        simulation_text = '\n'.join(navigation_lines)
        full_prompt, instructions_prompt = generate_prompt(self.prompt_template,
                                                           navigation_text,
                                                           simulation_text)

        item = self.tokenize(full_prompt)
        tokenized_instructions_prompt = self.tokenize(instructions_prompt)

        # only predict action tokens by masking out all other labels
        i = len(tokenized_instructions_prompt['input_ids']) - 1
        while i < len(item['input_ids']):
            for _action_token_ids in self.actions_token_ids:
                num_tokens = len(_action_token_ids)
                if item['input_ids'][i:i + num_tokens] == _action_token_ids:
                    for j in range(num_tokens - 1):
                        item['labels'][i + j] = _action_token_ids[j]
                    i += (num_tokens - 1)
                    break
            i += 1

        if gt_action_map:
            sample_token_ids = [a[0] for a in self.actions_token_ids]
            actions = ['forward', 'left', 'right', 'turn_around', 'stop']
            action_token_id_map = {a: t for a, t in zip(actions, sample_token_ids)}
            for i in range(len(item['labels'])):
                label_token_id = item['labels'][i]
                if label_token_id in sample_token_ids:
                    agent_action, gt_action = gt_action_map.pop(0)
                    assert action_token_id_map[agent_action] == label_token_id
                    item['labels'][i] = action_token_id_map[gt_action]
            assert len(gt_action_map) == 0

        self.max_input_length = max(self.max_input_length, len(item["input_ids"]))

        if len(item["input_ids"]) > 500:
            print(len(item["input_ids"]))
        # too long: 1703
        # avoid memory errors on my setup
        if self.split == 'train' and len(item["input_ids"]) > 1560:
            print('IL sequence too long, return gold sequence from dataset instead')
            item = self.__getitem__(idx, force_no_rl=True)
        return item

    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=opts.cutoff_len,
            padding=False,
            return_tensors=None
        )
        if len(result["input_ids"]) > opts.cutoff_len:
            print('cutoff', len(result["input_ids"]))

        if result["input_ids"][-1] != self.tokenizer.eos_token_id and len(result["input_ids"]) < opts.cutoff_len and add_eos_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = [-100] * len(result["input_ids"])

        return result

    def __len__(self):
        return self.num_instances


if __name__ == "__main__":
    main()
