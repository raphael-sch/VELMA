import os
import argparse
import random
import json
import dataclasses

import torch
import transformers

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import LlamaForCausalLM, LlamaTokenizer

from vln.clip import PanoCLIP
from vln.env import ClipEnv, get_gold_nav
from vln.prompt_builder import get_navigation_lines
from vln.dataset import load_dataset
from utils import get_prompt_template, setup_seed
from utils import run_navigation

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--model_name', default='decapoda-research/llama-7b-hf', type=str)
parser.add_argument('--exp_name', default='default', type=str)
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='map2seq', choices=['touchdown', 'map2seq'], type=str)
parser.add_argument('--scenario', default='unseen', choices=['seen', 'unseen'], type=str)
parser.add_argument('--image', default='none', choices=['none', 'clip', 'openclip'], type=str)
parser.add_argument('--image_prompt', default='picture of {}', type=str)
parser.add_argument('--image_threshold', default=3.5, type=float)
parser.add_argument('--landmarks_name', default='gpt3_5shot', type=str)
parser.add_argument('--clip_cache_dir', default='./features', type=str)
parser.add_argument('--output_dir', default='./outputs_finetuned_ft/', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_steps', default=55, type=int)
parser.add_argument('--micro_batch_size', default=1, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--lr_scheduler_type', default='linear', type=str)
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
group = 'ft_paper'
group = dict(touchdown='td', map2seq='m2s')[opts.dataset_name] + f'_{group}_' + opts.model_name.split('/')[-1]
print('args', opts)

dataset_dir = os.path.join(opts.datasets_dir, opts.dataset_name + '_' + opts.scenario)
graph_dir = os.path.join(dataset_dir, 'graph')
landmarks_dir = os.path.join(opts.datasets_dir, 'landmarks')
landmarks_file = os.path.join(landmarks_dir, opts.dataset_name, f'{opts.landmarks_name}_unfiltered.json')
is_map2seq = opts.dataset_name == 'map2seq'
output_dir = os.path.join(opts.output_dir, group, opts.exp_name)

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
    device_map = "auto"

    tokenizer = LlamaTokenizer.from_pretrained(opts.model_name)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = LlamaForCausalLM.from_pretrained(opts.model_name,
                                             torch_dtype=torch.float16,
                                             device_map=device_map)
    model.config.pad_token_id = 0

    config = LoraConfig(
        r=opts.lora_r,
        lora_alpha=opts.lora_alpha,
        target_modules=opts.lora_target_modules,
        lora_dropout=opts.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

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
        metric_for_best_model='spd',
        greater_is_better=False,
        seed=opts.seed,
        data_seed=opts.seed
    )

    prompt_template = get_prompt_template()
    train_dataset = VLNDataset(train_instances, tokenizer, model, prompt_template, 'train')
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
        preprocess_logits_for_metrics=lambda logits, _: torch.argmax(logits, dim=-1),
        compute_metrics=compute_metrics,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    # This will not work in newer versions of peft! https://github.com/huggingface/peft/issues/317
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

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


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    print('predictions', predictions.shape)
    print('labels', labels.shape)

    total = 0.0
    correct = 0
    for batch_id in range(predictions.shape[0]):
        for i in range(predictions.shape[1]-1):
            label = labels[batch_id, i+1]
            if label != -100:
                total += 1
                if label == predictions[batch_id, i]:
                    correct += 1
    accuracy = correct / total * 100
    return dict(accuracy=round(accuracy, 2))


def generate_prompt(prompt_template, navigation_text, simulation_text):
    instructions_prompt = prompt_template.format(navigation_text)
    full_prompt = instructions_prompt + '{}\n'.format(simulation_text)

    return full_prompt, instructions_prompt


class VLNDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer, model, prompt_template, split):
        self.instances = instances
        self._prepare_instances()

        self.tokenizer = tokenizer
        self.model = model
        self.actions_token_ids = self._get_actions_token_ids()
        self.split = split
        self.prompt_template = prompt_template

        self.max_input_length = 0

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

    def __getitem__(self, idx):
        instance = self.instances[idx]
        navigation_lines = instance['navigation_lines']
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

        self.max_input_length = max(self.max_input_length, len(item["input_ids"]))

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

        if result["input_ids"][-1] != self.tokenizer.eos_token_id and len(
                result["input_ids"]) < opts.cutoff_len and add_eos_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = [-100] * len(result["input_ids"])

        return result

    def __len__(self):
        return len(self.instances)


if __name__ == "__main__":
    main()
