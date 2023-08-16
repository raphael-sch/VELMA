import os
import json
import argparse

from vln.graph_loader import GraphLoader
from vln.evaluate import get_metrics_from_results


parser = argparse.ArgumentParser(description='Define evaluation parameters')
parser.add_argument('--output_file', default='outputs/map2seq_unseen/gpt-4-0314/paper_openclip_Lgpt3_5shot_gpt-4-0314_dev.json', type=str)
parser.add_argument('--datasets_dir', default='./datasets', type=str)
opts = parser.parse_args()

dataset_dir = os.path.join(opts.datasets_dir, 'map2seq_unseen')  # does not matter for evaluation, only need graph files
graph_dir = os.path.join(dataset_dir, 'graph')

graph = GraphLoader(graph_dir).construct_graph()

with open(opts.output_file) as f:
    results = json.load(f)

correct, tc, spd, kpa, results = get_metrics_from_results(results, graph)
print('tc', round(tc, 1))
print('spd', round(spd, 1))
print('kpa', round(kpa, 1))

