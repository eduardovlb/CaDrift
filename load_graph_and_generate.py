import SCM.CausalGraph as cg
import pandas as pd
import random
from DAG_Utils import build_er_dag
from DAG_Utils import extract_graph_statistics

import numpy as np

# dataset_num = '009'
# graph: cg.CausalGraph = cg.load_graph(f'generated_graphs/graph_{dataset_num}.pkl')

seed = np.random.randint(0, 10000)
# seed = 991
random.seed(seed)

# Generate DAG and datasets
dataset_num = '002'
graph, label_node = build_er_dag(n_features=20, p_edge=0.3, task='classification', ensure_label_parents=False, n_confounders=2, create_confounder=True, seed=seed)

print(f"Random seed for dataset {dataset_num}: {seed}")

type = 'target'

# Only valid type/time combinations

if type == 'exogenous':
    valid_combinations = [
        ('exogenous', 'abrupt'),
        ('exogenous', 'gradual'),
        ('exogenous', 'incremental'),
        ('recurrent', 'abrupt'),
        ('recurrent', 'gradual')
    ]
elif type == 'target':
    valid_combinations = [
        ('target', 'abrupt'),
        ('target', 'gradual'),
        ('target', 'incremental'),
        ('recurrent', 'abrupt'),
        ('recurrent', 'gradual')
    ]

elif type == 'endogenous':  
    valid_combinations = [
        ('endogenous', 'abrupt'),
        ('endogenous', 'gradual'),
        ('endogenous', 'incremental'),
        ('recurrent', 'abrupt'),
        ('recurrent', 'gradual')     # recurrent should be abrupt or gradual, but always after a previous drift
    ]

# valid_combinations = [
#     ('exogenous', 'abrupt'),
#     ('exogenous', 'gradual'),
#     ('exogenous', 'incremental'),
#     ('recurrent', 'abrupt'),
#     ('recurrent', 'gradual')
# ]

# valid_combinations = [
#     ('target', 'abrupt'),
#     ('target', 'gradual'),
#     ('target', 'incremental'),
#     ('recurrent', 'abrupt'),
#     ('recurrent', 'gradual')
# ]

# Parameters
max_samples = 1000000
len_min = 50
len_max = 500

# Fixed drift points every 10k (excluding 0)
drift_points = list(range(10000, max_samples, 10000))

drift_sizes = []
drift_types = []
drift_types_time = []

for i, p in enumerate(drift_points):
    if i == 0:
        # First drift cannot be recurrent
        allowed_combs = [(t, tt) for (t, tt) in valid_combinations if t != 'recurrent']
    else:
        allowed_combs = valid_combinations[:]

    drift_type, drift_time = random.choice(allowed_combs)

    # abrupt always size 1
    if drift_time == 'abrupt':
        size = 1
    else:
        size = random.randint(len_min, len_max)

    drift_types.append(drift_type)
    drift_types_time.append(drift_time)
    drift_sizes.append(size)

print("Generated drift events:")
for p, s, t, tt in zip(drift_points, drift_sizes, drift_types, drift_types_time):
    # print(f"Point: {p}, Size: {s}, Type: {t}, Time: {tt}")
    # Write to file
    with open(f'generated_datasets/dataset_{dataset_num}_drift_events_{type}.txt', 'a') as f:
        f.write(f"Point: {p}, Size: {s}, Type: {t}, Time: {tt}\n")

print(f"\nGenerating Data for Graph #{dataset_num} ...")
df = pd.DataFrame(graph.generate(
    max_samples,
    intervention_prob=0,
    drift_points=drift_points,
    drift_sizes=drift_sizes,
    drift_types_time=drift_types_time,
    drift_types=drift_types,
    missing_prob=0
))

print(df.loc[:, 'y'].value_counts())

graph.save_graph(f"generated_datasets/graph_{dataset_num}_{type}_checkpoint.pkl")
graph.save_graph_to_json(f'generated_datasets/dataset_{dataset_num}_graph_{type}.json', seed=seed)
df.to_csv(f'generated_datasets/dataset_{dataset_num}_drifted_{type}.csv', index=False)
graph.save_drift_events_to_json(f'generated_datasets/dataset_{dataset_num}_drift_events_{type}.json')

stats = extract_graph_statistics(graph)

import json

# Save stats to JSON
with open(f'generated_datasets/dataset_{dataset_num}_stats_{type}.json', 'w') as f:
    json.dump(stats, f, indent=4)

# to arff
import csv_to_arff
csv_to_arff.csv_to_arff(f'generated_datasets/dataset_{dataset_num}_drifted_{type}.csv', f'generated_datasets/dataset_{dataset_num}_drifted_{type}.arff')