import time
import json
import datetime
import os

import RandomSCMGenerator as rsg
import pandas as pd


def run_benchmark(size: int, n_features: int = 10, n_roots: int = 5, max_parents: int = 3, task: str = 'classification', warmup: int = 0, output_csv: str = 'benchmark_results.csv') -> dict:
    # Build a random graph
    graph, roots, label_node = rsg.build_connected_dag(n_features, n_roots, max_parents, task=task)

    # train graph so mappers are initialized
    graph.train_graph(200)

    if warmup and warmup > 0:
        print(f"Running warmup generation of {warmup} samples...")
        _ = graph.generate(warmup, intervention_prob=0.0, drift_points=[], drift_sizes=[], drift_types=[], drift_types_time=[], missing_prob=0)

    print(f"Starting timed generation of {size} samples...")
    t0 = time.perf_counter()
    samples = graph.generate(size, intervention_prob=0, drift_points=[], drift_sizes=[], drift_types=[], drift_types_time=[], missing_prob=0)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    throughput = size / elapsed if elapsed > 0 else float('inf')

    # Summary of results
    result = {
        'n_features': n_features,
        'n_roots': n_roots,
        'max_parents': max_parents,
        'size': size,
        'elapsed_seconds': elapsed,
        'samples_per_second': throughput,
    }

    df = pd.DataFrame([result])
    if not os.path.exists(output_csv):
        df.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, mode='a', header=False, index=False)

    return result


def main():
    size = 500000
    n_features = 10
    n_roots = 3
    max_parents = 3
    warmup = 100
    out = 'benchmark_results.csv'

    print(f"Benchmark parameters: size={size}, n_features={n_features}, n_roots={n_roots}, max_parents={max_parents}")

    result = run_benchmark(size, n_features=n_features, n_roots=n_roots, max_parents=max_parents, warmup=warmup, output_csv=out)

    print("Benchmark finished:")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
