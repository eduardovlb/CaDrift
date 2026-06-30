# CaDrift: Causal Drift Generator
## CaDrift — dataset & synthetic SCM generator

CaDrift is a Python library to create synthetic datasets from causal structural causal models (SCMs) with controlled concept drift. It contains:

- A flexible causal-graph representation (`SCM/CausalGraph.py`).
- A set of mappers (generative functions) for nodes and label generation (`SCM/Mappers.py`).
- A set of target functions used by label mappers (`SCM/TargetFunctions.py`).
- Example scripts to create random or manual SCMs and to analyze outputs.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate a random dataset (CLI):

```bash
python RandomSCMGenerator.py
```

The default `RandomSCMGenerator.py` creates a random SCM (default 20 features) and writes:

- `data_sample.csv` — generated data
- `data_sample.json` / `data_sample.pkl` — saved graph
- `data_sample_stats.json` - graph statistics

3. Usage:

```python
from DAG_Utils import build_connected_dag

# build a graph for classification (or 'regression')
graph, roots, label_node = build_connected_dag(n_features=25, n_roots=5, max_parents=10, problem='classification')

# generate 1000 samples (generate will call training internally when needed)
samples = graph.generate(dataset_size=1000, intervention_prob=0.1,
												 drift_points=[1000,2000], drift_sizes=[500,1],
												 drift_types=['real','severe'], drift_types_time=['incremental','abrupt'],
												 missing_prob=0.1)

import pandas as pd
df = pd.DataFrame(samples)
df.to_csv('dataset.csv', index=False)
```

## Main API (useful functions & parameters)

- `build_connected_dag(n_features: int, n_roots: int, max_parents: int, problem: str='classification')`
	- Builds a random connected DAG and assigns mappers to vertices. `problem` can be `'classification'` or `'regression'`.

- `build_er_dag(n_features, p_edge=0.3, task='classification', ensure_label_parents=True, n_confounders=0, create_confounder=False, seed=42)`
	- Builds a random DAG using the Erdos-Renyi model.

- `CausalGraph.generate(dataset_size: int = 1000, intervention_prob: float = 0.05, drift_points: list = [], drift_sizes: list = [], drift_types: list = [], drift_types_time: list = [], missing_prob: float = 0.05)`
	- Generates samples from the SCM. If the graph is not trained, `generate` calls `train_graph()` automatically.
	- `drift_points` is a list of integer indexes where drift events occur.
	- `drift_sizes`, `drift_types`, `drift_types_time` control how and how big drifts are applied.

- `CausalGraph.train_graph(train_size: int = 100)`
	- Internally used to fit mappers that require training from parent->child samples.

- Drift control methods on `CausalGraph` (examples): `real_drift()`, `virtual_drift()`, `local_drift()`, `severe_drift()`, `recurrent_drift()`.

## Target functions (short overview)

These are the functions used by mappers to compute outputs from parent values. See `SCM/TargetFunctions.py` for full code.

- LinearFunction — linear combination of parent features. Good for regression-like continuous labels.
- PolynomialFunction — polynomial transform of inputs (degree >= 2).
- SineFunction — sums sin(X) plus small noise; useful to test non-linear periodic patterns.
- ThresholdFunction — boolean threshold on sum(X) with small noise; useful for binary labels.
- RadialBasisFunction — RBF-like kernel: exp(-||X||^2 / (2 sigma^2)).
- CheckerboardFunction — a piecewise / parity-like function (floor + modulo) to create complex discrete patterns.

These functions are simple and designed for synthetic experiments — see `TargetFunctions.py` if you want to add or customize functions.

## Mapper classes (high-level summary)

Mapper classes live in `SCM/Mappers.py`. A short map of the most relevant classes:

- Mapping (abstract) — base class for all mappers.
- IncrementalMapping (abstract) — base for mappers that support incremental drift / partial_fit.
- NormalMapper — root mapper that samples from a Normal distribution (supports EWMA-like dynamics and noise).
- UniformMapper — root mapper that samples uniformly from a range.
- RandomMLPMapper — small randomly initialized MLP used as a mapping; useful for non-linear behavior.
- MLPMapping — wrapper using sklearn-like MLP/online updates; supports incremental updates.
- TreeMapper — decision-tree-based mapper.
- SGDMapper — linear estimator with SGD; supports incremental drift via partial_fit.
- AbstractCategoricalMapper — base class for categorical/label mappers.
- PrototypeCategoricalMapper — k-prototype style mapper using nearest prototypes.
- OnlineGaussianCategoricalMapper — Gaussian prototype style mapper (online updates supported).
- RandomRBFCategoricalMapper — RBF-based categorical mapper.
- RotatingHyperplaneMapper — a rotating hyperplane classifier useful to test concept drift.

Refer to `SCM/Mappers.py` for implementation details and exact constructor parameters.

## Example: manual SCM (from `manual_SCM.py`)

`manual_SCM.py` contains a short example of building a small graph by hand, applying different drifts, and concatenating generated data segments. It demonstrates:

- constructing `Vertex` objects with explicit mappers (e.g., `NormalMapper`, `TreeMapper`, `SGDMapper`, `OnlineGaussianCategoricalMapper`)
- connecting vertices using `graph.add_edge(parent, child)`
- visualizing the graph using `graph.visualize_graph()`
- applying targeted drift by calling mapper methods such as `mapper.drift()`, `mapper.severe_drift()`, or via helper functions in the graph (e.g., `drift_node` in the example script)

- ## Citation

If you use this project in your research, please cite:

```bibtex
@misc{barboza2026cadrifttimedependentcausalgenerator,
      title={CaDrift: A Time-dependent Causal Generator of Drifting Data Streams}, 
      author={Eduardo V. L. Barboza and Jean Paul Barddal and Robert Sabourin and Rafael M. O. Cruz},
      year={2026},
      eprint={2602.20329},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.20329}
}
```
