# ktbench

A clean, modern Knowledge Tracing (KT) testbed in PyTorch. 

Mostly built to have a sane, working reimplementation of **AKT** ([Ghosh et al., KDD 2020](https://dl.acm.org/doi/10.1145/3394486.3403239)) along with classic baselines (**DKT**, **SAKT**), without having to fight 6-year-old unmaintained research code.

Runs smoothly on Apple Silicon (`mps`), Nvidia GPUs (`cuda`), or plain `cpu` without blowing up your RAM.

---

## Why this exists

If you've ever tried running original KT repos:
* They're often stuck on PyTorch 1.x or Python 3.7.
* Hardcoded paths and weird custom text file formats everywhere.
* Memory leaks / OOM errors on sequences longer than a few dozen steps.

**ktbench** wraps everything into a small, modular package:
* Modern PyTorch 2.x with clean typed configs (YAML + CLI overrides).
* Dynamic batching & length bucketing so attention doesn't eat all your memory.
* Verified against original AKT oracle outputs in `tests/test_akt_equivalence.py`.
* Works out of the box with synthetic data or standard datasets (ASSISTments, etc.).

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/your-username/ktbench.git
cd ktbench
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Make some fake data

If you just want to see it work without downloading big datasets first:

```bash
python -m ktbench synth --out data/synthetic --students 600 --concepts 20 --questions 400
```

### 3. Train a model

Train AKT-R on the synthetic data:

```bash
python -m ktbench train -c configs/akt_synthetic.yaml
```

You can tweak any config on the fly from the CLI:

```bash
python -m ktbench train -c configs/akt_synthetic.yaml train.epochs=10 train.lr=0.001 model.params.d_model=128
```

### 4. Run the full benchmark

Runs AKT, DKT, and SAKT side-by-side on the same splits:

```bash
python -m ktbench benchmark -c configs/benchmark_synthetic.yaml
```

Results get saved to `runs/benchmark_synthetic/results.csv`.

### 5. Run tests

```bash
pytest -v
```

---

## Models included

| Model | Paper | What it is |
| :--- | :--- | :--- |
| `akt` | [Ghosh et al. (KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403239) | Context-aware monotonic attention + Rasch item difficulty embeddings (supports AKT-R and AKT-NR). |
| `dkt` | [Piech et al. (NeurIPS 2015)](https://arxiv.org/abs/1506.05908) | The classic LSTM baseline. Simple, fast, surprisingly hard to beat. |
| `sakt` | [Pandey & Karypis (EDM 2019)](https://arxiv.org/abs/1907.06837) | Transformer decoder / self-attention over past interactions. |

See what's registered right now:
```bash
python -m ktbench list
```

---

## Quick benchmark results

Here's how they perform out of the box on synthetic student logs (evaluated on an Apple M3, `mps`):

| Dataset | Model | Test AUC | Test RMSE | Test Acc | Peak RAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `synthetic_rasch` (with Questions) | **AKT-R** | **0.7498** | **0.4504** | **68.8%** | ~0.9 GB |
| `synthetic_rasch` | AKT (no decay ablation) | 0.7461 | 0.4521 | 68.4% | ~0.5 GB |
| `synthetic_rasch` | DKT | 0.7448 | 0.4521 | 68.2% | ~0.1 GB |
| `synthetic_rasch` | SAKT | 0.7382 | 0.4561 | 67.8% | ~0.1 GB |
| `synthetic_concepts` (Concept-only) | **AKT-NR** | **0.7875** | **0.4339** | **71.9%** | ~0.9 GB |
| `synthetic_concepts` | DKT | 0.7812 | 0.4369 | 70.8% | ~0.1 GB |
| `synthetic_concepts` | SAKT | 0.7809 | 0.4373 | 71.1% | ~0.1 GB |

---

## Project layout

```text
ktbench/
├── config.py         # Config loader & CLI override logic
├── device.py         # Handles mps / cuda / cpu setup
├── registry.py       # Simple @register decorator for models & data adapters
├── experiment.py     # Runs training or multi-model benchmarks
├── cli.py            # Entry point (`python -m ktbench ...`)
├── data/             # Batching, dynamic padding, CSV/synthetic loaders
├── models/           # AKT, DKT, SAKT implementations
└── training/         # Trainer loop, loss, metrics, memory manager
```

---

## More details (if you actually need them)

* [THEORY_AKT.md](docs/THEORY_AKT.md) — Rasch model embeddings, forgetting curves, and math derivations if you're writing a paper or curious about the mechanics.
* [ARCHITECTURE_AND_DEVELOPER_GUIDE.md](docs/ARCHITECTURE_AND_DEVELOPER_GUIDE.md) — How the memory manager works and how to plug in your own models or custom datasets in ~30 lines of code.

---

## License

MIT
