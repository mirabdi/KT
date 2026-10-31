# `ktbench` — Extensible Knowledge Tracing Framework & AKT Implementation

A modular, object-oriented **Knowledge Tracing (KT)** benchmarking framework built on **PyTorch 2.x**, featuring a numerically verified re-implementation of **Context-Aware Attentive Knowledge Tracing (AKT, KDD 2020)** alongside **DKT** and **SAKT**, engineered to run within an **11 GB unified memory budget on Apple Silicon (M3 `mps`)** (with automatic fallback to `cuda` and `cpu`).

---

## Documentation

* **[Theoretical Foundations of AKT (`docs/THEORY_AKT.md`)](docs/THEORY_AKT.md)** — Mathematical derivations and cognitive science grounding of AKT:
  * **Rasch Model-Based Embeddings (1PL IRT):** $\mathbf{x}_t = \mathbf{c}_{c_t} + \mu_{q_t}\mathbf{d}_{c_t}$ and empirical Bayes $L_2$ difficulty shrinkage.
  * **Context-Aware Monotonic Attention:** Ebbinghaus forgetting curves, stop-gradient contextual distance $d(t, \tau) = |t - \tau|\sum_{t'=\tau+1}^t \gamma_{t,t'}$, and per-head learnable exponential decay rates $\theta_h = \text{softplus}(\gamma_h)$.
  * **Dual-Encoder & Strictly Causal Knowledge Retriever:** How AKT contextualizes questions and past interactions without leaking current responses.
* **[Architecture & Developer Guide (`docs/ARCHITECTURE_AND_DEVELOPER_GUIDE.md`)](docs/ARCHITECTURE_AND_DEVELOPER_GUIDE.md)** — Detailed design rationale and step-by-step extension guides:
  * **4-Layer Decoupled Architecture:** `config` $\leftrightarrow$ `data` $\leftrightarrow$ `models` $\leftrightarrow$ `training`.
  * **Apple M3 (16 GB RAM / 11 GB GPU Cap) Optimization Strategy:** MPS memory fraction cap, sequence windowing, length-bucketed dynamic padding, probed cost-model micro-batching ($B \cdot L^p$), token-normalized gradient accumulation, and automatic OOM budget-halving recovery.
  * **How to Add a Custom KT Model** (`@MODEL_REGISTRY.register` + `BaseKTModel`) in ~30 lines of Python.
  * **How to Integrate a New Educational Dataset** (`csv` YAML mapping, `akt_legacy`, or custom `@ADAPTER_REGISTRY.register` subclass).

---

## Repository Structure

```text
.
├── configs/
│   ├── akt_synthetic.yaml           # Single-run AKT-R configuration
│   ├── akt_assist2009_legacy.yaml   # Paper reproduction config for original AKT files
│   └── benchmark_synthetic.yaml     # Head-to-head multi-model x multi-dataset benchmark
├── docs/
│   ├── THEORY_AKT.md                # Math, psychometrics & cognitive science of AKT
│   └── ARCHITECTURE_AND_DEVELOPER_GUIDE.md  # OOP design, M3 memory, extension tutorials
├── ktbench/
│   ├── config.py                    # Typed YAML dataclasses + dotted CLI overrides
│   ├── device.py                    # mps -> cuda -> cpu resolution & allocator controls
│   ├── registry.py                  # Generic plug-in registry for models & adapters
│   ├── experiment.py                # Single-run & multi-model benchmark orchestration
│   ├── cli.py                       # `python -m ktbench {train,benchmark,synth,list}`
│   ├── data/
│   │   ├── schema.py                # Canonical DataFrame contract & DatasetMeta
│   │   ├── synthetic.py             # IRT + Ebbinghaus forgetting-curve log simulator
│   │   ├── adapters.py              # CSV, AKTLegacy, and Synthetic adapters
│   │   ├── preprocessing.py         # 1-based Vocab, student-level split, windowing
│   │   ├── dataset.py               # Batch, dynamic padding, LengthBucketBatchSampler
│   │   └── datamodule.py            # KTDataModule unified data pipeline
│   ├── models/
│   │   ├── base.py                  # BaseKTModel & ModelOutput contract
│   │   ├── akt.py                   # AKT / AKT-R / AKT-NR (Ghosh et al., KDD 2020)
│   │   ├── dkt.py                   # DKT (Piech et al., NeurIPS 2015)
│   │   └── sakt.py                  # SAKT (Pandey & Karypis, EDM 2019)
│   └── training/
│       ├── losses.py                # Token-summed masked BCE + model regularization
│       ├── metrics.py               # Streaming CPU accumulator (AUC, RMSE, ACC, MAE, BCE)
│       ├── memory.py                # Cost calibration, dynamic micro-batching, OOM retry
│       └── trainer.py               # Model-agnostic Trainer
└── tests/
    ├── reference/akt_legacy.py      # Vendored KDD 2020 AKT oracle for equivalence testing
    ├── test_akt_equivalence.py      # Forward/backward numerical parity vs legacy AKT
    ├── test_data.py                 # Schema, splits, windowing, adapters, bucketing
    ├── test_models.py               # Causality (no future leakage), padding invariance, plugins
    └── test_memory_and_training.py  # MPS detection, micro-batch grad equivalence, OOM, E2E
```

---

## Quickstart

### 1. Environment Setup (Apple Silicon M3 / PyTorch 2.x)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Inspect Available Models and Dataset Adapters

```bash
python -m ktbench list
# models:   akt, dkt, sakt
# adapters: akt_legacy, csv, synthetic
```

### 3. Generate Synthetic Student Interaction Logs

```bash
python -m ktbench synth --out data/synthetic --students 600 --concepts 20 --questions 400 --max-len 200
```

### 4. Train & Evaluate AKT on Apple Silicon MPS

```bash
python -m ktbench train -c configs/akt_synthetic.yaml
# Any config key can be overridden via dotted CLI arguments:
python -m ktbench train -c configs/akt_synthetic.yaml train.epochs=10 model.params.d_model=128
```

### 5. Run a Head-to-Head Multi-Model Benchmark

```bash
python -m ktbench benchmark -c configs/benchmark_synthetic.yaml
```

### 6. Run the Verification Test Suite

```bash
pytest -v
```

---

## Proof-of-Work Benchmark on Apple M3 (`mps`, 11 GB Cap)

Running `python -m ktbench benchmark -c configs/benchmark_synthetic.yaml` compares **AKT-R**, **AKT (no monotonic decay ablation)**, **DKT**, and **SAKT** head-to-head on identical student splits (`runs/benchmark_synthetic/results.csv`):

| Dataset | Model | Test AUC $\uparrow$ | Test RMSE $\downarrow$ | Test ACC $\uparrow$ | Peak MPS Memory |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `synthetic_rasch` (with Question IDs) | **`akt` (AKT-R)** | **0.7498** | **0.4504** | **0.6878** | 0.93 GB / 11.0 GB |
| `synthetic_rasch` (with Question IDs) | `akt_no_mono` (Ablation) | 0.7461 | 0.4521 | 0.6835 | 0.47 GB / 11.0 GB |
| `synthetic_rasch` (with Question IDs) | `dkt` | 0.7448 | 0.4521 | 0.6822 | 0.05 GB / 11.0 GB |
| `synthetic_rasch` (with Question IDs) | `sakt` | 0.7382 | 0.4561 | 0.6778 | 0.14 GB / 11.0 GB |
| `synthetic_concepts` (Concept-Only) | **`akt` (AKT-NR)** | **0.7875** | **0.4339** | **0.7193** | 0.93 GB / 11.0 GB |
| `synthetic_concepts` (Concept-Only) | `akt_no_mono` (Ablation) | 0.7871 | 0.4338 | 0.7140 | 0.46 GB / 11.0 GB |
| `synthetic_concepts` (Concept-Only) | `dkt` | 0.7812 | 0.4369 | 0.7081 | 0.05 GB / 11.0 GB |
| `synthetic_concepts` (Concept-Only) | `sakt` | 0.7809 | 0.4373 | 0.7114 | 0.14 GB / 11.0 GB |

