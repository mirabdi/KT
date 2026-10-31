"""ktbench: an extensible Knowledge Tracing benchmarking framework.

Layers (strictly decoupled):

* ``ktbench.data``      - ingestion adapters -> canonical interaction table -> sequences -> batches
* ``ktbench.models``    - model zoo behind a single ``BaseKTModel`` contract + registry
* ``ktbench.training``  - trainer, losses, metrics, memory management
* ``ktbench.config``    - typed, YAML-serialisable experiment configuration
* ``ktbench.experiment``- orchestration (single runs and model x dataset benchmarks)
"""

import os as _os

# Must be set before torch initialises the MPS backend: lets unsupported ops fall
# back to CPU instead of crashing on Apple Silicon.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

from ktbench.registry import Registry  # noqa: E402,F401
