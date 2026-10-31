"""Model zoo. Importing this package registers all built-in models.

To add a model, create a module here (or anywhere that is imported before the
experiment is built) and decorate the class with ``@MODEL_REGISTRY.register``.
"""

from ktbench.models.base import MODEL_REGISTRY, BaseKTModel, ModelOutput, build_model
from ktbench.models import akt, ckt, cogkt, dkt, dkvmn, sakt, saint, simplekt, skvmn  # noqa: F401  (registration side-effects)
from ktbench.models.akt import AKT, AKTConfig
from ktbench.models.ckt import CKT, CKTConfig
from ktbench.models.cogkt import CogKT, CogKTConfig
from ktbench.models.dkt import DKT, DKTConfig, DKTPlus, DKTPlusConfig
from ktbench.models.dkvmn import DKVMN, DKVMNConfig
from ktbench.models.sakt import SAKT, SAKTConfig
from ktbench.models.saint import SAINT, SAINTConfig
from ktbench.models.simplekt import SimpleKT, SimpleKTConfig
from ktbench.models.skvmn import SKVMN, SKVMNConfig

__all__ = [
    "MODEL_REGISTRY", "BaseKTModel", "ModelOutput", "build_model",
    "AKT", "AKTConfig", "CKT", "CKTConfig", "CogKT", "CogKTConfig", "DKT", "DKTConfig",
    "DKTPlus", "DKTPlusConfig", "DKVMN", "DKVMNConfig", "SAKT", "SAKTConfig",
    "SAINT", "SAINTConfig", "SimpleKT", "SimpleKTConfig", "SKVMN", "SKVMNConfig",
]
