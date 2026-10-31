import logging

import pytest
import torch

from ktbench.config import DataConfig
from ktbench.data import KTDataModule
from ktbench.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Small-but-real hyper-parameters per model so the suite stays fast.
SMALL_PARAMS = {
    "akt": {"d_model": 64, "d_ff": 128, "n_heads": 4, "final_fc_dim": 64, "final_fc_dim2": 32, "dropout": 0.0},
    "ckt": {"d_model": 32, "kernel_sizes": [3, 5], "n_layers": 1, "dropout": 0.0},
    "cogkt": {"d_model": 32, "d_ff": 64, "n_heads": 2, "conv_kernel_sizes": [3, 5], "conv_layers": 1, "final_fc_dim": 32, "final_fc_dim2": 16, "dropout": 0.0},
    "dkt": {"emb_dim": 32, "hidden_dim": 64, "dropout": 0.0},
    "dktplus": {"emb_dim": 32, "hidden_dim": 64, "dropout": 0.0, "lambda_r": 0.1, "lambda_w1": 0.1, "lambda_w2": 1.0},
    "dkvmn": {"memory_size": 16, "key_dim": 32, "value_dim": 64, "final_fc_dim": 32, "dropout": 0.0},
    "saint": {"d_model": 32, "n_heads": 2, "n_encoder_layers": 1, "n_decoder_layers": 1, "d_ff": 64, "dropout": 0.0},
    "sakt": {"d_model": 64, "n_heads": 4, "dropout": 0.0},
    "simplekt": {"d_model": 32, "n_heads": 2, "n_blocks": 1, "d_ff": 64, "dropout": 0.0},
    "skvmn": {"memory_size": 16, "key_dim": 32, "value_dim": 64, "hop_lstm_dim": 32, "final_fc_dim": 32, "dropout": 0.0},
}


@pytest.fixture(scope="session")
def device() -> torch.device:
    dev = resolve_device("auto")
    print(f"\n[ktbench tests] running on device: {dev}")
    return dev


def make_dm(use_questions: bool = True, n_students: int = 120, max_seq_len: int = 60, seed: int = 0) -> KTDataModule:
    cfg = DataConfig(
        adapter="synthetic",
        adapter_params={"n_students": n_students, "n_concepts": 12, "n_questions": 80,
                        "min_len": 10, "max_len": 120, "seed": seed, "use_questions": use_questions},
        max_seq_len=max_seq_len,
    )
    return KTDataModule(cfg, seed=seed).setup()


@pytest.fixture(scope="session")
def dm() -> KTDataModule:
    return make_dm()


@pytest.fixture(scope="session")
def dm_no_q() -> KTDataModule:
    return make_dm(use_questions=False)
