# Knowledge Tracing Benchmark Results: Algebra2005 & NIPS34

Total benchmark runtime: 272.4s on Apple Silicon MPS (`mps`).

## Table 1: Dataset Statistics
| Dataset | Interactions | Learners (Sequences) | Concepts | Questions |
| --- | --- | --- | --- | --- |
| Algebra2005 | 884102 | 574 | 112 | 173113 |
| NIPS34 | 1399520 | 4918 | 57 | 948 |

## Table 2: Main Knowledge Tracing Benchmark (5-Fold Test AUC)
| Dataset | DKT | DKT+ | DKVMN | SAKT | AKT-NR | AKT-R |
|---|---|---|---|---|---|---|
| Algebra2005 | 0.9061 ± 0.0005 | 0.9065 ± 0.0004 | 0.9072 ± 0.0004 | 0.8627 ± 0.0012 | 0.9027 ± 0.0182 | **0.9167 ± 0.0094** |
| NIPS34 | 0.7587 ± 0.0003 | 0.7593 ± 0.0002 | 0.7598 ± 0.0002 | 0.7446 ± 0.0006 | 0.6633 ± 0.1197 | **0.6661 ± 0.0805** |

## Table 2b: Test RMSE (5-Fold Mean ± Std)
| Dataset | DKT | DKT+ | DKVMN | SAKT | AKT-NR | AKT-R |
|---|---|---|---|---|---|---|
| Algebra2005 | 0.3177 ± 0.0004 | 0.3174 ± 0.0003 | 0.3172 ± 0.0004 | 0.3519 ± 0.0006 | 0.3196 ± 0.0102 | **0.3101 ± 0.0063** |
| NIPS34 | 0.4450 ± 0.0004 | 0.4447 ± 0.0003 | 0.4445 ± 0.0003 | 0.4511 ± 0.0003 | 0.5004 ± 0.0515 | **0.5054 ± 0.0205** |

## Table 2c: Test Accuracy (5-Fold Mean ± Std)
| Dataset | DKT | DKT+ | DKVMN | SAKT | AKT-NR | AKT-R |
|---|---|---|---|---|---|---|
| Algebra2005 | 0.8595 ± 0.0005 | 0.8598 ± 0.0005 | 0.8605 ± 0.0007 | 0.8285 ± 0.0016 | 0.8590 ± 0.0056 | **0.8666 ± 0.0045** |
| NIPS34 | 0.6944 ± 0.0007 | 0.6948 ± 0.0005 | 0.6950 ± 0.0005 | 0.6810 ± 0.0005 | 0.6233 ± 0.0863 | **0.6270 ± 0.0494** |

## Table 3: Ablation 1 — Context-Aware Encoders vs Raw Embeddings (5-Fold Test AUC)
| Dataset | AKT^raw-NR | AKT-NR | AKT^raw-R | AKT-R |
|---|---|---|---|---|
| Algebra2005 | 0.9061 ± 0.0053 | 0.9027 ± 0.0182 | 0.9199 ± 0.0020 | **0.9167 ± 0.0094** |
| NIPS34 | 0.7490 ± 0.0541 | 0.6633 ± 0.1197 | 0.7740 ± 0.0128 | **0.6661 ± 0.0805** |

## Table 4: Ablation 2 — Monotonic Attention vs Positional Encoding (5-Fold Test AUC)
| Dataset | SAKT | AKT-NR^pos | AKT-NR^fixed | AKT-NR |
|---|---|---|---|---|
| Algebra2005 | 0.8627 ± 0.0012 | 0.8711 ± 0.0011 | 0.8714 ± 0.0011 | **0.9027 ± 0.0182** |
| NIPS34 | 0.7446 ± 0.0006 | 0.7501 ± 0.0005 | 0.7501 ± 0.0005 | **0.6633 ± 0.1197** |

## Table 5: Ablation 3 — Rasch Embeddings Across Models (5-Fold Test AUC)
| Dataset | DKT | DKT-R | DKT+ | DKT+-R | DKVMN | DKVMN-R | SAKT | SAKT-R | AKT-NR | AKT-R |
|---|---|---|---|---|---|---|---|---|---|---|
| Algebra2005 | 0.9061 ± 0.0005 | 0.9183 ± 0.0003 | 0.9065 ± 0.0004 | 0.9182 ± 0.0002 | 0.9072 ± 0.0004 | 0.9185 ± 0.0002 | 0.8627 ± 0.0012 | 0.8810 ± 0.0011 | 0.9027 ± 0.0182 | **0.9167 ± 0.0094** |
| NIPS34 | 0.7587 ± 0.0003 | 0.7886 ± 0.0005 | 0.7593 ± 0.0002 | 0.7889 ± 0.0004 | 0.7598 ± 0.0002 | 0.7891 ± 0.0004 | 0.7446 ± 0.0006 | 0.7804 ± 0.0010 | 0.6633 ± 0.1197 | **0.6661 ± 0.0805** |
