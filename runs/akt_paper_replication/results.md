# KDD 2020 AKT Paper Replication Results

Total benchmark runtime: 429.0s on Apple Silicon MPS (`mps`).

## Table 2: Main Knowledge Tracing Benchmark (5-Fold Test AUC)
| Dataset | DKT | DKT+ | DKVMN | SAKT | AKT-NR | AKT-R |
|---|---|---|---|---|---|---|
| Statics2011 | 0.8233 ± 0.0044 | 0.8294 ± 0.0042 | 0.8215 ± 0.0044 | 0.8015 ± 0.0051 | 0.8266 ± 0.0042 | - |
| ASSISTments2009 | 0.8176 ± 0.0039 | 0.8109 ± 0.0040 | 0.8093 ± 0.0038 | 0.7526 ± 0.0067 | 0.8175 ± 0.0039 | 0.8361 ± 0.0041 |
| ASSISTments2015 | 0.7317 ± 0.0023 | 0.7322 ± 0.0023 | 0.7273 ± 0.0024 | 0.7227 ± 0.0018 | 0.7834 ± 0.0023 | - |
| ASSISTments2017 | 0.7276 ± 0.0052 | 0.7135 ± 0.0045 | 0.7079 ± 0.0042 | 0.6592 ± 0.0044 | 0.7295 ± 0.0053 | 0.7672 ± 0.0035 |

## Table 3: Ablation 1 — Context-Aware Encoders vs Raw Embeddings (5-Fold Test AUC)
| Dataset | AKT^raw-NR | AKT-NR | AKT^raw-R | AKT-R |
|---|---|---|---|---|
| Statics2011 | 0.8250 ± 0.0043 | 0.8266 ± 0.0042 | - | - |
| ASSISTments2009 | 0.8101 ± 0.0039 | 0.8175 ± 0.0039 | 0.8278 ± 0.0043 | 0.8361 ± 0.0041 |
| ASSISTments2015 | 0.7338 ± 0.0023 | 0.7834 ± 0.0023 | - | - |
| ASSISTments2017 | 0.7074 ± 0.0042 | 0.7295 ± 0.0053 | 0.7614 ± 0.0039 | 0.7672 ± 0.0035 |

## Table 4: Ablation 2 — Monotonic Attention vs Positional Encoding (5-Fold Test AUC)
| Dataset | SAKT | AKT-NR^pos | AKT-NR^fixed | AKT-NR |
|---|---|---|---|---|
| Statics2011 | 0.8015 ± 0.0051 | 0.8252 ± 0.0037 | 0.8252 ± 0.0037 | 0.8266 ± 0.0042 |
| ASSISTments2009 | 0.7526 ± 0.0067 | 0.7764 ± 0.0067 | 0.7768 ± 0.0067 | 0.8175 ± 0.0039 |
| ASSISTments2015 | 0.7227 ± 0.0018 | 0.7271 ± 0.0013 | 0.7272 ± 0.0013 | 0.7834 ± 0.0023 |
| ASSISTments2017 | 0.6592 ± 0.0044 | 0.6757 ± 0.0045 | 0.6758 ± 0.0045 | 0.7295 ± 0.0053 |

## Table 5: Ablation 3 — Rasch Embeddings Across Models (5-Fold Test AUC)
| Dataset | DKT | DKT-R | DKT+ | DKT+-R | DKVMN | DKVMN-R | SAKT | SAKT-R | AKT-NR | AKT-R |
|---|---|---|---|---|---|---|---|---|---|---|
| ASSISTments2009 | 0.8176 ± 0.0039 | 0.8189 ± 0.0040 | 0.8109 ± 0.0040 | 0.8123 ± 0.0040 | 0.8093 ± 0.0038 | 0.8247 ± 0.0043 | 0.7526 ± 0.0067 | 0.7750 ± 0.0070 | 0.8175 ± 0.0039 | 0.8361 ± 0.0041 |
| ASSISTments2017 | 0.7276 ± 0.0052 | 0.7606 ± 0.0039 | 0.7135 ± 0.0045 | 0.7477 ± 0.0037 | 0.7079 ± 0.0042 | 0.7572 ± 0.0031 | 0.6592 ± 0.0044 | 0.7067 ± 0.0034 | 0.7295 ± 0.0053 | 0.7672 ± 0.0035 |
