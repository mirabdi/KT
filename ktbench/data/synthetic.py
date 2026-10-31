"""Synthetic student-interaction generator.

The simulator is deliberately grounded in the same cognitive assumptions AKT
encodes, so that the synthetic benchmark has *learnable* structure:

* **Rasch / 1PL IRT item difficulty** - ``b_q = beta_c + mu_q``: each question
  is a perturbation of its concept's difficulty (exactly AKT's Rasch embedding
  assumption ``x_t = c_{c_t} + mu_{q_t} * d_{c_t}``).
* **Learning** - every practice of a concept increases mastery, more so when
  the answer was correct (learning from success / feedback).
* **Forgetting** - mastery decays exponentially with the time since the concept
  was last practised (Ebbinghaus forgetting curve, ``R = exp(-dt / S)``), and
  memory stability ``S`` grows with repetitions (spacing effect). This is the
  signal AKT's monotonic (exponentially-decaying) attention is designed for.
* **Student ability** - a per-student latent trait ``theta_s``.

    P(r_t = 1) = sigmoid(theta_s + m_{s,c}(t) - b_q)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ktbench.data import schema as S


@dataclass
class SyntheticConfig:
    n_students: int = 500
    n_concepts: int = 20
    n_questions: int = 400
    min_len: int = 20
    max_len: int = 200
    seed: int = 0
    question_noise: float = 0.6
    """Std of the per-question difficulty deviation ``mu_q``."""
    learn_rate: float = 0.35
    max_mastery: float = 3.0
    base_stability: float = 2.0 * 3600
    """Forgetting time-constant (seconds) after the first exposure."""
    p_stay_on_concept: float = 0.65
    """Probability the next interaction stays on the current concept (runs)."""
    mean_gap: float = 60.0
    """Mean seconds between interactions inside a session."""
    p_new_session: float = 0.05
    """Probability of a long break (hours to days) before the next interaction."""


def generate_interactions(cfg: SyntheticConfig | None = None, **overrides) -> pd.DataFrame:
    """Simulate interaction logs in the canonical long format."""
    cfg = cfg or SyntheticConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if cfg.n_questions < cfg.n_concepts:
        raise ValueError("n_questions must be >= n_concepts")
    rng = np.random.default_rng(cfg.seed)

    # ---- item bank (Rasch structure) ----------------------------------------
    concept_difficulty = rng.normal(0.0, 1.0, cfg.n_concepts)
    # every concept owns at least one question, the rest are assigned randomly
    q_concept = np.concatenate(
        [np.arange(cfg.n_concepts), rng.integers(0, cfg.n_concepts, cfg.n_questions - cfg.n_concepts)]
    )
    rng.shuffle(q_concept)
    q_difficulty = concept_difficulty[q_concept] + rng.normal(0.0, cfg.question_noise, cfg.n_questions)
    questions_of = [np.flatnonzero(q_concept == c) for c in range(cfg.n_concepts)]

    rows = []
    for s in range(cfg.n_students):
        theta = rng.normal(0.0, 1.0)
        lr = cfg.learn_rate * rng.lognormal(0.0, 0.3)
        mastery = np.full(cfg.n_concepts, -1.0)
        last_seen = np.full(cfg.n_concepts, np.nan)
        reps = np.zeros(cfg.n_concepts)
        t = float(rng.uniform(0, 86400))
        length = int(rng.integers(cfg.min_len, cfg.max_len + 1))
        c = int(rng.integers(cfg.n_concepts))
        for _ in range(length):
            if rng.random() > cfg.p_stay_on_concept:
                c = int(rng.integers(cfg.n_concepts))
            q = int(rng.choice(questions_of[c]))
            # forgetting since last exposure of this concept
            if not np.isnan(last_seen[c]):
                stability = cfg.base_stability * (1.0 + reps[c])
                retention = np.exp(-(t - last_seen[c]) / stability)
                mastery[c] = -1.0 + (mastery[c] + 1.0) * retention
            p = 1.0 / (1.0 + np.exp(-(theta + mastery[c] - q_difficulty[q])))
            r = int(rng.random() < p)
            rows.append((s, q, c, r, t))
            # learning update (bounded growth, stronger after success)
            gain = lr * (1.0 + 0.5 * r) * (cfg.max_mastery - mastery[c]) / (cfg.max_mastery + 1.0)
            mastery[c] += gain
            reps[c] += 1
            last_seen[c] = t
            if rng.random() < cfg.p_new_session:
                t += float(rng.uniform(3 * 3600, 3 * 86400))
            else:
                t += float(rng.exponential(cfg.mean_gap)) + 1.0

    df = pd.DataFrame(rows, columns=[S.USER, S.QUESTION, S.CONCEPT, S.CORRECT, S.TIMESTAMP])
    return df


def write_canonical_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_akt_legacy(df: pd.DataFrame, path: str | Path, with_questions: bool = True) -> Path:
    """Export to the AKT/DKVMN line-based format (1-based ids, 0 = padding).

    With questions (4 lines / student)::

        <n_interactions>,<user_id>
        q_1,q_2,...        (problem ids)
        c_1,c_2,...        (concept ids)
        r_1,r_2,...

    Without questions the problem line is omitted (3 lines / student).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [S.USER] + ([S.TIMESTAMP] if S.TIMESTAMP in df.columns else [])
    df = df.sort_values(order, kind="stable")
    c_codes = pd.factorize(df[S.CONCEPT], sort=True)[0] + 1
    q_codes = pd.factorize(df[S.QUESTION], sort=True)[0] + 1 if with_questions else None
    df = df.assign(_c=c_codes, **({"_q": q_codes} if with_questions else {}))
    with open(path, "w") as f:
        for uid, g in df.groupby(S.USER, sort=False):
            f.write(f"{len(g)},{uid}\n")
            if with_questions:
                f.write(",".join(map(str, g["_q"].tolist())) + "\n")
            f.write(",".join(map(str, g["_c"].tolist())) + "\n")
            f.write(",".join(map(str, g[S.CORRECT].tolist())) + "\n")
    return path
