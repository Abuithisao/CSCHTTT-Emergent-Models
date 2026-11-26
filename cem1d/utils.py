# utils.py
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from general_ga import GA, GAConfig


CHECKPOINT_SCHEMA = "cem1d_cartpole_lamarckian_fixed_dt_v1"


# =============================================================================
# Config
# =============================================================================


@dataclass
class EMCartPoleConfig:
    # Continuous 1D EM substrate
    L: int = 19
    T: int = 30
    dt: float = 0.1
    gamma: float = 0.03
    invalid_threshold: float = 50.0
    boundary_mode: str = "wrap"  # "wrap" | "reflect" | "zero"

    # Fixed interface
    input_cells: tuple[int, int, int, int] = (3, 6, 12, 15)
    output_cell: int = 9
    threshold: float = 0.04

    # CartPole
    F_max: float = 10.0
    max_steps: int = 800

    g: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    tau: float = 0.02

    x_threshold: float = 2.4
    theta_threshold: float = 12.0 * np.pi / 180.0

    # Observation normalization
    norm_x: float = 1.0
    norm_dx: float = 2.0
    norm_theta: float = 8.0 * np.pi / 180.0
    norm_dtheta: float = 3.0

    # Initial CartPole state distribution
    init_mode: str = "boundary_mixture"  # "uniform" | "boundary_mixture"
    init_x_range: float = 1.5
    init_dx_range: float = 0.1
    init_theta_range: float = 0.1
    init_dtheta_range: float = 0.1

    boundary_prob: float = 0.8
    boundary_x_low: float = 1.8
    boundary_x_high: float = 2.2
    center_x_range: float = 1.5

    # Fitness
    fitness_std_penalty: float = 0.02

    # Evaluation
    train_rollouts: int = 100
    eval_rollouts: int = 500


def validate_config(cfg: EMCartPoleConfig) -> None:
    if cfg.L <= 2:
        raise ValueError("cfg.L must be > 2.")
    if cfg.T <= 0:
        raise ValueError("cfg.T must be > 0.")
    if cfg.dt <= 0:
        raise ValueError("cfg.dt must be > 0.")
    if cfg.gamma < 0:
        raise ValueError("cfg.gamma must be >= 0.")
    if cfg.invalid_threshold <= 0:
        raise ValueError("cfg.invalid_threshold must be > 0.")
    if cfg.boundary_mode not in ("wrap", "reflect", "zero"):
        raise ValueError("cfg.boundary_mode must be 'wrap', 'reflect', or 'zero'.")

    if len(cfg.input_cells) != 4:
        raise ValueError("cfg.input_cells must contain exactly 4 cells.")
    for cell in cfg.input_cells:
        if not (0 <= int(cell) < cfg.L):
            raise ValueError(f"input cell {cell} is outside [0, {cfg.L - 1}].")

    if not (0 <= int(cfg.output_cell) < cfg.L):
        raise ValueError(f"output_cell {cfg.output_cell} is outside [0, {cfg.L - 1}].")
    if cfg.threshold < 0:
        raise ValueError("cfg.threshold must be >= 0.")

    if cfg.F_max <= 0:
        raise ValueError("cfg.F_max must be > 0.")
    if cfg.max_steps <= 0:
        raise ValueError("cfg.max_steps must be > 0.")

    if cfg.g <= 0:
        raise ValueError("cfg.g must be > 0.")
    if cfg.masscart <= 0:
        raise ValueError("cfg.masscart must be > 0.")
    if cfg.masspole <= 0:
        raise ValueError("cfg.masspole must be > 0.")
    if cfg.length <= 0:
        raise ValueError("cfg.length must be > 0.")
    if cfg.tau <= 0:
        raise ValueError("cfg.tau must be > 0.")

    if cfg.x_threshold <= 0:
        raise ValueError("cfg.x_threshold must be > 0.")
    if cfg.theta_threshold <= 0:
        raise ValueError("cfg.theta_threshold must be > 0.")

    if cfg.norm_x <= 0 or cfg.norm_dx <= 0 or cfg.norm_theta <= 0 or cfg.norm_dtheta <= 0:
        raise ValueError("normalization constants must be > 0.")

    if cfg.init_mode not in ("uniform", "boundary_mixture"):
        raise ValueError("cfg.init_mode must be 'uniform' or 'boundary_mixture'.")
    if not (0.0 <= cfg.init_x_range < cfg.x_threshold):
        raise ValueError("cfg.init_x_range must be in [0, cfg.x_threshold).")
    if cfg.init_dx_range < 0:
        raise ValueError("cfg.init_dx_range must be >= 0.")
    if not (0.0 <= cfg.init_theta_range < cfg.theta_threshold):
        raise ValueError("cfg.init_theta_range must be in [0, cfg.theta_threshold).")
    if cfg.init_dtheta_range < 0:
        raise ValueError("cfg.init_dtheta_range must be >= 0.")

    if not (0.0 <= cfg.boundary_prob <= 1.0):
        raise ValueError("cfg.boundary_prob must be in [0, 1].")
    if not (0.0 <= cfg.center_x_range < cfg.x_threshold):
        raise ValueError("cfg.center_x_range must be in [0, cfg.x_threshold).")
    if not (0.0 <= cfg.boundary_x_low < cfg.boundary_x_high < cfg.x_threshold):
        raise ValueError("boundary_x range must lie inside [0, cfg.x_threshold).")

    if cfg.fitness_std_penalty < 0:
        raise ValueError("cfg.fitness_std_penalty must be >= 0.")
    if cfg.train_rollouts <= 0:
        raise ValueError("cfg.train_rollouts must be > 0.")
    if cfg.eval_rollouts <= 0:
        raise ValueError("cfg.eval_rollouts must be > 0.")


def validate_population(pop: dict[str, np.ndarray], cfg: EMCartPoleConfig, A: int | None = None) -> int:
    validate_config(cfg)

    if "a" not in pop:
        raise KeyError("pop is missing key 'a'.")
    if "s" not in pop:
        raise KeyError("pop is missing key 's'.")

    a = np.asarray(pop["a"], dtype=np.float32)
    s = np.asarray(pop["s"], dtype=np.float32)

    if s.ndim != 2 or s.shape[1] != cfg.L:
        raise ValueError(f"pop['s'] must have shape (A, {cfg.L}), got {s.shape}.")

    inferred_A = int(s.shape[0])
    if A is not None and inferred_A != int(A):
        raise ValueError(f"expected population size {A}, got {inferred_A}.")

    if a.shape != (inferred_A, 3):
        raise ValueError(f"pop['a'] must have shape ({inferred_A}, 3), got {a.shape}.")

    if not np.isfinite(a).all():
        raise ValueError("pop['a'] contains NaN or infinite values.")
    if not np.isfinite(s).all():
        raise ValueError("pop['s'] contains NaN or infinite values.")
    if np.max(np.abs(s)) > float(cfg.invalid_threshold):
        raise ValueError("pop['s'] contains values outside cfg.invalid_threshold.")

    return inferred_A


# =============================================================================
# Checkpoint / small helpers
# =============================================================================


def install_population_in_ga(
    ga: GA,
    pop: dict[str, np.ndarray],
    *,
    generation: int = 0,
    clear_bank: bool = True,
) -> dict[str, np.ndarray]:
    if not pop:
        raise ValueError("pop must be non-empty.")

    ga.pop_cur = {k: np.asarray(v).copy() for k, v in pop.items()}
    ga.pop_next = {k: np.empty_like(v) for k, v in ga.pop_cur.items()}
    ga.generation = int(generation)

    if clear_bank:
        ga.bank.clear()

    return ga.pop_cur


def save_training_checkpoint(
    path: str | Path,
    *,
    pop: dict[str, np.ndarray],
    cfg: EMCartPoleConfig,
    ga_cfg: GAConfig,
    history: list[dict[str, Any]],
    next_gen: int,
    completed_gen: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    validate_population(pop, cfg)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "schema": np.asarray(CHECKPOINT_SCHEMA),
        "next_gen": np.asarray(int(next_gen), dtype=np.int32),
        "completed_gen": np.asarray(int(completed_gen), dtype=np.int32),
        "cfg_json": np.asarray(_json_dumps(cfg)),
        "ga_cfg_json": np.asarray(_json_dumps(ga_cfg)),
        "history_json": np.asarray(_json_dumps(history)),
        "metadata_json": np.asarray(_json_dumps(metadata or {})),
    }

    for key, value in pop.items():
        arrays[f"pop__{key}"] = np.asarray(value)

    np.savez_compressed(path, **arrays)
    return path


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with np.load(path, allow_pickle=False) as z:
        schema = str(z["schema"].item())
        if schema != CHECKPOINT_SCHEMA:
            raise ValueError(f"unexpected checkpoint schema: {schema!r}")

        cfg = EMCartPoleConfig(**_json_load_0d(z["cfg_json"]))
        ga_cfg = GAConfig(**_json_load_0d(z["ga_cfg_json"]))
        pop = _load_pop(z)

        validate_population(pop, cfg)

        return {
            "path": path,
            "schema": schema,
            "next_gen": int(z["next_gen"].item()),
            "completed_gen": int(z["completed_gen"].item()),
            "cfg": cfg,
            "ga_cfg": ga_cfg,
            "history": _json_load_0d(z["history_json"]),
            "metadata": _json_load_0d(z["metadata_json"]),
            "pop": pop,
        }


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    keys: list[str] = []
    for row in history:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(history)


def top_percentile_mean(values: np.ndarray, pct: float = 0.05) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if values.size == 0:
        raise ValueError("values must be non-empty.")
    if not (0.0 < pct <= 1.0):
        raise ValueError("pct must be in (0, 1].")

    k = max(1, int(np.ceil(pct * values.size)))
    part = np.partition(values, values.size - k)
    return float(part[values.size - k:].mean())


# =============================================================================
# CartPole
# =============================================================================


def init_cartpole_states(n: int, cfg: EMCartPoleConfig, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be > 0.")

    if cfg.init_mode == "uniform":
        x = rng.uniform(-cfg.init_x_range, cfg.init_x_range, size=n).astype(np.float32)

    elif cfg.init_mode == "boundary_mixture":
        x = np.empty(n, dtype=np.float32)
        near_boundary = rng.random(n) < cfg.boundary_prob

        n_boundary = int(near_boundary.sum())
        n_center = n - n_boundary

        if n_center:
            x[~near_boundary] = rng.uniform(
                -cfg.center_x_range,
                cfg.center_x_range,
                size=n_center,
            ).astype(np.float32)

        if n_boundary:
            side = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n_boundary)
            mag = rng.uniform(cfg.boundary_x_low, cfg.boundary_x_high, size=n_boundary).astype(np.float32)
            x[near_boundary] = side * mag

    else:
        raise ValueError(f"unsupported init_mode: {cfg.init_mode}")

    state = np.stack(
        [
            x,
            rng.uniform(-cfg.init_dx_range, cfg.init_dx_range, size=n),
            rng.uniform(-cfg.init_theta_range, cfg.init_theta_range, size=n),
            rng.uniform(-cfg.init_dtheta_range, cfg.init_dtheta_range, size=n),
        ],
        axis=1,
    )

    return state.astype(np.float32)


def normalize_obs(state: np.ndarray, cfg: EMCartPoleConfig) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)

    if state.ndim == 1:
        if state.shape != (4,):
            raise ValueError(f"state must have shape (4,), got {state.shape}.")

        obs = np.empty(4, dtype=np.float32)
        obs[0] = state[0] / cfg.norm_x
        obs[1] = state[1] / cfg.norm_dx
        obs[2] = state[2] / cfg.norm_theta
        obs[3] = state[3] / cfg.norm_dtheta
        return np.clip(obs, -4.0, 4.0).astype(np.float32)

    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(f"state must have shape (A, 4), got {state.shape}.")

    obs = np.empty_like(state, dtype=np.float32)
    obs[:, 0] = state[:, 0] / cfg.norm_x
    obs[:, 1] = state[:, 1] / cfg.norm_dx
    obs[:, 2] = state[:, 2] / cfg.norm_theta
    obs[:, 3] = state[:, 3] / cfg.norm_dtheta
    return np.clip(obs, -4.0, 4.0).astype(np.float32)


def cartpole_step(state: np.ndarray, force: np.ndarray, cfg: EMCartPoleConfig) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    force = np.asarray(force, dtype=np.float32)

    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(f"state must have shape (A, 4), got {state.shape}.")
    if force.shape != (state.shape[0],):
        raise ValueError(f"force must have shape ({state.shape[0]},), got {force.shape}.")

    x = state[:, 0]
    x_dot = state[:, 1]
    theta = state[:, 2]
    theta_dot = state[:, 3]

    total_mass = cfg.masscart + cfg.masspole
    polemass_length = cfg.masspole * cfg.length

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass

    thetaacc = (cfg.g * sintheta - costheta * temp) / (
        cfg.length * (4.0 / 3.0 - cfg.masspole * costheta * costheta / total_mass)
    )

    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    next_state = np.stack(
        [
            x + cfg.tau * x_dot,
            x_dot + cfg.tau * xacc,
            theta + cfg.tau * theta_dot,
            theta_dot + cfg.tau * thetaacc,
        ],
        axis=1,
    ).astype(np.float32)

    done = (
        (np.abs(next_state[:, 0]) > cfg.x_threshold)
        | (np.abs(next_state[:, 2]) > cfg.theta_threshold)
        | (~np.isfinite(next_state).all(axis=1))
    )

    return next_state, done.astype(bool)


def cartpole_step_one(state: np.ndarray, force: float, cfg: EMCartPoleConfig) -> tuple[np.ndarray, bool]:
    next_state, done = cartpole_step(
        state=np.asarray(state, dtype=np.float32)[None, :],
        force=np.asarray([force], dtype=np.float32),
        cfg=cfg,
    )
    return next_state[0], bool(done[0])


# =============================================================================
# CEM1D dynamics
# =============================================================================


def valid_internal_state(s: np.ndarray, cfg: EMCartPoleConfig) -> np.ndarray | bool:
    s = np.asarray(s)

    if s.ndim == 1:
        return bool(np.isfinite(s).all() and np.max(np.abs(s)) <= float(cfg.invalid_threshold))

    if s.ndim != 2:
        raise ValueError(f"s must have shape (L,) or (A, L), got {s.shape}.")

    if s.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    return np.isfinite(s).all(axis=1) & (np.max(np.abs(s), axis=1) <= float(cfg.invalid_threshold))


def neighbor_arrays_batch(s: np.ndarray, cfg: EMCartPoleConfig) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.float32)

    if s.ndim != 2 or s.shape[1] != cfg.L:
        raise ValueError(f"s must have shape (A, {cfg.L}), got {s.shape}.")

    left = np.empty_like(s)
    right = np.empty_like(s)

    left[:, 1:] = s[:, :-1]
    right[:, :-1] = s[:, 1:]

    if cfg.boundary_mode == "wrap":
        left[:, 0] = s[:, -1]
        right[:, -1] = s[:, 0]
    elif cfg.boundary_mode == "reflect":
        left[:, 0] = s[:, 0]
        right[:, -1] = s[:, -1]
    elif cfg.boundary_mode == "zero":
        left[:, 0] = 0.0
        right[:, -1] = 0.0
    else:
        raise ValueError(f"unsupported boundary_mode: {cfg.boundary_mode}")

    return left, right


def neighbor_arrays_one(s: np.ndarray, cfg: EMCartPoleConfig) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.float32)

    if s.shape != (cfg.L,):
        raise ValueError(f"s must have shape ({cfg.L},), got {s.shape}.")

    left = np.empty_like(s)
    right = np.empty_like(s)

    left[1:] = s[:-1]
    right[:-1] = s[1:]

    if cfg.boundary_mode == "wrap":
        left[0] = s[-1]
        right[-1] = s[0]
    elif cfg.boundary_mode == "reflect":
        left[0] = s[0]
        right[-1] = s[-1]
    elif cfg.boundary_mode == "zero":
        left[0] = 0.0
        right[-1] = 0.0
    else:
        raise ValueError(f"unsupported boundary_mode: {cfg.boundary_mode}")

    return left, right


def ode_rollout_batch(s: np.ndarray, a: np.ndarray, cfg: EMCartPoleConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs fixed-T internal dynamics.

    Invalid candidate states are not committed.
    Rows that become invalid are marked invalid and frozen at their last valid state.
    """
    s = np.asarray(s, dtype=np.float32).copy()
    a = np.asarray(a, dtype=np.float32)

    if s.ndim != 2 or s.shape[1] != cfg.L:
        raise ValueError(f"s must have shape (A, {cfg.L}), got {s.shape}.")
    if a.shape != (s.shape[0], 3):
        raise ValueError(f"a must have shape ({s.shape[0]}, 3), got {a.shape}.")

    valid = np.asarray(valid_internal_state(s, cfg), dtype=bool)
    executed = np.zeros(s.shape[0], dtype=np.float32)

    for _ in range(int(cfg.T)):
        idx = np.flatnonzero(valid)

        if idx.size == 0:
            break

        sv = s[idx]
        av = a[idx]
        left, right = neighbor_arrays_batch(sv, cfg)

        with np.errstate(over="ignore", invalid="ignore"):
            ds = (
                av[:, 0:1] * left
                + av[:, 1:2] * sv
                + av[:, 2:3] * right
                - np.float32(cfg.gamma) * (sv ** 3)
            )
            candidate = sv + np.float32(cfg.dt) * ds

        candidate_valid = np.asarray(valid_internal_state(candidate, cfg), dtype=bool)

        if candidate_valid.any():
            commit_idx = idx[candidate_valid]
            s[commit_idx] = candidate[candidate_valid]
            executed[commit_idx] += 1.0

        if (~candidate_valid).any():
            valid[idx[~candidate_valid]] = False

    return s.astype(np.float32), valid.astype(bool), executed.astype(np.float32)


def ode_step_one(s: np.ndarray, a: np.ndarray, cfg: EMCartPoleConfig) -> np.ndarray:
    s = np.asarray(s, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)

    if s.shape != (cfg.L,):
        raise ValueError(f"s must have shape ({cfg.L},), got {s.shape}.")
    if a.shape != (3,):
        raise ValueError(f"a must have shape (3,), got {a.shape}.")

    left, right = neighbor_arrays_one(s, cfg)

    with np.errstate(over="ignore", invalid="ignore"):
        ds = a[0] * left + a[1] * s + a[2] * right - np.float32(cfg.gamma) * (s ** 3)
        return (s + np.float32(cfg.dt) * ds).astype(np.float32)


def decode_actions(s: np.ndarray, cfg: EMCartPoleConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.float32)

    if s.ndim != 2 or s.shape[1] != cfg.L:
        raise ValueError(f"s must have shape (A, {cfg.L}), got {s.shape}.")

    z = s[:, int(cfg.output_cell)]
    random_mask = np.abs(z) <= float(cfg.threshold)

    action = np.empty(s.shape[0], dtype=np.float32)
    action[z < -float(cfg.threshold)] = -1.0
    action[z > float(cfg.threshold)] = 1.0

    n_random = int(random_mask.sum())
    if n_random:
        action[random_mask] = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n_random)

    return (float(cfg.F_max) * action).astype(np.float32), z.astype(np.float32), random_mask.astype(bool)


def decode_action_one(s: np.ndarray, cfg: EMCartPoleConfig, rng: np.random.Generator) -> tuple[float, float, bool]:
    s = np.asarray(s, dtype=np.float32)

    if s.shape != (cfg.L,):
        raise ValueError(f"s must have shape ({cfg.L},), got {s.shape}.")

    z = float(s[int(cfg.output_cell)])

    if z < -float(cfg.threshold):
        return -float(cfg.F_max), z, False
    if z > float(cfg.threshold):
        return float(cfg.F_max), z, False

    action = float(rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32)))
    return float(cfg.F_max * action), z, True


# =============================================================================
# Rollouts and evaluation
# =============================================================================


def run_rollout_batch(
    s: np.ndarray,
    a: np.ndarray,
    cfg: EMCartPoleConfig,
    rng: np.random.Generator,
    *,
    initial_states: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Runs one CartPole episode for the full population.

    Once an agent dies, it cannot re-enter the episode.
    Invalid internal dynamics are treated as death.
    Invalid macrosteps do not increase score.
    """
    validate_config(cfg)

    s = np.asarray(s, dtype=np.float32).copy()
    a = np.asarray(a, dtype=np.float32)

    if s.ndim != 2 or s.shape[1] != cfg.L:
        raise ValueError(f"s must have shape (A, {cfg.L}), got {s.shape}.")
    if a.shape != (s.shape[0], 3):
        raise ValueError(f"a must have shape ({s.shape[0]}, 3), got {a.shape}.")
    if not np.isfinite(s).all():
        raise ValueError("s contains NaN or infinite values.")
    if not np.isfinite(a).all():
        raise ValueError("a contains NaN or infinite values.")

    A = int(s.shape[0])

    if initial_states is None:
        state = init_cartpole_states(A, cfg, rng)
    else:
        state = np.asarray(initial_states, dtype=np.float32)
        if state.shape != (A, 4):
            raise ValueError(f"initial_states must have shape ({A}, 4), got {state.shape}.")
        state = state.copy()

    alive = np.asarray(valid_internal_state(s, cfg), dtype=bool)

    returns = np.zeros(A, dtype=np.float32)
    random_action_count = np.zeros(A, dtype=np.float32)
    output_abs_sum = np.zeros(A, dtype=np.float32)
    decision_count = np.zeros(A, dtype=np.float32)
    state_norm_sum = np.zeros(A, dtype=np.float32)
    state_norm_count = np.zeros(A, dtype=np.float32)
    call_count = np.zeros(A, dtype=np.float32)
    invalid_count = np.zeros(A, dtype=np.float32)
    internal_step_sum = np.zeros(A, dtype=np.float32)

    input_cells = np.asarray(cfg.input_cells, dtype=np.int32)

    for _ in range(int(cfg.max_steps)):
        active_idx = np.flatnonzero(alive)

        if active_idx.size == 0:
            break

        prev_s = s[active_idx].copy()
        obs = normalize_obs(state[active_idx], cfg)

        s_in = prev_s.copy()
        s_in[:, input_cells] = obs

        post_valid = np.asarray(valid_internal_state(s_in, cfg), dtype=bool)

        s_after = prev_s.copy()
        valid_s = np.zeros(active_idx.size, dtype=bool)
        executed = np.zeros(active_idx.size, dtype=np.float32)

        if post_valid.any():
            local = np.flatnonzero(post_valid)
            rolled_s, rolled_valid, rolled_executed = ode_rollout_batch(
                s=s_in[local],
                a=a[active_idx[local]],
                cfg=cfg,
            )

            s_after[local] = rolled_s
            valid_s[local] = rolled_valid
            executed[local] = rolled_executed

        done = np.ones(active_idx.size, dtype=bool)
        next_state = state[active_idx].copy()

        if valid_s.any():
            valid_local = np.flatnonzero(valid_s)
            valid_global = active_idx[valid_local]

            force, z, random_action = decode_actions(s_after[valid_local], cfg, rng)

            random_action_count[valid_global] += random_action.astype(np.float32)
            output_abs_sum[valid_global] += np.abs(z).astype(np.float32)
            decision_count[valid_global] += 1.0

            state_norm_sum[valid_global] += np.linalg.norm(s_after[valid_local], axis=1).astype(np.float32)
            state_norm_count[valid_global] += 1.0

            next_state_valid, done_valid = cartpole_step(state[valid_global], force, cfg)
            next_state[valid_local] = next_state_valid
            done[valid_local] = done_valid

            # A valid CartPole step counts even if it terminates the episode.
            # Invalid internal dynamics do not count.
            returns[valid_global] += 1.0

        survived = valid_s & (~done)

        call_count[active_idx] += 1.0
        invalid_count[active_idx] += (~valid_s).astype(np.float32)
        internal_step_sum[active_idx] += executed

        s[active_idx] = s_after
        state[active_idx] = next_state
        alive[active_idx] = survived

    return {
        "returns": returns,
        "random_action_rate": random_action_count / np.maximum(decision_count, 1.0),
        "mean_abs_output": output_abs_sum / np.maximum(decision_count, 1.0),
        "mean_state_norm": state_norm_sum / np.maximum(state_norm_count, 1.0),
        "invalid_rate": invalid_count / np.maximum(call_count, 1.0),
        "mean_internal_steps": internal_step_sum / np.maximum(call_count, 1.0),
        "finite_end": np.asarray(valid_internal_state(s, cfg), dtype=bool),
        "final_internal_state": s.astype(np.float32),
    }


def aggregate_fitness(returns_all: np.ndarray, cfg: EMCartPoleConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns_all = np.asarray(returns_all, dtype=np.float32)

    if returns_all.ndim != 2:
        raise ValueError(f"returns_all must have shape (n_rollouts, A), got {returns_all.shape}.")
    if returns_all.shape[0] <= 0:
        raise ValueError("returns_all must contain at least one rollout.")

    mean_return = returns_all.mean(axis=0).astype(np.float32)
    std_return = returns_all.std(axis=0, ddof=0).astype(np.float32)

    fitness = (
        mean_return / (1.0 + float(cfg.fitness_std_penalty) * std_return)
    ).astype(np.float32)

    return fitness, mean_return, std_return


def evaluate_population(
    pop: dict[str, np.ndarray],
    cfg: EMCartPoleConfig,
    rng: np.random.Generator,
    *,
    n_rollouts: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    validate_config(cfg)
    A = validate_population(pop, cfg)

    if n_rollouts is None:
        n_rollouts = cfg.train_rollouts
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be > 0.")

    s = np.asarray(pop["s"], dtype=np.float32).copy()
    a = np.asarray(pop["a"], dtype=np.float32)

    returns_all = np.zeros((n_rollouts, A), dtype=np.float32)
    random_all = np.zeros((n_rollouts, A), dtype=np.float32)
    output_all = np.zeros((n_rollouts, A), dtype=np.float32)
    norm_all = np.zeros((n_rollouts, A), dtype=np.float32)
    invalid_all = np.zeros((n_rollouts, A), dtype=np.float32)
    finite_all = np.zeros((n_rollouts, A), dtype=bool)
    internal_all = np.zeros((n_rollouts, A), dtype=np.float32)

    for r in range(int(n_rollouts)):
        shared_state = init_cartpole_states(1, cfg, rng)
        initial_states = np.repeat(shared_state, A, axis=0)

        out = run_rollout_batch(
            s=s,
            a=a,
            cfg=cfg,
            rng=rng,
            initial_states=initial_states,
        )

        returns_all[r] = out["returns"]
        random_all[r] = out["random_action_rate"]
        output_all[r] = out["mean_abs_output"]
        norm_all[r] = out["mean_state_norm"]
        invalid_all[r] = out["invalid_rate"]
        finite_all[r] = out["finite_end"]
        internal_all[r] = out["mean_internal_steps"]

        # Lamarckian retention across rollouts.
        s = out["final_internal_state"]

    fitness, mean_return, std_return = aggregate_fitness(returns_all, cfg)
    best = int(np.argmax(fitness))

    metrics = {
        "current_fitness": fitness,
        "mean_return_by_genome": mean_return,
        "std_return_by_genome": std_return,

        "current_best_idx": best,
        "current_best_fitness": float(fitness[best]),
        "population_current_fitness": float(fitness.mean()),
        "population_current_fitness_std": float(fitness.std(ddof=0)),

        "current_best_mean_return": float(mean_return[best]),
        "current_best_std_return": float(std_return[best]),
        "population_mean_return": float(mean_return.mean()),
        "population_std_return": float(std_return.mean()),

        "fitness_std_penalty": float(cfg.fitness_std_penalty),

        "mean_random_action_rate": float(random_all.mean()),
        "mean_abs_output": float(output_all.mean()),
        "mean_state_norm": float(norm_all.mean()),
        "mean_invalid_rate": float(invalid_all.mean()),
        "finite_rate": float(finite_all.mean()),
        "mean_internal_steps": float(internal_all.mean()),

        "random_action_rate_by_genome": random_all.mean(axis=0).astype(np.float32),
        "mean_abs_output_by_genome": output_all.mean(axis=0).astype(np.float32),
        "mean_state_norm_by_genome": norm_all.mean(axis=0).astype(np.float32),
        "invalid_rate_by_genome": invalid_all.mean(axis=0).astype(np.float32),
        "finite_rate_by_genome": finite_all.mean(axis=0).astype(np.float32),
        "mean_internal_steps_by_genome": internal_all.mean(axis=0).astype(np.float32),
    }

    return fitness, metrics, s.astype(np.float32)


def evaluate_single_genome(
    *,
    s: np.ndarray,
    a: np.ndarray,
    cfg: EMCartPoleConfig,
    rng: np.random.Generator,
    n_rollouts: int | None = None,
) -> dict[str, Any]:
    if n_rollouts is None:
        n_rollouts = cfg.eval_rollouts
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be > 0.")

    s_cur = np.asarray(s, dtype=np.float32).reshape(1, cfg.L).copy()
    a_batch = np.asarray(a, dtype=np.float32).reshape(1, 3)

    returns = np.zeros(n_rollouts, dtype=np.float32)
    random_rate = np.zeros(n_rollouts, dtype=np.float32)
    abs_output = np.zeros(n_rollouts, dtype=np.float32)
    state_norm = np.zeros(n_rollouts, dtype=np.float32)
    invalid_rate = np.zeros(n_rollouts, dtype=np.float32)
    finite_end = np.zeros(n_rollouts, dtype=bool)
    internal_steps = np.zeros(n_rollouts, dtype=np.float32)

    for r in range(int(n_rollouts)):
        out = run_rollout_batch(s_cur, a_batch, cfg, rng)

        returns[r] = float(out["returns"][0])
        random_rate[r] = float(out["random_action_rate"][0])
        abs_output[r] = float(out["mean_abs_output"][0])
        state_norm[r] = float(out["mean_state_norm"][0])
        invalid_rate[r] = float(out["invalid_rate"][0])
        finite_end[r] = bool(out["finite_end"][0])
        internal_steps[r] = float(out["mean_internal_steps"][0])

        s_cur = out["final_internal_state"]

    fitness, mean_return, std_return = aggregate_fitness(returns[:, None], cfg)

    return {
        "n_rollouts": int(n_rollouts),
        "fitness": float(fitness[0]),
        "mean_return": float(mean_return[0]),
        "std_return": float(std_return[0]),
        "best_return": float(returns.max()),
        "min_return": float(returns.min()),
        "mean_random_action_rate": float(random_rate.mean()),
        "mean_abs_output": float(abs_output.mean()),
        "mean_state_norm": float(state_norm.mean()),
        "mean_invalid_rate": float(invalid_rate.mean()),
        "finite_rate": float(finite_end.mean()),
        "mean_internal_steps": float(internal_steps.mean()),
        "final_internal_state": s_cur[0].astype(np.float32),
    }


def trace_single_episode(
    *,
    s: np.ndarray,
    a: np.ndarray,
    cfg: EMCartPoleConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    s = np.asarray(s, dtype=np.float32).reshape(cfg.L).copy()
    a = np.asarray(a, dtype=np.float32).reshape(3)

    if not np.isfinite(s).all():
        raise ValueError("s contains NaN or infinite values.")
    if not np.isfinite(a).all():
        raise ValueError("a contains NaN or infinite values.")

    state = init_cartpole_states(1, cfg, rng)[0]

    cart_states: list[np.ndarray] = []
    forces: list[float] = []
    outputs: list[float] = []
    random_actions: list[bool] = []
    macro_boundaries: list[int] = []
    automaton_rows: list[np.ndarray] = []

    input_cells = np.asarray(cfg.input_cells, dtype=np.int32)
    invalid = False

    for _ in range(int(cfg.max_steps)):
        macro_boundaries.append(len(automaton_rows))

        prev_s = s.copy()
        obs = normalize_obs(state, cfg)

        s_work = prev_s.copy()
        s_work[input_cells] = obs
        automaton_rows.append(s_work.copy())

        if not bool(valid_internal_state(s_work, cfg)):
            invalid = True
            s = prev_s
            break

        for _micro in range(int(cfg.T)):
            candidate = ode_step_one(s_work, a, cfg)

            if not bool(valid_internal_state(candidate, cfg)):
                invalid = True
                break

            s_work = candidate
            automaton_rows.append(s_work.copy())

        if invalid:
            s = s_work
            break

        force, z, random_action = decode_action_one(s_work, cfg, rng)
        next_state, done = cartpole_step_one(state, force, cfg)

        cart_states.append(state.copy())
        forces.append(float(force))
        outputs.append(float(z))
        random_actions.append(bool(random_action))

        s = s_work
        state = next_state

        if done:
            break

    return {
        "return": int(len(cart_states)),
        "invalid": bool(invalid),
        "cart_state": np.asarray(cart_states, dtype=np.float32),
        "force": np.asarray(forces, dtype=np.float32),
        "output": np.asarray(outputs, dtype=np.float32),
        "random_action": np.asarray(random_actions, dtype=bool),
        "automaton": np.asarray(automaton_rows, dtype=np.float32),
        "macro_boundaries": np.asarray(macro_boundaries, dtype=np.int32),
        "final_internal_state": s.astype(np.float32),
    }


# =============================================================================
# Plotting / reporting
# =============================================================================


def plot_training(history: list[dict[str, Any]]):
    if not history:
        return None, None

    gens = [h["gen"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(gens, _series(history, "selected_fitness"), label="selected")
    axes[0].plot(gens, _series(history, "top5_fitness"), label="top 5%")
    axes[0].plot(gens, _series(history, "population_fitness"), label="population")
    axes[0].set_title("fitness")
    axes[0].set_xlabel("generation")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(gens, _series(history, "selected_mean_return"), label="selected mean")
    axes[1].plot(gens, _series(history, "selected_std_return"), label="selected std")
    axes[1].plot(gens, _series(history, "population_mean_return"), label="population mean")
    axes[1].set_title("return")
    axes[1].set_xlabel("generation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(gens, _series(history, "selected_random_action_rate"), label="selected random")
    axes[2].plot(gens, _series(history, "selected_invalid_rate"), label="selected invalid")
    axes[2].plot(gens, _series(history, "finite_rate"), label="finite rate")
    axes[2].set_title("diagnostics")
    axes[2].set_xlabel("generation")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    return fig, axes


def plot_automaton_trace(trace: dict[str, Any], cfg: EMCartPoleConfig):
    arr = np.asarray(trace["automaton"], dtype=np.float32)

    if arr.size == 0:
        raise ValueError("trace contains no automaton rows.")

    n_rows = arr.shape[0]
    height = max(8, min(80, n_rows / 12))

    fig, ax = plt.subplots(figsize=(8, height))
    im = ax.imshow(arr, aspect="equal", interpolation="nearest", origin="upper", cmap="coolwarm")

    for y in np.asarray(trace["macro_boundaries"], dtype=np.int32):
        if 0 <= int(y) < n_rows:
            ax.axhline(int(y) - 0.5, color="black", linewidth=0.25, alpha=0.45)

    ax.set_xlabel("cell")
    ax.set_ylabel("internal microstep")
    ax.set_title("Continuous 1D EM automaton trace")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig, ax


def print_eval_report(title: str, report: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")

    keys = [
        "n_rollouts",
        "fitness",
        "mean_return",
        "std_return",
        "best_return",
        "min_return",
        "mean_random_action_rate",
        "mean_abs_output",
        "mean_state_norm",
        "mean_invalid_rate",
        "finite_rate",
        "mean_internal_steps",
    ]

    for key in keys:
        if key not in report:
            continue
        value = report[key]
        if isinstance(value, float):
            print(f"{key:25s}: {value:.6f}")
        else:
            print(f"{key:25s}: {value}")


def make_cartpole_mp4(
    trace: dict[str, Any],
    cfg: EMCartPoleConfig,
    path: str | Path,
    *,
    fps: int = 30,
    width: int = 480,
    height: int = 300,
) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    states = np.asarray(trace["cart_state"], dtype=np.float32)
    forces = np.asarray(trace["force"], dtype=np.float32)

    if states.shape[0] == 0:
        raise ValueError("trace contains no CartPole states.")
    if forces.shape != (states.shape[0],):
        raise ValueError(f"trace['force'] must have shape ({states.shape[0]},), got {forces.shape}.")

    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for i in range(states.shape[0]):
            img = _render_cartpole_frame(states[i], float(forces[i]), cfg, width, height)
            writer.append_data(_pad_rgb(img))

    return path


# =============================================================================
# Private helpers
# =============================================================================


def _series(history: list[dict[str, Any]], key: str, default: float = float("nan")) -> list[float]:
    return [float(row.get(key, default)) for row in history]


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default)


def _json_default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def _json_load_0d(arr: np.ndarray) -> Any:
    value = arr.item() if hasattr(arr, "item") else str(arr)
    return json.loads(value)


def _load_pop(z: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    pop: dict[str, np.ndarray] = {}

    for key in z.files:
        if key.startswith("pop__"):
            pop[key[len("pop__"):]] = z[key].copy()

    if not pop:
        raise ValueError("checkpoint contains no population arrays.")

    return pop


def _render_cartpole_frame(
    state: np.ndarray,
    force: float,
    cfg: EMCartPoleConfig,
    width: int,
    height: int,
) -> np.ndarray:
    x = float(state[0])
    theta = float(state[2])

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_xlim(-cfg.x_threshold * 1.15, cfg.x_threshold * 1.15)
    ax.set_ylim(-0.35, 1.2)
    ax.axis("off")

    ax.plot([-cfg.x_threshold, cfg.x_threshold], [0, 0], linewidth=2)

    cart_y = 0.1
    cart_w = 0.35
    cart_h = 0.18

    ax.add_patch(plt.Rectangle((x - cart_w / 2.0, cart_y - cart_h / 2.0), cart_w, cart_h))

    pole_len = cfg.length * 2.0
    x0 = x
    y0 = cart_y + cart_h / 2.0
    x1 = x0 + pole_len * np.sin(theta)
    y1 = y0 + pole_len * np.cos(theta)

    ax.plot([x0, x1], [y0, y1], linewidth=4)

    ax.text(
        0.02,
        0.95,
        f"x={x:.2f}  theta={theta:.3f}  F={force:.1f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    img = (
        np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        .reshape(h, w, 4)[..., :3]
        .copy()
    )

    plt.close(fig)
    return img


def _pad_rgb(img: np.ndarray, m: int = 16) -> np.ndarray:
    h, w, c = img.shape
    H = ((h + m - 1) // m) * m
    W = ((w + m - 1) // m) * m

    if H == h and W == w:
        return img

    out = np.ones((H, W, c), dtype=np.uint8) * 255
    out[:h, :w] = img
    return out