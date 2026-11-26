"""
general_ga.py

Generic, simple genetic algorithm engine for fixed-shape NumPy genomes.

Core conventions
----------------
- First axis of every genome variable is the population axis: shape = (A, ...).
- Parent A is the base parent.
- Parent B is the donor parent.
- Trainable child variables start from parent A, receive sparse donor entries
  from parent B, then mutate.
- Non-trainable variables are copied from parent A unchanged.
- This keeps inherited/Lamarckian state coupled to the base parent.

Reproduction order
------------------
Next population is filled in this order:

1. elites
2. bank immigrants
3. random immigrants
4. children until full population

Mutation types
--------------
- "additive":
    float32 only. Add noise, then clamp to [min_val, max_val].

- "additive_wrap":
    float32 only. Add noise, then wrap into [min_val, max_val).

- "resample":
    float32/int32/int8. Resample mutated entries.

- "none":
    no mutation.

Adaptive mutation scale
-----------------------
For additive/additive_wrap float mutation:

- adaptive_mutation_scale=True:
    scale = sigma_base * (abs(x) + eps_adapt)

- adaptive_mutation_scale=False:
    scale = sigma_base

Bounds
------
- int32/int8 ranges use [min_val, max_val), max_val exclusive.
- float32 uniform init uses [min_val, max_val).
- float32 additive uses clipping to [min_val, max_val].
- float32 additive_wrap uses wrapping into [min_val, max_val).

No silent fallbacks:
- Invalid configurations raise ValueError/TypeError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Config dataclasses
# ----------------------------------------------------------------------


@dataclass
class VarSpec:
    name: str
    dtype: str
    shape: Tuple[int, ...]

    trainable: bool = True

    init_mode: str = "gaussian"
    min_val: float = -1.0
    max_val: float = 1.0
    const_val: float = 0.0

    resample_min: Optional[float] = None
    resample_max: Optional[float] = None

    mut_type: str = "additive"  # "additive" | "additive_wrap" | "resample" | "none"
    adaptive_mutation_scale: bool = True

    mut_prob: float = 0.0
    sigma_base: float = 0.0
    crossover_prob: float = 0.0


@dataclass
class GAConfig:
    population_size: int
    elitism: int

    # Bank immigrants are sampled from the stored bank of previous bests.
    bank_max: int = 0
    bank_add_every: int = 1
    immigrants_per_gen: int = 0

    # Random immigrants are freshly initialized individuals.
    random_immigrants_per_gen: int = 0

    tournament_k: int = 2

    eps_adapt: float = 1e-3
    t_df: float = 2.0
    noise_dist: str = "student_t"  # "student_t" | "gaussian" | "uniform"


# ----------------------------------------------------------------------
# GA engine
# ----------------------------------------------------------------------


class GA:
    def __init__(self, config: GAConfig, var_specs: List[VarSpec]):
        self.config = config
        self.var_specs = [self._normalize_spec(v) for v in var_specs]
        self._validate()

        self.pop_cur: Dict[str, np.ndarray] | None = None
        self.pop_next: Dict[str, np.ndarray] | None = None

        # Each bank entry is a full individual:
        # dict[var_name] -> one individual's array for that variable.
        self.bank: List[Dict[str, np.ndarray]] = []

        self.generation = 0
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, init: bool = False) -> Dict[str, np.ndarray]:
        if self.pop_cur is None or init:
            self._init_population()
            self.generation = 0
            self.bank.clear()

        if self.pop_cur is None:
            raise RuntimeError("Population initialization failed.")

        return self.pop_cur

    def tell(self, fitness: np.ndarray) -> None:
        if self.pop_cur is None or self.pop_next is None:
            raise RuntimeError("tell() called before ask()/initialization.")

        fitness = np.asarray(fitness, dtype=np.float32)
        A = self.config.population_size

        if fitness.shape != (A,):
            raise ValueError(f"fitness must have shape ({A},), got {fitness.shape}.")

        if not np.isfinite(fitness).all():
            raise ValueError("fitness contains NaN or infinite values.")

        order = np.argsort(-fitness)

        self._update_bank(fitness, order)
        self._build_next_population(fitness, order)

        self.pop_cur, self.pop_next = self.pop_next, self.pop_cur
        self.generation += 1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_spec(spec: VarSpec) -> VarSpec:
        dtype = str(spec.dtype).lower().strip()

        if dtype in ("float32", "float"):
            dtype = "float32"
        elif dtype in ("int32", "int"):
            dtype = "int32"
        elif dtype in ("int8", "i8"):
            dtype = "int8"
        else:
            raise ValueError(f"Unsupported dtype in VarSpec {spec.name!r}: {spec.dtype!r}.")

        return VarSpec(
            name=spec.name,
            dtype=dtype,
            shape=tuple(spec.shape),
            trainable=bool(spec.trainable),
            init_mode=str(spec.init_mode),
            min_val=float(spec.min_val),
            max_val=float(spec.max_val),
            const_val=float(spec.const_val),
            resample_min=None if spec.resample_min is None else float(spec.resample_min),
            resample_max=None if spec.resample_max is None else float(spec.resample_max),
            mut_type=str(spec.mut_type),
            adaptive_mutation_scale=bool(spec.adaptive_mutation_scale),
            mut_prob=float(spec.mut_prob),
            sigma_base=float(spec.sigma_base),
            crossover_prob=float(spec.crossover_prob),
        )

    @staticmethod
    def _is_int_valued(x: float) -> bool:
        return float(x).is_integer()

    def _validate(self) -> None:
        cfg = self.config
        A = cfg.population_size

        if A <= 0:
            raise ValueError("population_size must be > 0.")
        if cfg.elitism < 0 or cfg.elitism >= A:
            raise ValueError("elitism must satisfy 0 <= elitism < population_size.")
        if cfg.bank_max < 0:
            raise ValueError("bank_max must be >= 0.")
        if cfg.bank_add_every < 0:
            raise ValueError("bank_add_every must be >= 0.")
        if cfg.immigrants_per_gen < 0:
            raise ValueError("immigrants_per_gen must be >= 0.")
        if cfg.random_immigrants_per_gen < 0:
            raise ValueError("random_immigrants_per_gen must be >= 0.")
        if cfg.tournament_k < 1:
            raise ValueError("tournament_k must be >= 1.")
        if cfg.eps_adapt < 0:
            raise ValueError("eps_adapt must be >= 0.")
        if cfg.t_df <= 0:
            raise ValueError("t_df must be > 0.")
        if cfg.noise_dist not in ("student_t", "gaussian", "uniform"):
            raise ValueError("noise_dist must be 'student_t', 'gaussian', or 'uniform'.")

        names = [v.name for v in self.var_specs]
        if len(set(names)) != len(names):
            raise ValueError("VarSpec names must be unique.")

        for spec in self.var_specs:
            self._validate_spec(spec, A)

    def _validate_spec(self, spec: VarSpec, A: int) -> None:
        if not spec.name:
            raise ValueError("VarSpec name must be non-empty.")

        if len(spec.shape) == 0:
            raise ValueError(f"VarSpec {spec.name}: shape must include population axis.")

        if spec.shape[0] != A:
            raise ValueError(
                f"VarSpec {spec.name}: first dimension must be population size {A}, "
                f"got {spec.shape[0]}."
            )

        if spec.min_val >= spec.max_val:
            raise ValueError(
                f"VarSpec {spec.name}: min_val must be < max_val, "
                f"got {spec.min_val} >= {spec.max_val}."
            )

        if not (0.0 <= spec.mut_prob <= 1.0):
            raise ValueError(f"VarSpec {spec.name}: mut_prob must be in [0, 1].")

        if not (0.0 <= spec.crossover_prob <= 1.0):
            raise ValueError(f"VarSpec {spec.name}: crossover_prob must be in [0, 1].")

        if spec.sigma_base < 0.0:
            raise ValueError(f"VarSpec {spec.name}: sigma_base must be >= 0.")

        if spec.mut_type not in ("additive", "additive_wrap", "resample", "none"):
            raise ValueError(
                f"VarSpec {spec.name}: mut_type must be 'additive', "
                "'additive_wrap', 'resample', or 'none'."
            )

        if spec.dtype == "float32":
            if spec.init_mode not in ("gaussian", "uniform", "constant"):
                raise ValueError(
                    f"VarSpec {spec.name}: float32 requires init_mode "
                    "'gaussian', 'uniform', or 'constant'."
                )

        elif spec.dtype in ("int32", "int8"):
            if spec.init_mode not in ("int_uniform", "constant"):
                raise ValueError(
                    f"VarSpec {spec.name}: {spec.dtype} requires init_mode "
                    "'int_uniform' or 'constant'."
                )

            if spec.mut_type in ("additive", "additive_wrap"):
                raise ValueError(
                    f"VarSpec {spec.name}: {spec.dtype} cannot use mut_type "
                    f"{spec.mut_type!r}; use 'resample' or 'none'."
                )

            if not (self._is_int_valued(spec.min_val) and self._is_int_valued(spec.max_val)):
                raise ValueError(
                    f"VarSpec {spec.name}: integer dtype requires integer-valued "
                    "min_val and max_val."
                )

            if spec.init_mode == "constant" and not self._is_int_valued(spec.const_val):
                raise ValueError(
                    f"VarSpec {spec.name}: integer dtype requires integer-valued const_val."
                )

        else:
            raise ValueError(f"VarSpec {spec.name}: unsupported dtype {spec.dtype!r}.")

        if spec.dtype == "int8":
            if spec.min_val < -128 or spec.min_val > 127:
                raise ValueError(f"VarSpec {spec.name}: int8 min_val must be in [-128, 127].")
            if spec.max_val < -127 or spec.max_val > 128:
                raise ValueError(
                    f"VarSpec {spec.name}: int8 max_val must be in [-127, 128] "
                    "as exclusive high."
                )

        if (spec.resample_min is None) ^ (spec.resample_max is None):
            raise ValueError(
                f"VarSpec {spec.name}: resample_min and resample_max must be both set or both None."
            )

        if spec.resample_min is not None and spec.resample_max is not None:
            if not (spec.min_val <= spec.resample_min < spec.resample_max <= spec.max_val):
                raise ValueError(
                    f"VarSpec {spec.name}: resample bounds must satisfy "
                    "min_val <= resample_min < resample_max <= max_val."
                )

            if spec.dtype in ("int32", "int8"):
                if not (
                    self._is_int_valued(spec.resample_min)
                    and self._is_int_valued(spec.resample_max)
                ):
                    raise ValueError(
                        f"VarSpec {spec.name}: integer dtype requires integer-valued "
                        "resample_min and resample_max."
                    )

        if not spec.trainable:
            if spec.mut_type != "none":
                raise ValueError(
                    f"VarSpec {spec.name}: non-trainable variables must use mut_type='none'."
                )
            if spec.mut_prob != 0.0:
                raise ValueError(
                    f"VarSpec {spec.name}: non-trainable variables must have mut_prob=0."
                )
            if spec.crossover_prob != 0.0:
                raise ValueError(
                    f"VarSpec {spec.name}: non-trainable variables must have crossover_prob=0."
                )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_population(self) -> None:
        self.pop_cur = {}

        for spec in self.var_specs:
            self.pop_cur[spec.name] = self._init_var(spec, spec.shape)

        self.pop_next = {
            name: np.empty_like(arr)
            for name, arr in self.pop_cur.items()
        }

    def _init_random_individual(self, spec: VarSpec) -> np.ndarray:
        return self._init_var(spec, spec.shape[1:])

    def _init_var(self, spec: VarSpec, shape: Tuple[int, ...]) -> np.ndarray:
        if spec.dtype == "float32":
            return self._init_float(spec, shape)
        if spec.dtype in ("int32", "int8"):
            return self._init_int(spec, shape)
        raise TypeError(f"Unsupported dtype {spec.dtype!r} for VarSpec {spec.name}.")

    def _init_float(self, spec: VarSpec, shape: Tuple[int, ...]) -> np.ndarray:
        if spec.init_mode == "gaussian":
            std = (spec.max_val - spec.min_val) / 4.0
            if std <= 0.0:
                raise ValueError(f"VarSpec {spec.name}: invalid Gaussian init std.")

            arr = self.rng.normal(0.0, std, size=shape).astype(np.float32)
            return np.clip(arr, spec.min_val, spec.max_val).astype(np.float32, copy=False)

        if spec.init_mode == "uniform":
            return self.rng.uniform(
                spec.min_val,
                spec.max_val,
                size=shape,
            ).astype(np.float32)

        if spec.init_mode == "constant":
            arr = np.full(shape, spec.const_val, dtype=np.float32)
            return np.clip(arr, spec.min_val, spec.max_val).astype(np.float32, copy=False)

        raise ValueError(f"VarSpec {spec.name}: unsupported float init_mode {spec.init_mode!r}.")

    def _init_int(self, spec: VarSpec, shape: Tuple[int, ...]) -> np.ndarray:
        low = int(spec.min_val)
        high = int(spec.max_val)

        if high <= low:
            raise ValueError(f"VarSpec {spec.name}: integer init requires max_val > min_val.")

        if spec.init_mode == "int_uniform":
            if spec.dtype == "int8":
                return self.rng.integers(low, high, size=shape, dtype=np.int16).astype(np.int8)
            return self.rng.integers(low, high, size=shape, dtype=np.int32)

        if spec.init_mode == "constant":
            val = int(spec.const_val)

            if spec.dtype == "int8":
                arr = np.full(shape, val, dtype=np.int16)
                arr = np.maximum(arr, low)
                arr = np.minimum(arr, high - 1)
                return arr.astype(np.int8)

            arr = np.full(shape, val, dtype=np.int32)
            arr = np.maximum(arr, low)
            arr = np.minimum(arr, high - 1)
            return arr.astype(np.int32, copy=False)

        raise ValueError(f"VarSpec {spec.name}: unsupported int init_mode {spec.init_mode!r}.")

    # ------------------------------------------------------------------
    # Bank
    # ------------------------------------------------------------------

    def _update_bank(self, fitness: np.ndarray, order: np.ndarray) -> None:
        cfg = self.config

        if cfg.bank_max == 0:
            return
        if cfg.bank_add_every == 0:
            return
        if self.generation % cfg.bank_add_every != 0:
            return
        if self.pop_cur is None:
            raise RuntimeError("Population is not initialized.")

        best_idx = int(order[0])
        individual = {
            spec.name: self.pop_cur[spec.name][best_idx].copy()
            for spec in self.var_specs
        }
        individual["fitness"] = np.asarray(float(fitness[best_idx]), dtype=np.float32)

        self.bank.append(individual)

        if len(self.bank) > cfg.bank_max:
            self.bank = self.bank[-cfg.bank_max:]

    def _sample_bank_immigrants(self, n: int) -> List[Dict[str, np.ndarray]]:
        if n < 0:
            raise ValueError("Number of bank immigrants must be >= 0.")
        if n == 0:
            return []
        if n > len(self.bank):
            raise ValueError("Requested more bank immigrants than available bank entries.")

        idx = self.rng.choice(len(self.bank), size=n, replace=False)
        return [self.bank[int(i)] for i in idx]

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def _build_next_population(self, fitness: np.ndarray, order: np.ndarray) -> None:
        if self.pop_cur is None or self.pop_next is None:
            raise RuntimeError("Population buffers are not initialized.")

        A = self.config.population_size
        cursor = 0

        # 1. Elites.
        elite_n = min(self.config.elitism, A)
        if elite_n > 0:
            elite_idx = order[:elite_n]
            for spec in self.var_specs:
                self.pop_next[spec.name][cursor:cursor + elite_n] = self.pop_cur[spec.name][elite_idx]
            cursor += elite_n

        # 2. Bank immigrants.
        remaining = A - cursor
        bank_imm_n = min(self.config.immigrants_per_gen, len(self.bank), remaining)
        if bank_imm_n > 0:
            immigrants = self._sample_bank_immigrants(bank_imm_n)
            for spec in self.var_specs:
                self.pop_next[spec.name][cursor:cursor + bank_imm_n] = np.stack(
                    [ind[spec.name] for ind in immigrants],
                    axis=0,
                )
            cursor += bank_imm_n

        # 3. Random immigrants.
        remaining = A - cursor
        rand_imm_n = min(self.config.random_immigrants_per_gen, remaining)
        if rand_imm_n > 0:
            for i in range(rand_imm_n):
                dest = cursor + i
                for spec in self.var_specs:
                    self.pop_next[spec.name][dest] = self._init_random_individual(spec)
            cursor += rand_imm_n

        # 4. Children until full.
        while cursor < A:
            self._write_child(dest=cursor, fitness=fitness)
            cursor += 1

        if cursor != A:
            raise RuntimeError(f"Internal reproduction size error: cursor={cursor}, A={A}.")

        for spec in self.var_specs:
            if self.pop_next[spec.name].shape != spec.shape:
                raise RuntimeError(
                    f"Internal shape error for {spec.name}: "
                    f"got {self.pop_next[spec.name].shape}, expected {spec.shape}."
                )

    def _write_child(self, *, dest: int, fitness: np.ndarray) -> None:
        if self.pop_cur is None or self.pop_next is None:
            raise RuntimeError("Population buffers are not initialized.")

        pa = self._tournament_select(fitness)  # base parent
        pb = self._tournament_select(fitness)  # donor parent; may equal pa

        for spec in self.var_specs:
            name = spec.name
            parent_a = self.pop_cur[name][pa]
            parent_b = self.pop_cur[name][pb]

            if not spec.trainable:
                self.pop_next[name][dest] = parent_a
                continue

            child = self._crossover(parent_a, parent_b, spec)
            child = self._mutate(child, spec)
            child = self._enforce_bounds(child, spec)

            self.pop_next[name][dest] = child

    def _tournament_select(self, fitness: np.ndarray) -> int:
        A = fitness.shape[0]
        idx = self.rng.integers(0, A, size=self.config.tournament_k)
        return int(idx[np.argmax(fitness[idx])])

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray, spec: VarSpec) -> np.ndarray:
        if spec.crossover_prob == 0.0:
            return parent_a.copy()

        donor_mask = self.rng.random(parent_a.shape) < spec.crossover_prob

        child = parent_a.copy()
        child[donor_mask] = parent_b[donor_mask]
        return child.astype(parent_a.dtype, copy=False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, child: np.ndarray, spec: VarSpec) -> np.ndarray:
        if spec.mut_type == "none" or spec.mut_prob == 0.0:
            return child

        mut_mask = self.rng.random(child.shape) < spec.mut_prob
        if not mut_mask.any():
            return child

        if spec.mut_type in ("additive", "additive_wrap"):
            return self._mutate_additive(child, mut_mask, spec)

        if spec.mut_type == "resample":
            return self._mutate_resample(child, mut_mask, spec)

        raise ValueError(f"VarSpec {spec.name}: unsupported mut_type {spec.mut_type!r}.")

    def _mutate_additive(self, child: np.ndarray, mut_mask: np.ndarray, spec: VarSpec) -> np.ndarray:
        if spec.dtype != "float32":
            raise TypeError(f"VarSpec {spec.name}: additive mutation requires float32.")

        if spec.sigma_base == 0.0:
            return child

        noise = self._draw_noise(int(mut_mask.sum()))

        if spec.adaptive_mutation_scale:
            scale = spec.sigma_base * (
                np.abs(child[mut_mask]).astype(np.float32) + np.float32(self.config.eps_adapt)
            )
        else:
            scale = np.float32(spec.sigma_base)

        child[mut_mask] = child[mut_mask] + scale * noise
        return child

    def _mutate_resample(self, child: np.ndarray, mut_mask: np.ndarray, spec: VarSpec) -> np.ndarray:
        n = int(mut_mask.sum())
        lo, hi = self._resample_bounds(spec)

        if spec.dtype == "float32":
            values = self._sample_float_values(spec, n, lo, hi)
        elif spec.dtype in ("int32", "int8"):
            values = self._sample_int_values(spec, n, lo, hi)
        else:
            raise TypeError(f"VarSpec {spec.name}: unsupported dtype {spec.dtype!r} for resample.")

        child[mut_mask] = values
        return child

    def _draw_noise(self, size: int) -> np.ndarray:
        if size < 0:
            raise ValueError("Noise size must be >= 0.")

        if self.config.noise_dist == "student_t":
            x = self.rng.standard_t(df=self.config.t_df, size=size)
        elif self.config.noise_dist == "gaussian":
            x = self.rng.normal(0.0, 1.0, size=size)
        elif self.config.noise_dist == "uniform":
            x = self.rng.uniform(-1.0, 1.0, size=size)
        else:
            raise ValueError(f"Unsupported noise_dist {self.config.noise_dist!r}.")

        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _resample_bounds(self, spec: VarSpec) -> Tuple[float, float]:
        if spec.resample_min is None or spec.resample_max is None:
            return spec.min_val, spec.max_val
        return spec.resample_min, spec.resample_max

    def _sample_float_values(self, spec: VarSpec, size: int, lo: float, hi: float) -> np.ndarray:
        if spec.init_mode == "gaussian":
            std = (hi - lo) / 4.0
            if std <= 0.0:
                raise ValueError(f"VarSpec {spec.name}: invalid Gaussian resample std.")

            values = self.rng.normal(0.0, std, size=size).astype(np.float32)
            return np.clip(values, lo, hi).astype(np.float32, copy=False)

        if spec.init_mode == "uniform":
            return self.rng.uniform(lo, hi, size=size).astype(np.float32)

        if spec.init_mode == "constant":
            values = np.full(size, spec.const_val, dtype=np.float32)
            return np.clip(values, lo, hi).astype(np.float32, copy=False)

        raise ValueError(f"VarSpec {spec.name}: unsupported float init_mode {spec.init_mode!r}.")

    def _sample_int_values(self, spec: VarSpec, size: int, lo: float, hi: float) -> np.ndarray:
        low = int(lo)
        high = int(hi)

        if high <= low:
            raise ValueError(f"VarSpec {spec.name}: integer resample requires high > low.")

        if spec.init_mode == "int_uniform":
            if spec.dtype == "int8":
                return self.rng.integers(low, high, size=size, dtype=np.int16).astype(np.int8)
            return self.rng.integers(low, high, size=size, dtype=np.int32)

        if spec.init_mode == "constant":
            val = int(spec.const_val)

            if spec.dtype == "int8":
                values = np.full(size, val, dtype=np.int16)
                values = np.maximum(values, low)
                values = np.minimum(values, high - 1)
                return values.astype(np.int8)

            values = np.full(size, val, dtype=np.int32)
            values = np.maximum(values, low)
            values = np.minimum(values, high - 1)
            return values.astype(np.int32, copy=False)

        raise ValueError(f"VarSpec {spec.name}: unsupported int init_mode {spec.init_mode!r}.")

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def _enforce_bounds(self, child: np.ndarray, spec: VarSpec) -> np.ndarray:
        if spec.dtype == "float32":
            if spec.mut_type == "additive_wrap":
                return self._wrap_float(child, spec)
            return self._clip_float(child, spec)

        if spec.dtype in ("int32", "int8"):
            return self._clip_int(child, spec)

        raise TypeError(f"VarSpec {spec.name}: unsupported dtype {spec.dtype!r}.")

    @staticmethod
    def _clip_float(child: np.ndarray, spec: VarSpec) -> np.ndarray:
        return np.clip(child, spec.min_val, spec.max_val).astype(np.float32, copy=False)

    @staticmethod
    def _wrap_float(child: np.ndarray, spec: VarSpec) -> np.ndarray:
        lo = np.float32(spec.min_val)
        hi = np.float32(spec.max_val)
        width = np.float32(hi - lo)

        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"VarSpec {spec.name}: additive_wrap requires finite max_val > min_val.")

        return (lo + np.mod(child.astype(np.float32, copy=False) - lo, width)).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _clip_int(child: np.ndarray, spec: VarSpec) -> np.ndarray:
        low = int(spec.min_val)
        high = int(spec.max_val)

        if spec.dtype == "int8":
            arr = child.astype(np.int16, copy=False)
            arr = np.maximum(arr, low)
            arr = np.minimum(arr, high - 1)
            return arr.astype(np.int8, copy=False)

        if spec.dtype == "int32":
            arr = child.astype(np.int32, copy=False)
            arr = np.maximum(arr, low)
            arr = np.minimum(arr, high - 1)
            return arr.astype(np.int32, copy=False)

        raise TypeError(f"VarSpec {spec.name}: unsupported integer dtype {spec.dtype!r}.")


ga = GA