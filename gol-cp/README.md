# GoL-EM — CartPole Control

An Emergent Model built on **Conway's Game of Life** used as a **reactive controller** for
CartPole. The GoL rule is fixed; only the initial board (the program) is trained. At every
control step the board is reset to the program, the observation is encoded, the GoL dynamics are
run to a fixed point, and the resulting live-cell pattern is decoded into a force.

## Model

```
F_i = D( f_GoL^T ( p (+) E(x_i) ) )
```

- **Substrate:** a finite, **bounded** GoL board (no wraparound). Values `W = {0,1}`.
- **Program `p` (trainable):** placed in the upper-left `Lp×Lp` region, built from
  `n_gliders` gliders (each a `(row, col)` anchor + on/off bit) **XOR** a sparse `Lp×Lp` binary
  `singletons` mask. So the program is "a few gliders plus sparse binary perturbations."
- **Interface (fixed):** the board carries a `V × B` grid of interface tiles built by
  `build_layout`:
  - `V = 4` horizontal **lanes**, one per CartPole variable `(x, ẋ, θ, θ̇)`;
  - `B = 6` **bins** per lane. Each observation variable is clamped to `±limit` and discretized
    into one of the 6 bins; a `2×2` live block is written at the selected bin's **input** cell.
  - Each bin also owns a left **L-collector** and a right **R-collector** `2×2` region.
- **Update `f`:** the standard Conway rule.
- **Halting `H`:** fixed point (`f(s) = s`), checked every `halt_check_every` steps up to
  `T_max`. If it does not settle, the terminal board is still decoded but a small fitness penalty
  is applied.
- **Decoding `D` (continuous force):** count live cells in the L and R collector masks and take
  their imbalance:

  ```
  F = clip( gain * (R - L) / (R + L + b),  -F_clip, +F_clip )
  ```

- **No state retention:** the board is reset to `p` at every CartPole macrostep (purely
  reactive). Residual state is not carried between steps.

## Training — mutation-only GA

A simple genetic algorithm over the program genome:

- **Selection:** elitism (`elitism_E`) + tournament selection (`tournament_k`).
- **Mutation:** glider positions (Student-t jitter, `p_pos_mut`), active-bit flips
  (`p_active_mut`), singleton-bit flips (`p_singleton_mut`). **No crossover.**
- **Fitness:** per surviving CartPole step, add `1 - nonhalt_penalty * not_halted`, so balancing
  is rewarded but steps where GoL did not halt in budget are penalized. Averaged over
  `n_rollouts_train` rollouts.

Training runs on the GPU-batched simulator; evaluation uses the NumPy path.

## Files

```
utils.py     # Cfg, board layout, program rendering, encode/decode,
             # CPU + Taichi GoL simulators (with equivalence self-tests),
             # CartPole physics, GA (mutate/select/next_generation),
             # training loop, evaluation, GoL + CartPole video rendering
test.ipynb   # driver: build layout, self-tests, train, evaluate, (optional) videos
charts.ipynb # figure / analysis helpers
```

### Key pieces in `utils.py`

- `Cfg` — all hyperparameters (GoL, interface, GA, CartPole physics).
- `build_layout` / `validate_layout` / `plot_layout` — the input/L/R interface geometry.
- `render_programs` — gliders XOR singletons → program board.
- `encode_inputs_cpu`, `to_bin_idx` — observation → bin → `2×2` input blocks.
- `gol_step_cpu`, `run_gol_until_stable_cpu`, `decode_force_cpu` — CPU reference dynamics/decode.
- `TaichiGoLSimulator` — GPU batch simulator (`run_control_batch`).
- `cartpole_step` — standard CartPole dynamics.
- `train`, `eval_genome`, `next_generation`, `mutate_genomes`, `tournament_select`.
- `run_self_tests` — asserts the **Taichi GPU path is bit-identical** to the NumPy reference
  (board, halted flags, L/R counts, force), plus encoding/decoding/physics sanity checks.
- `trace_episode`, `make_videos` — record a rollout as GoL + CartPole MP4s.

## Configuration (paper values)

These are the settings of the run reported in the paper (population 100, 10 generations,
GoL budget `T = 300`; best controller return 417.20, population mean 171.36, evaluation
316.90 ± 121.52 over 20 episodes, non-halting rate ≈ 54.5%).

```python
Cfg(
    N = 100,                 # population size
    n_gens = 10,             # generations
    n_rollouts_train = 5,
    n_rollouts_eval = 20,
    T_max = 300,             # GoL computation budget per control step
    halt_check_every = 20,
    max_control_steps = 500, # CartPole episode cap
    nonhalt_penalty = 0.3,   # note: the dataclass default is 0.9
    Lp = 16, Ls = 10, V = 4, B = 6,
    n_gliders = 3,
    gain = 10.0, b = 1.0, F_clip = 10.0,
    elitism_E = 20, tournament_k = 2,
    p_pos_mut = 0.1, p_active_mut = 0.05, p_singleton_mut = 0.003,
)
```

## Running

Requires a CUDA GPU and `taichi`.

1. Open `test.ipynb`.
2. Run the first cell to build the layout, plot it, run the self-tests, and initialise genomes.
3. Run the training cell (`train(...)`), which saves `runs/best_genome.npz` and `history.csv`.
4. Run the evaluation cell (`eval_genome(...)`).
5. Optionally set `MAKE_VIDEOS = True` and run the video cell.

## Results (paper)

The evolved controller performs clearly above the random baseline (~20), typically adopting a
one-sided balancing strategy. It uses angular information far more effectively than cart
position, and episodes usually end by exceeding the position range. Non-halting rates are high
(often >50%). State retention (feeding the previous board into the next step) was tested and made
the controller unstable, as debris accumulated and interfered with the next observation.

## Limitations

- It is unknown whether this interface + fixed-point halting preserves GoL's universality.
- The genotype→phenotype map is brittle; the controller is not robust.
