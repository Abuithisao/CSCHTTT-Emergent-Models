# GoL-EM — Arithmetic Extrapolation

An Emergent Model built on **Conway's Game of Life** that learns simple integer arithmetic
functions (`x + k`) and extrapolates far beyond its training range. The GoL rule is **fixed**;
training acts only on the **initial condition** (the program), matching the latent-program view:
task-specific computation lives in the initial state, not in the transition rule.

## Model

- **Substrate:** a finite, **bounded** square GoL board (no wraparound; out-of-board neighbours
  count as dead). Values `W = {0,1}`.
- **Layout:** the board is split along each axis into a program region of side `Lp`, a separator
  of width `Ls`, and an encoding region of width `Le`. The encoding origin is `x0 = Lp + Ls`.
- **Program `p` (trainable):** `n_gliders` south-east–moving gliders. Each glider is a
  `(row, col)` anchor in `[0, Lp)` plus an on/off `active` bit. Nothing else is trained — the
  genome is a small set of integers.
- **Encoding `E(x)`:** the integer `x` is written as a single `2×2` live block placed **on the
  diagonal** at `(x0 + s·x, x0 + s·x)`, stride `s = 2`. The block is a still life, so it persists
  until program gliders reach it. Diagonal placement is used because gliders travel diagonally.
- **Update `f`:** the standard Conway rule, applied synchronously.
- **Halting `H`:** the system halts at a **fixed point** (`f(s) = s`).
- **Decoding `D`:** scan the halted board for **isolated `2×2` live blocks** (a 2×2 core with a
  dead surrounding ring). Among valid candidates (column ≥ `x0`), take the **leftmost** and map
  its horizontal position back to a scalar: `y = (col − x0) / s`. Output blocks need not lie on
  the diagonal; unrelated debris is ignored.

A prediction is **valid** iff the run reached a fixed point, did not touch the board boundary,
and produced a decodable output block.

## Training — random search

Training is **plain random search** over glider placements (a genetic algorithm was found to
perform almost identically, so the simpler method is used):

1. sample `P` random programs (glider positions + active bits);
2. evaluate all of them on the training dataset in one GPU-batched pass;
3. keep the best-so-far genome;
4. repeat for `generations` rounds.

The per-genome **loss** (minimized) combines several terms:

```
loss = mae
     + k_invalid * invalid_fraction     # penalize non-halting / undecodable runs
     - k_interact * interact_fraction   # reward the program interacting with the input block
     - k_exact    * exact_fraction      # reward exact integer matches
     + k_multi    * extra_block_count   # penalize producing multiple output blocks
```

Only the decoded output drives the loss; the space-time trajectory is unsupervised.

## Files

```
utils.py       # GoL substrate, encode/decode, Taichi GPU batch simulator,
               # CPU self-extending evaluator, plotting/animation, save helpers
test.ipynb     # driver: hyperparameters, random search, save best genome,
               # single-input inspection, large-range extrapolation eval
```

### Key functions in `utils.py`

- `init_population`, `render_program_board`, `build_single_init_board` — program/genome + board.
- `build_input_interaction_mask` — the ring used for the `interact` reward term.
- `TaichiGoLBatch` — GPU double-buffered batch simulator (`simulate_input`): init, step to
  `T_max`, detect fixed point, decode isolated blocks.
- `evaluate_population_dataset` — batched training loss over the dataset.
- `run_single_input_cpu` / `evaluate_genome_dataset_cpu` — **evaluation** path on a
  **self-extending** board (grows right/down as activity nears the edge) with a large time
  budget, so inference is unbounded in space and time.
- `decode_batch_outputs`, `find_valid_output_blocks` — output decoding / reference semantics.

## Configuration (paper values)

The notebook ships with the values used for the paper's `x ± k` results: a search budget of
**40,000 sampled programs**, **2 gliders**, and a program board of side **18** (`Lp = 16` anchor
range plus the 3×3 glider footprint).

```python
P = 5000              # random genomes sampled per generation
Lp, Ls, Le = 16, 6, 40
s = 2                 # encoding stride
n_gliders = 2
p_active_on = 0.95
T_max = 600           # training step budget
generations = 8       # => 40,000 random genomes total (the paper's n_evals)

k_invalid, k_interact, k_exact, k_multi = 1.6, 0.6, 0.1, 0.01

task_fn = lambda x: x + 2      # swap for other x + k tasks
xs_train = arange(0, 10)       # train on x in [0, 9]
# extrapolation eval: x in [0, 102], T_max = 3000, self-extending board
```

To change the task, edit `task_fn` (e.g. `x + 1`, `x - 1`, `x + 5`). Lower bounds should keep
the target non-negative, since the positional decoder does not represent negative integers.

## Running

Requires a CUDA GPU and `taichi`.

1. Open `test.ipynb`.
2. Run the setup + random-search cells to train and save `runs/best_random_genome.npz`.
3. Run the inspection cell to view a single input's space-time evolution.
4. Run the large-range cell to measure exact-match extrapolation on `x ∈ [0, 102]`.

## Results (paper)

The model learns the `x + k` family and generalizes exactly far outside the training range
(trained on `[0, 9]`, exact on `[0, 200]`). More structured tasks (e.g. an even/odd conditional)
were not reliably learned. Random search matching the GA is itself informative: the
genotype→phenotype map is brittle and the search landscape weakly structured.

## Limitations

- GoL is universal under dedicated constructions, but it is **unknown** whether this minimal
  `2×2`-block interface and fixed-point halting preserve that universality.
- The search landscape is weakly structured; small glider moves can cause large fitness changes.
