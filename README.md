# Emergent Models — Experiment Code, Part 2

> **Warning:** code written with the assistance of AI, then extensively tested — but not
> guaranteed to be bug-free.

Reproduction code for three experiments from *"Emergent Models: Intelligence from Tiny
Substrates."* Each subfolder is self-contained and trains one model from scratch.

**Part 2** covers the **CartPole control** experiments and the **Game-of-Life arithmetic**
experiment — that is, `gol-arithm`, `gol-cp` and `cem1d`. The EM43 arithmetic experiments
(Part 1) live in their own repository.

An Emergent Model treats learning not as fitting a direct input→output map, but as finding a
configuration of a simple dynamical substrate whose time evolution *computes* the answer:

```
x  ->  E(x)  ->  s0 = p (+) E(x)  --f^T-->  s_T  ->  D(s_T)  ->  y
```

where `f` is a fixed local update rule iterated for `T` steps on a latent state `s`, `p` is a
"program" (a latent configuration), and `(E, D)` is a fixed encode/decode interface. Training
searches over the trainable components with a black-box evolutionary method — no gradients.

## Contents

| Folder       | Experiment                       | Substrate                     | Accel       |
|--------------|----------------------------------|-------------------------------|-------------|
| `gol-arithm` | Arithmetic extrapolation (`x+k`) | Conway's Game of Life (2D CA) | Taichi/CUDA |
| `gol-cp`     | CartPole control                 | Conway's Game of Life (2D CA) | Taichi/CUDA |
| `cem1d`      | CartPole control                 | 1D continuous lattice (ℝ¹⁹)   | NumPy       |

Each folder has its own `README.md` with the model definition, interface, training procedure,
and how to run it.

## What differs across the three

|                 | `gol-arithm`           | `gol-cp`               | `cem1d`                     |
|-----------------|------------------------|------------------------|-----------------------------|
| Latent state    | GoL board `{0,1}`      | GoL board `{0,1}`      | 1D lattice `ℝ¹⁹`            |
| Trainable part  | glider positions       | gliders + sparse bits  | 3 rule coefficients         |
| Rule `f`        | fixed (Conway)         | fixed (Conway)         | trained local ODE step      |
| Halting         | fixed point            | fixed point            | fixed `T = 30` microsteps   |
| State retention | none (reset per input) | none (reset per step)  | **Lamarckian** (retained)   |
| Search method   | **random search**      | mutation-only GA       | full GA (+ bank/immigrants) |
| Encode `E`      | diagonal `2×2` block   | lane/bin `2×2` blocks  | overwrite fixed cells       |
| Decode `D`      | leftmost output block  | (R−L) collector vote   | thresholded output cell     |

## Reproducing the paper's numbers

Every notebook ships with **the hyperparameters of the run reported in the paper**, so running
it top to bottom reproduces that setting directly:

| Experiment   | Paper setting                                     | Reported result                          |
|--------------|---------------------------------------------------|------------------------------------------|
| `gol-arithm` | 40,000 sampled programs, 2 gliders, program side 18 | 100% train / 100% test on the `x ± k` family |
| `gol-cp`     | population 100, 10 generations, `T = 300`         | best 417.20, population mean 171.36; eval 316.90 ± 121.52 |
| `cem1d`      | population 120, 70 generations, `L = 19`, `T = 30` | train 737.32 ± 104.63; eval 734.93 ± 107.00 |

Search is stochastic and unseeded, so an individual run will not land on exactly these numbers;
the settings, not the outcomes, are what is pinned here.

## Common design conventions

- **Program vs input.** The initial state factors as `p ⊕ E(x)`: a task-specific program `p`
  combined with a fixed encoding of the input. Search touches `p` (and, where applicable, the
  rule), never the interface.
- **Fixed, task-independent interface.** `E`, `D`, and the halting rule are chosen once and held
  fixed, so the same interface serves every input of a task.
- **Black-box search only.** Fitness depends only on the decoded output; the internal space-time
  trajectory is never supervised. Any structured "algorithm" that emerges is a byproduct of
  optimizing the scalar error / reward.
- **Unbounded inference.** For the GoL arithmetic experiment, training uses a bounded board and a
  step budget, but evaluation lets the board self-extend and runs to halting, so inference is not
  artificially capped in space or time.

## Requirements

- Python 3.10+
- `numpy`, `matplotlib` (all folders)
- `taichi` with a CUDA-capable GPU (`gol-arithm`, `gol-cp`)
- `pandas`, `imageio`, `imageio-ffmpeg` (plotting / video helpers)
- `jupyter` to run the driver notebooks

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install numpy matplotlib pandas imageio imageio-ffmpeg jupyter taichi
```

## Running

Open the `test.ipynb` in any subfolder and run the cells top to bottom. `gol-arithm` also has
`run_multitask.ipynb`, which sweeps the whole `x ± k` family in one pass.

Each notebook writes its checkpoints, logs, and figures to a local `runs/` directory. **Those
artifacts are deliberately not tracked** (see `.gitignore`) — this repository holds code only.
The trained programs that drive the interactive demos are baked into the demo pages instead.

## Interactive demos

The browser demos run the same models client-side, with pre-trained programs embedded, and need
no Python or GPU. See the project site for links.
