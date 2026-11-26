# CEM1D — Continuous 1D Emergent Model for CartPole

A minimal **continuous one-dimensional Emergent Model** that controls **CartPole**. The model is
not a neural network: it is a small local dynamical substrate — a 1D chain of real-valued cells.
At each CartPole step the observation is written into fixed cells, the substrate evolves for a
fixed number of internal microsteps, and one output cell is read to choose the force.

Its distinguishing feature is a **retained, Lamarckian latent state**: only three rule
coefficients are directly trained, while the lattice state itself is carried across rollouts and
inherited across generations.

## Model

- **Latent state:** `s ∈ ℝ^L`, with `L = 19`.
- **Local update rule (radius-1, explicit Euler):**

  ```
  s_j(t+1) = s_j(t) + dt * ( a_l * s_{j-1}(t)
                           + a_c * s_j(t)
                           + a_r * s_{j+1}(t)
                           - gamma * s_j(t)^3 )
  ```

  The cubic term is nonlinear damping to prevent unbounded growth. Only `a = (a_l, a_c, a_r)` are
  trainable; `gamma`, `dt`, and the microstep count `T` are fixed. Boundaries can be `wrap`,
  `reflect`, or `zero`.

- **Encoding `E` (overwrite):** the normalized CartPole state `(x, ẋ, θ, θ̇)` is written into
  four fixed input cells `(3, 6, 12, 15)`.
- **Update:** iterate the rule for a fixed `T = 30` microsteps (no adaptive halting).
- **Decoding `D`:** read the output cell (index `9`) and threshold it:

  ```
  F = -F_max            if z < -eta
  F = +F_max            if z >  eta
  F = random +/-F_max   if |z| <= eta
  ```

  The stochastic dead-zone around zero prevents a degenerate strategy that exploits tiny
  near-zero outputs.

- **Retention (Lamarckian):** `s` is a **non-trainable inherited state**. It is carried across
  rollouts (`s <- final internal state`) and inherited from Parent A during reproduction. Only
  the three coefficients `a` are mutated/crossed over.

- **Invalidity / death:** a candidate update that becomes non-finite or exceeds
  `invalid_threshold` is **not committed**; the agent is frozen and dies. Death is absorbing
  within an episode, and invalid macrosteps do not add reward.

## Training

Black-box genetic algorithm (`general_ga.py`) over the three coefficients:

- Parent A is the base parent, Parent B the sparse donor; non-trainable `s` is copied from
  Parent A (this is what makes state inheritance Lamarckian).
- Elitism + tournament selection + a bank of past-bests + random immigrants; Student-t mutation
  noise on `a`.
- **Fitness** favors robust controllers:

  ```
  F = mean_return / (1 + lambda_std * std_return)
  ```

  computed over `train_rollouts` rollouts with shared per-rollout initial states.

**Critical training invariant:** the lifetime-updated internal state must be synced into the
population *before* reproduction:

```python
fitness, metrics, updated_s = evaluate_population(...)
pop["s"][:] = updated_s     # sync BEFORE ga.tell
ga.tell(fitness)
pop = ga.ask()
```

If this sync is skipped or placed after `ga.tell`, the inherited state becomes stale.

## Files

```
general_ga.py  # generic NumPy GA: VarSpec/GAConfig, elitism, bank & random immigrants,
               # tournament selection, parent-A base / parent-B donor crossover, mutation
utils.py       # EMCartPoleConfig, CEM1D dynamics, CartPole physics, encode/decode,
               # batched rollouts, evaluation, checkpoint/resume, plotting, MP4 export
test.ipynb     # driver: config, train, checkpoint/resume, evaluate, trace/visualize
```

## Configuration (paper values)

These are the settings of the run reported in the paper (population 120, 70 generations;
learned rule `a = (0.796, -0.319, -1.628)`, training return 737.32 ± 104.63, evaluation
734.93 ± 107.00 over 1000 rollouts).

```python
TARGET_GEN = 70                # generations
A = 120                        # population size

EMCartPoleConfig(
    L = 19, T = 30, dt = 0.1, gamma = 0.03,
    invalid_threshold = 50.0,
    boundary_mode = "reflect",
    input_cells = (3, 6, 12, 15),
    output_cell = 9,
    threshold = 0.05,          # dead-zone eta
    F_max = 10.0,
    max_steps = 800,           # CartPole episode cap
    init_mode = "boundary_mixture",  # 70% of episodes start near the position boundary
    fitness_std_penalty = 0.06,
    train_rollouts = 250,
    eval_rollouts = 1000,
)

GAConfig(
    population_size = 120, elitism = 10, tournament_k = 2,
    bank_max = 2000, bank_add_every = 1,
    immigrants_per_gen = 4, random_immigrants_per_gen = 20,
    noise_dist = "student_t", t_df = 2.0, eps_adapt = 0.03,
)
```

Trainable scalars per individual: **3** (`a`). Inherited-state scalars: **19** (`s`).

## Running

Pure NumPy — no GPU required.

1. Open `test.ipynb`.
2. For a fresh run set `RESUME_FROM = None`; to continue, point it at a
   `runs/continuous_em1d_*/latest_ready.npz` checkpoint.
3. Run Cell 1 (setup) and Cell 2 (training). Checkpoints and `history.csv` are written under
   `runs/`.
4. Run the evaluation / visualization cells for the batch eval, the automaton space-time trace,
   the CartPole MP4, and the interface/state figure.

## Results (paper)

CEM1D learns high-performing CartPole policies, reaching returns near the `800` cap and well
above the random baseline (~20), and re-evaluating on fresh rollouts confirms a genuine policy.
A representative learned rule is:

```
a = (0.79581404, -0.3190665, -1.6281466)
```

The controller is very robust to pole-angle deviations but tends to eventually drift out of the
allowed cart-position range — it does not integrate position information as effectively. The
evolved dynamics show localized, glider-like information carriers, with stronger left-moving than
right-moving transport.

## Limitations

- Discrete force output (`±F_max`) with a stochastic dead-zone; no continuous force.
- Fixed `T`, fixed input/output ports, and a predominantly linear rule (small cubic term) limit
  expressivity.
- Single representative run; no multi-seed statistics.
