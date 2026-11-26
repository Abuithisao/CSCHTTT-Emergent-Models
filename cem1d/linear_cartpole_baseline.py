from __future__ import annotations

from pathlib import Path
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from general_ga import GA, GAConfig, VarSpec
from utils import (
    EMCartPoleConfig,
    cartpole_step,
    init_cartpole_states,
    normalize_obs,
    plot_training,
    write_history_csv,
)


SEED = 0
POPULATION_SIZE = 120
TARGET_GENERATIONS = 70
TRAIN_EPISODES_PER_GENERATION = 100
EVALUATION_EPISODES = 200
SAVE_EVERY = 5


def make_run_dir(root: Path = Path("runs")) -> Path:
    run_dir = root / time.strftime("linear4_no_retention_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_ga() -> GA:
    spec = VarSpec(
        name="weights",
        dtype="float32",
        shape=(POPULATION_SIZE, 4),
        trainable=True,
        init_mode="uniform",
        min_val=-4.0,
        max_val=4.0,
        mut_type="additive",
        adaptive_mutation_scale=True,
        mut_prob=0.25,
        sigma_base=0.12,
        crossover_prob=0.1,
    )
    config = GAConfig(
        population_size=POPULATION_SIZE,
        elitism=10,
        tournament_k=2,
        bank_max=2000,
        bank_add_every=1,
        immigrants_per_gen=4,
        random_immigrants_per_gen=20,
        eps_adapt=0.03,
        noise_dist="student_t",
        t_df=2.0,
    )
    ga = GA(config, [spec])
    ga.rng = np.random.default_rng(SEED + 1)
    return ga


def run_episode_batch(
    weights: np.ndarray,
    cfg: EMCartPoleConfig,
    initial_state: np.ndarray,
) -> np.ndarray:
    population_size = weights.shape[0]
    states = np.repeat(np.asarray(initial_state, dtype=np.float32).reshape(1, 4), population_size, axis=0)
    alive = np.ones(population_size, dtype=bool)
    returns = np.zeros(population_size, dtype=np.float32)

    for _ in range(cfg.max_steps):
        active = np.flatnonzero(alive)
        if active.size == 0:
            break

        observations = normalize_obs(states[active], cfg)
        scores = np.einsum("ij,ij->i", weights[active], observations)
        forces = np.where(scores >= 0.0, cfg.F_max, -cfg.F_max).astype(np.float32)

        next_states, done = cartpole_step(states[active], forces, cfg)
        returns[active] += 1.0
        states[active] = next_states
        alive[active] = ~done

    return returns


def evaluate_population(
    weights: np.ndarray,
    cfg: EMCartPoleConfig,
    rng: np.random.Generator,
    n_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weights.shape != (POPULATION_SIZE, 4):
        raise ValueError(f"weights must have shape ({POPULATION_SIZE}, 4), got {weights.shape}.")
    if n_episodes <= 0:
        raise ValueError("n_episodes must be > 0.")

    returns = np.empty((n_episodes, POPULATION_SIZE), dtype=np.float32)
    for episode in range(n_episodes):
        initial_state = init_cartpole_states(1, cfg, rng)[0]
        returns[episode] = run_episode_batch(weights, cfg, initial_state)

    mean_return = returns.mean(axis=0)
    std_return = returns.std(axis=0)
    fitness = mean_return / (1.0 + cfg.fitness_std_penalty * std_return)
    return fitness.astype(np.float32), mean_return, std_return


def save_checkpoint(
    path: Path,
    *,
    weights: np.ndarray,
    best_weights: np.ndarray,
    completed_generation: int,
    history: list[dict],
) -> None:
    temporary_path = path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            weights=weights,
            best_weights=best_weights,
            completed_generation=np.asarray(completed_generation, dtype=np.int32),
            history_json=np.asarray(json.dumps(history)),
        )
    os.replace(temporary_path, path)


def main(run_dir: Path | None = None) -> None:
    cfg = EMCartPoleConfig(
        L=19,
        T=30,
        F_max=10.0,
        max_steps=800,
        init_mode="boundary_mixture",
        boundary_prob=0.7,
        boundary_x_low=1.8,
        boundary_x_high=2.2,
        center_x_range=1.5,
        fitness_std_penalty=0.06,
        train_rollouts=TRAIN_EPISODES_PER_GENERATION,
        eval_rollouts=EVALUATION_EPISODES,
    )
    env_rng = np.random.default_rng(SEED)
    ga = make_ga()
    population = ga.ask(init=True)["weights"]
    if run_dir is None:
        run_dir = make_run_dir()
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_weights: np.ndarray | None = None
    best_fitness = -np.inf

    experiment = {
        "controller": {
            "type": "linear threshold policy",
            "parameters": ["w_x", "w_x_dot", "w_theta", "w_theta_dot"],
            "parameter_count": 4,
            "action_rule": "score = dot(weights, normalized_observation); score >= 0 selects right, otherwise left",
            "retained_internal_state": False,
        },
        "environment": {
            "name": "CartPole with standard classic-control dynamics",
            "integration": "Euler",
            "g": cfg.g,
            "masscart": cfg.masscart,
            "masspole": cfg.masspole,
            "half_pole_length": cfg.length,
            "time_step_seconds": cfg.tau,
            "max_steps_per_episode": cfg.max_steps,
            "position_threshold": cfg.x_threshold,
            "pole_angle_threshold_radians": cfg.theta_threshold,
            "pole_angle_threshold_degrees": float(np.degrees(cfg.theta_threshold)),
            "reset_distribution": {
                "type": cfg.init_mode,
                "boundary_probability": cfg.boundary_prob,
                "boundary_position_abs_range": [cfg.boundary_x_low, cfg.boundary_x_high],
                "center_position_abs_range": [0.0, cfg.center_x_range],
                "velocity_abs_range": cfg.init_dx_range,
                "pole_angle_abs_range": cfg.init_theta_range,
                "angular_velocity_abs_range": cfg.init_dtheta_range,
            },
        },
        "action_space": {
            "type": "Discrete(2)",
            "actions": {"0": f"left force {-cfg.F_max}", "1": f"right force +{cfg.F_max}"},
        },
        "observation": {
            "variables": ["x", "x_dot", "theta", "theta_dot"],
            "normalization_divisors": [cfg.norm_x, cfg.norm_dx, cfg.norm_theta, cfg.norm_dtheta],
            "clip": [-4.0, 4.0],
        },
        "reward": {
            "per_valid_step": 1.0,
            "definition": "One point for each executed CartPole step, including the terminating step.",
            "maximum_per_episode": cfg.max_steps,
            "fitness": f"mean_return / (1 + {cfg.fitness_std_penalty} * std_return)",
        },
        "termination": {
            "cart_position": f"abs(x) > {cfg.x_threshold}",
            "pole_angle": f"abs(theta) > {cfg.theta_threshold} radians",
            "non_finite_state": True,
            "time_limit": cfg.max_steps,
        },
        "seed": {"environment": SEED, "genetic_algorithm": SEED + 1, "final_evaluation": SEED + 2},
        "training": {
            "population_size": POPULATION_SIZE,
            "generations": TARGET_GENERATIONS,
            "episodes_per_generation": TRAIN_EPISODES_PER_GENERATION,
            "total_training_episodes": TARGET_GENERATIONS * TRAIN_EPISODES_PER_GENERATION,
            "evaluation_episodes": EVALUATION_EPISODES,
            "automaton_L_T": "not applicable to a 4-parameter linear controller",
        },
    }
    record_path = run_dir / "experiment_record.json"
    record_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    save_checkpoint(
        run_dir / "latest_ready.npz",
        weights=population,
        best_weights=population[0],
        completed_generation=0,
        history=history,
    )

    print("run dir:", run_dir.resolve(), flush=True)
    print("seed:", experiment["seed"], flush=True)
    print("controller:", experiment["controller"], flush=True)
    print("training:", experiment["training"], flush=True)

    for generation in range(1, TARGET_GENERATIONS + 1):
        fitness, mean_return, std_return = evaluate_population(
            population,
            cfg,
            env_rng,
            TRAIN_EPISODES_PER_GENERATION,
        )
        selected = int(np.argmax(fitness))
        if float(fitness[selected]) > best_fitness:
            best_fitness = float(fitness[selected])
            best_weights = population[selected].copy()

        row = {
            "generation": generation,
            "selected_fitness": float(fitness[selected]),
            "top5_fitness": float(np.mean(np.sort(fitness)[-max(1, int(np.ceil(0.05 * POPULATION_SIZE))):])),
            "population_fitness": float(fitness.mean()),
            "selected_mean_return": float(mean_return[selected]),
            "selected_std_return": float(std_return[selected]),
            "population_mean_return": float(mean_return.mean()),
            "population_std_return": float(std_return.mean()),
            "selected_w_x": float(population[selected, 0]),
            "selected_w_x_dot": float(population[selected, 1]),
            "selected_w_theta": float(population[selected, 2]),
            "selected_w_theta_dot": float(population[selected, 3]),
            "best_fitness_so_far": best_fitness,
        }
        history.append(row)
        write_history_csv(run_dir / "history.csv", history)

        print(
            f"gen={generation:03d}/{TARGET_GENERATIONS} "
            f"fitness={row['selected_fitness']:.3f} "
            f"return={row['selected_mean_return']:.1f}±{row['selected_std_return']:.1f} "
            f"weights={population[selected].round(3).tolist()}",
            flush=True,
        )

        if generation < TARGET_GENERATIONS:
            ga.tell(fitness)
            population = ga.ask()["weights"]

        if generation % SAVE_EVERY == 0 or generation == TARGET_GENERATIONS:
            save_checkpoint(
                run_dir / "latest_ready.npz",
                weights=population,
                best_weights=best_weights,
                completed_generation=generation,
                history=history,
            )

    if best_weights is None:
        raise RuntimeError("Training completed without selecting a best controller.")

    eval_rng = np.random.default_rng(SEED + 2)
    eval_returns = np.empty(EVALUATION_EPISODES, dtype=np.float32)
    for episode in range(EVALUATION_EPISODES):
        initial_state = init_cartpole_states(1, cfg, eval_rng)[0]
        eval_returns[episode] = run_episode_batch(best_weights[None, :], cfg, initial_state)[0]
    eval_mean = float(eval_returns.mean())
    eval_std = float(eval_returns.std())
    evaluation = {
        "episodes": EVALUATION_EPISODES,
        "mean_return": eval_mean,
        "std_return": eval_std,
        "best_return": float(eval_returns.max()),
        "min_return": float(eval_returns.min()),
        "fitness": eval_mean / (1.0 + cfg.fitness_std_penalty * eval_std),
        "weights": best_weights.tolist(),
    }
    experiment["evaluation_result"] = evaluation
    experiment["training"]["completed_generations"] = TARGET_GENERATIONS
    record_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    fig, _ = plot_training(
        [
            {
                "gen": row["generation"],
                "selected_fitness": row["selected_fitness"],
                "top5_fitness": row["top5_fitness"],
                "population_fitness": row["population_fitness"],
                "selected_mean_return": row["selected_mean_return"],
                "selected_std_return": row["selected_std_return"],
                "population_mean_return": row["population_mean_return"],
                "selected_random_action_rate": 0.0,
                "selected_invalid_rate": 0.0,
                "finite_rate": 1.0,
            }
            for row in history
        ]
    )
    if fig is not None:
        fig.savefig(run_dir / "training_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("evaluation:", evaluation, flush=True)
    print("experiment record:", record_path.resolve(), flush=True)
    print("checkpoint:", (run_dir / "latest_ready.npz").resolve(), flush=True)


if __name__ == "__main__":
    main()
