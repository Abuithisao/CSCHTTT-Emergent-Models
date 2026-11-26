from dataclasses import dataclass, asdict, replace
from pathlib import Path
from datetime import datetime
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import taichi as ti


GLIDER_BR = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)

GLIDER_OFFSETS = np.argwhere(GLIDER_BR == 1).astype(np.int32)

_TAICHI_READY = False


def ensure_taichi():
    global _TAICHI_READY
    if not _TAICHI_READY:
        ti.init(arch=ti.cuda)
        _TAICHI_READY = True


@dataclass
class Cfg:
    Lp: int = 16
    Ls: int = 10
    V: int = 4
    B: int = 6
    Lis: int = 4

    N: int = 100
    n_gliders: int = 3
    p_glider_on: float = 0.8
    p_singleton: float = 0.005

    T_max: int = 300
    halt_check_every: int = 20
    max_control_steps: int = 500
    n_gens: int = 100
    n_rollouts_train: int = 5
    n_rollouts_eval: int = 30

    gain: float = 10.0
    b: float = 1.0
    F_clip: float = 10.0
    nonhalt_penalty: float = 0.9

    elitism_E: int = 20
    tournament_k: int = 2
    p_pos_mut: float = 0.1
    p_active_mut: float = 0.05
    p_singleton_mut: float = 0.003
    student_t_df: float = 3.0
    student_t_scale: float = 2.0

    g: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    tau: float = 0.02
    x_threshold: float = 2.4
    theta_threshold: float = 12.0 * np.pi / 180.0

    Lx: float = 2.4
    Ldx: float = 3.0
    Lth: float = 12.0 * np.pi / 180.0
    Ldth: float = 4.0

    init_x_range: float = 1.5
    init_dx_range: float = 0.05
    init_theta_range: float = 0.05
    init_dtheta_range: float = 0.05

    seed: int = 0
    report_frac: float = 0.01
    live_plot: bool = True


@dataclass
class Layout:
    H: int
    W: int
    r0: int
    c0: int
    tile_h: int
    tile_w: int
    input_rows: np.ndarray
    input_cols: np.ndarray
    L_mask: np.ndarray
    R_mask: np.ndarray
    input_mask: np.ndarray
    program_mask: np.ndarray
    spill_mask: np.ndarray


@dataclass
class Genomes:
    glider_pos: np.ndarray
    glider_active: np.ndarray
    singletons: np.ndarray


def make_run_dir(root="runs"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_clean(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def build_layout(cfg):
    if cfg.Ls < 2:
        raise ValueError("Ls must be >= 2 because gliders may spill into the separator.")
    if cfg.halt_check_every < 1:
        raise ValueError("halt_check_every must be >= 1.")

    tile_h = 4
    tile_w = 6

    interface_h = cfg.V * tile_h + (cfg.V - 1) * cfg.Lis
    interface_w = cfg.B * tile_w + (cfg.B - 1) * cfg.Lis

    r0 = cfg.Lp + cfg.Ls
    c0 = cfg.Lp + cfg.Ls

    H = r0 + interface_h + cfg.Ls
    W = c0 + interface_w + cfg.Ls

    input_rows = np.array(
        [r0 + v * (tile_h + cfg.Lis) for v in range(cfg.V)],
        dtype=np.int32,
    )
    input_cols = np.array(
        [c0 + k * (tile_w + cfg.Lis) + 2 for k in range(cfg.B)],
        dtype=np.int32,
    )

    L_mask = np.zeros((H, W), dtype=bool)
    R_mask = np.zeros((H, W), dtype=bool)
    input_mask = np.zeros((H, W), dtype=bool)
    program_mask = np.zeros((H, W), dtype=bool)
    spill_mask = np.zeros((H, W), dtype=bool)

    program_mask[:cfg.Lp, :cfg.Lp] = True
    spill_mask[:cfg.Lp + 2, :cfg.Lp + 2] = True

    for v in range(cfg.V):
        r_tile = r0 + v * (tile_h + cfg.Lis)

        for k in range(cfg.B):
            c_tile = c0 + k * (tile_w + cfg.Lis)

            input_mask[r_tile:r_tile + 2, c_tile + 2:c_tile + 4] = True
            L_mask[r_tile + 2:r_tile + 4, c_tile:c_tile + 2] = True
            R_mask[r_tile + 2:r_tile + 4, c_tile + 4:c_tile + 6] = True

    layout = Layout(
        H=H,
        W=W,
        r0=r0,
        c0=c0,
        tile_h=tile_h,
        tile_w=tile_w,
        input_rows=input_rows,
        input_cols=input_cols,
        L_mask=L_mask,
        R_mask=R_mask,
        input_mask=input_mask,
        program_mask=program_mask,
        spill_mask=spill_mask,
    )

    validate_layout(layout, cfg)
    return layout


def validate_layout(layout, cfg):
    expected = cfg.V * cfg.B * 4

    assert layout.input_mask.shape == (layout.H, layout.W)
    assert layout.L_mask.shape == (layout.H, layout.W)
    assert layout.R_mask.shape == (layout.H, layout.W)

    assert int(layout.input_mask.sum()) == expected
    assert int(layout.L_mask.sum()) == expected
    assert int(layout.R_mask.sum()) == expected

    assert not np.any(layout.input_mask & layout.L_mask)
    assert not np.any(layout.input_mask & layout.R_mask)
    assert not np.any(layout.L_mask & layout.R_mask)

    assert not np.any(layout.spill_mask & layout.input_mask)
    assert not np.any(layout.spill_mask & layout.L_mask)
    assert not np.any(layout.spill_mask & layout.R_mask)

    r = layout.r0
    c = layout.c0

    input_tile = layout.input_mask[r:r + 4, c:c + 6].astype(np.uint8)
    L_tile = layout.L_mask[r:r + 4, c:c + 6].astype(np.uint8)
    R_tile = layout.R_mask[r:r + 4, c:c + 6].astype(np.uint8)

    expected_input = np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    expected_L = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    expected_R = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(input_tile, expected_input)
    assert np.array_equal(L_tile, expected_L)
    assert np.array_equal(R_tile, expected_R)


def plot_layout(layout, cfg):
    # Base board color: light gray everywhere, not white.
    img = np.full((layout.H, layout.W, 3), 0.90, dtype=float)

    # Colors
    spill_color = np.array([0.78, 0.78, 0.78])    # light gray separator/spill region
    program_color = np.array([0.30, 0.30, 0.30])  # dark gray program region
    input_color = np.array([0.10, 0.35, 1.00])    # blue
    L_color = np.array([1.00, 0.10, 0.10])        # red
    R_color = np.array([0.10, 0.70, 0.20])        # green

    # Use the masks already defined by build_layout.
    # This avoids manually redefining the separator size incorrectly.
    img[layout.spill_mask] = spill_color
    img[layout.program_mask] = program_color
    img[layout.input_mask] = input_color
    img[layout.L_mask] = L_color
    img[layout.R_mask] = R_color

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.imshow(
        img,
        interpolation="nearest",
        origin="upper",
        zorder=0,
    )

    ax.set_title("GoL CartPole EM layout")
    ax.set_xlabel("column")
    ax.set_ylabel("row")

    # Program region outline
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            cfg.Lp,
            cfg.Lp,
            fill=False,
            linewidth=2.5,
            edgecolor="black",
            zorder=20,
        )
    )

    # Actual spill/separator region outline, from layout.spill_mask.
    spill_rows, spill_cols = np.where(layout.spill_mask)
    if spill_rows.size > 0:
        spill_r0 = spill_rows.min()
        spill_r1 = spill_rows.max()
        spill_c0 = spill_cols.min()
        spill_c1 = spill_cols.max()

        ax.add_patch(
            plt.Rectangle(
                (spill_c0 - 0.5, spill_r0 - 0.5),
                spill_c1 - spill_c0 + 1,
                spill_r1 - spill_r0 + 1,
                fill=False,
                linewidth=2.0,
                edgecolor="dimgray",
                zorder=20,
            )
        )

    # Interface region outline
    ax.add_patch(
        plt.Rectangle(
            (layout.c0 - 0.5, layout.r0 - 0.5),
            layout.W - layout.c0 - cfg.Ls,
            layout.H - layout.r0 - cfg.Ls,
            fill=False,
            linewidth=2.0,
            edgecolor="black",
            zorder=20,
        )
    )

    # CartPole variable labels, matching encode_inputs_cpu order:
    # state[:, 0] = x
    # state[:, 1] = x_dot
    # state[:, 2] = theta
    # state[:, 3] = theta_dot
    var_labels = [
        r"$x$",
        r"$\dot{x}$",
        r"$\theta$",
        r"$\dot{\theta}$",
    ]

    label_x = layout.c0 - 3

    for v, label in enumerate(var_labels):
        r_tile = layout.r0 + v * (layout.tile_h + cfg.Lis)
        y_center = r_tile + layout.tile_h / 2.0 - 0.5

        ax.text(
            label_x,
            y_center,
            label,
            ha="right",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="black",
            zorder=30,
        )

    # Grid over the full board, drawn on top of the image.
    ax.set_xticks(np.arange(-0.5, layout.W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, layout.H, 1), minor=True)
    ax.grid(
        which="minor",
        color="black",
        linewidth=0.25,
        alpha=0.22,
        zorder=10,
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.show()



def init_genomes(cfg, rng):
    glider_pos = rng.integers(
        0,
        cfg.Lp,
        size=(cfg.N, cfg.n_gliders, 2),
        dtype=np.int32,
    )

    glider_active = (
        rng.random((cfg.N, cfg.n_gliders)) < cfg.p_glider_on
    ).astype(np.uint8)

    singletons = (
        rng.random((cfg.N, cfg.Lp, cfg.Lp)) < cfg.p_singleton
    ).astype(np.uint8)

    return Genomes(
        glider_pos=glider_pos,
        glider_active=glider_active,
        singletons=singletons,
    )


def select_genomes(genomes, idx):
    return Genomes(
        glider_pos=genomes.glider_pos[idx].copy(),
        glider_active=genomes.glider_active[idx].copy(),
        singletons=genomes.singletons[idx].copy(),
    )


def copy_genome(genomes, idx):
    return select_genomes(genomes, np.array([idx], dtype=np.int32))


def repeat_genome(genome, n):
    if genome.glider_pos.shape[0] != 1:
        raise ValueError("repeat_genome expects exactly one genome.")

    return Genomes(
        glider_pos=np.repeat(genome.glider_pos, n, axis=0).copy(),
        glider_active=np.repeat(genome.glider_active, n, axis=0).copy(),
        singletons=np.repeat(genome.singletons, n, axis=0).copy(),
    )


def render_programs(genomes, cfg, layout):
    n = genomes.glider_pos.shape[0]
    S = np.zeros((n, layout.H, layout.W), dtype=np.uint8)
    ids = np.arange(n)

    for g in range(cfg.n_gliders):
        active = genomes.glider_active[:, g].astype(bool)
        active_ids = ids[active]

        if active_ids.size == 0:
            continue

        rows = genomes.glider_pos[active, g, 0]
        cols = genomes.glider_pos[active, g, 1]

        for dr, dc in GLIDER_OFFSETS:
            S[active_ids, rows + dr, cols + dc] = 1

    S[:, :cfg.Lp, :cfg.Lp] ^= genomes.singletons
    return S.astype(np.uint8, copy=False)


def to_bin_idx(x, limit, B):
    y = np.clip(x, -limit, limit)
    k = np.floor(B * (y + limit) / (2.0 * limit)).astype(np.int32)
    return np.clip(k, 0, B - 1)


def encode_inputs_cpu(S, state, cfg, layout):
    bins = np.stack(
        [
            to_bin_idx(state[:, 0], cfg.Lx, cfg.B),
            to_bin_idx(state[:, 1], cfg.Ldx, cfg.B),
            to_bin_idx(state[:, 2], cfg.Lth, cfg.B),
            to_bin_idx(state[:, 3], cfg.Ldth, cfg.B),
        ],
        axis=1,
    )

    ids = np.arange(S.shape[0])

    for v in range(cfg.V):
        r = int(layout.input_rows[v])
        c = layout.input_cols[bins[:, v]]

        S[ids, r, c] = 1
        S[ids, r, c + 1] = 1
        S[ids, r + 1, c] = 1
        S[ids, r + 1, c + 1] = 1

    return S


def gol_step_cpu(S):
    N = np.zeros_like(S, dtype=np.uint8)

    N[:, 1:, 1:] += S[:, :-1, :-1]
    N[:, 1:, :] += S[:, :-1, :]
    N[:, 1:, :-1] += S[:, :-1, 1:]

    N[:, :, 1:] += S[:, :, :-1]
    N[:, :, :-1] += S[:, :, 1:]

    N[:, :-1, 1:] += S[:, 1:, :-1]
    N[:, :-1, :] += S[:, 1:, :]
    N[:, :-1, :-1] += S[:, 1:, 1:]

    alive = S == 1
    born = (~alive) & (N == 3)
    stay = alive & ((N == 2) | (N == 3))

    return (born | stay).astype(np.uint8)


def run_gol_until_stable_cpu(S, cfg):
    halted = np.zeros(S.shape[0], dtype=bool)
    steps_run = 0

    for t in range(1, cfg.T_max + 1):
        S_next = gol_step_cpu(S)

        if (t % cfg.halt_check_every == 0) or (t == cfg.T_max):
            stable = np.all(S_next == S, axis=(1, 2))
            halted |= stable

        S = S_next
        steps_run = t

        if halted.all():
            break

    return S, halted, steps_run


def decode_force_cpu(S, cfg, layout):
    L = S[:, layout.L_mask].sum(axis=1).astype(np.float64)
    R = S[:, layout.R_mask].sum(axis=1).astype(np.float64)

    F = cfg.gain * (R - L) / (R + L + cfg.b)
    F = np.clip(F, -cfg.F_clip, cfg.F_clip)

    return F, L, R


def init_cartpole_states(n, cfg, rng):
    state = np.stack(
        [
            rng.uniform(-cfg.init_x_range, cfg.init_x_range, size=n),
            rng.uniform(-cfg.init_dx_range, cfg.init_dx_range, size=n),
            rng.uniform(-cfg.init_theta_range, cfg.init_theta_range, size=n),
            rng.uniform(-cfg.init_dtheta_range, cfg.init_dtheta_range, size=n),
        ],
        axis=1,
    )

    return state.astype(np.float64)


def cartpole_step(state, force, cfg):
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
    ).astype(np.float64)

    done = (
        (np.abs(next_state[:, 0]) > cfg.x_threshold)
        | (np.abs(next_state[:, 2]) > cfg.theta_threshold)
    )

    return next_state, done


@ti.data_oriented
class TaichiGoLSimulator:
    def __init__(self, cfg, layout):
        ensure_taichi()

        self.N = int(cfg.N)
        self.H = int(layout.H)
        self.W = int(layout.W)
        self.V = int(cfg.V)
        self.Bins = int(cfg.B)
        self.mask_n = int(cfg.V * cfg.B * 4)

        self.T_max = int(cfg.T_max)
        self.halt_check_every = int(cfg.halt_check_every)

        self.gain = float(cfg.gain)
        self.bias = float(cfg.b)
        self.F_clip = float(cfg.F_clip)

        self.Lx = float(cfg.Lx)
        self.Ldx = float(cfg.Ldx)
        self.Lth = float(cfg.Lth)
        self.Ldth = float(cfg.Ldth)

        self.base = ti.field(dtype=ti.u8, shape=(self.N, self.H, self.W))
        self.state = ti.field(dtype=ti.u8, shape=(2, self.N, self.H, self.W))
        self.final_state = ti.field(dtype=ti.u8, shape=(self.N, self.H, self.W))

        self.bufid = ti.field(dtype=ti.i32, shape=())
        self.active_idx = ti.field(dtype=ti.i32, shape=(self.N,))
        self.cp_state = ti.field(dtype=ti.f32, shape=(self.N, 4))

        self.halted = ti.field(dtype=ti.i32, shape=(self.N,))
        self.mismatch = ti.field(dtype=ti.i32, shape=(self.N,))

        self.force = ti.field(dtype=ti.f32, shape=(self.N,))
        self.L_count = ti.field(dtype=ti.i32, shape=(self.N,))
        self.R_count = ti.field(dtype=ti.i32, shape=(self.N,))

        self.input_rows = ti.field(dtype=ti.i32, shape=(self.V,))
        self.input_cols = ti.field(dtype=ti.i32, shape=(self.Bins,))

        self.L_rows = ti.field(dtype=ti.i32, shape=(self.mask_n,))
        self.L_cols = ti.field(dtype=ti.i32, shape=(self.mask_n,))
        self.R_rows = ti.field(dtype=ti.i32, shape=(self.mask_n,))
        self.R_cols = ti.field(dtype=ti.i32, shape=(self.mask_n,))

        self.input_rows.from_numpy(layout.input_rows.astype(np.int32))
        self.input_cols.from_numpy(layout.input_cols.astype(np.int32))

        L_coords = np.argwhere(layout.L_mask).astype(np.int32)
        R_coords = np.argwhere(layout.R_mask).astype(np.int32)

        self.L_rows.from_numpy(L_coords[:, 0])
        self.L_cols.from_numpy(L_coords[:, 1])
        self.R_rows.from_numpy(R_coords[:, 0])
        self.R_cols.from_numpy(R_coords[:, 1])

    def set_base_programs(self, base_programs):
        base_programs = np.asarray(base_programs, dtype=np.uint8)
        if base_programs.shape != (self.N, self.H, self.W):
            raise ValueError(
                f"base_programs must have shape {(self.N, self.H, self.W)}, got {base_programs.shape}"
            )
        self.base.from_numpy(base_programs)

    def run_control_batch(self, active_idx, active_state, return_board=False):
        active_idx = np.asarray(active_idx, dtype=np.int32)
        active_state = np.asarray(active_state, dtype=np.float32)

        B = int(active_idx.shape[0])
        if B == 0:
            out = {
                "force": np.zeros(0, dtype=np.float64),
                "halted": np.zeros(0, dtype=bool),
                "L": np.zeros(0, dtype=np.int32),
                "R": np.zeros(0, dtype=np.int32),
            }
            if return_board:
                out["board"] = np.zeros((0, self.H, self.W), dtype=np.uint8)
            return out

        if B > self.N:
            raise ValueError("active batch is larger than simulator capacity.")
        if active_state.shape != (B, 4):
            raise ValueError(f"active_state must have shape {(B, 4)}, got {active_state.shape}")

        idx_buf = np.zeros(self.N, dtype=np.int32)
        state_buf = np.zeros((self.N, 4), dtype=np.float32)

        idx_buf[:B] = active_idx
        state_buf[:B] = active_state

        self.active_idx.from_numpy(idx_buf)
        self.cp_state.from_numpy(state_buf)

        self._init_batch(B)

        for t in range(1, self.T_max + 1):
            self._step(B)

            if (t % self.halt_check_every == 0) or (t == self.T_max):
                self._check_halted(B)

                halted_np = self.halted.to_numpy()[:B]
                if np.all(halted_np != 0):
                    break

        self._decode(B)

        out = {
            "force": self.force.to_numpy()[:B].astype(np.float64),
            "halted": self.halted.to_numpy()[:B].astype(bool),
            "L": self.L_count.to_numpy()[:B].astype(np.int32),
            "R": self.R_count.to_numpy()[:B].astype(np.int32),
        }

        if return_board:
            self._gather_final(B)
            out["board"] = self.final_state.to_numpy()[:B].astype(np.uint8)

        return out

    @ti.func
    def _to_bin(self, x, limit):
        y = x
        if y < -limit:
            y = -limit
        if y > limit:
            y = limit

        k = ti.cast(ti.floor(ti.cast(self.Bins, ti.f32) * (y + limit) / (2.0 * limit)), ti.i32)

        if k < 0:
            k = 0
        if k >= self.Bins:
            k = self.Bins - 1

        return k

    @ti.kernel
    def _init_batch(self, B: ti.i32):
        self.bufid[None] = 0

        for b, i, j in ti.ndrange(self.N, self.H, self.W):
            if b < B:
                src = self.active_idx[b]
                self.state[0, b, i, j] = self.base[src, i, j]
                self.state[1, b, i, j] = 0

        for b in range(self.N):
            if b < B:
                self.halted[b] = 0
                self.mismatch[b] = 0
                self.force[b] = 0.0
                self.L_count[b] = 0
                self.R_count[b] = 0

        for b in range(self.N):
            if b < B:
                k0 = self._to_bin(self.cp_state[b, 0], self.Lx)
                k1 = self._to_bin(self.cp_state[b, 1], self.Ldx)
                k2 = self._to_bin(self.cp_state[b, 2], self.Lth)
                k3 = self._to_bin(self.cp_state[b, 3], self.Ldth)

                for v in ti.static(range(4)):
                    k = 0
                    if v == 0:
                        k = k0
                    elif v == 1:
                        k = k1
                    elif v == 2:
                        k = k2
                    else:
                        k = k3

                    r = self.input_rows[v]
                    c = self.input_cols[k]

                    self.state[0, b, r, c] = 1
                    self.state[0, b, r, c + 1] = 1
                    self.state[0, b, r + 1, c] = 1
                    self.state[0, b, r + 1, c + 1] = 1

    @ti.kernel
    def _step(self, B: ti.i32):
        src = self.bufid[None]
        dst = src ^ 1

        for b, i, j in ti.ndrange(self.N, self.H, self.W):
            if b < B:
                n = 0

                for di in ti.static(range(-1, 2)):
                    for dj in ti.static(range(-1, 2)):
                        if di != 0 or dj != 0:
                            ii = i + di
                            jj = j + dj

                            if ii >= 0 and ii < self.H and jj >= 0 and jj < self.W:
                                n += self.state[src, b, ii, jj]

                alive = self.state[src, b, i, j]

                if alive == 1:
                    self.state[dst, b, i, j] = 1 if (n == 2 or n == 3) else 0
                else:
                    self.state[dst, b, i, j] = 1 if n == 3 else 0

        self.bufid[None] = dst

    @ti.kernel
    def _check_halted(self, B: ti.i32):
        curr = self.bufid[None]
        prev = curr ^ 1

        for b in range(self.N):
            if b < B:
                self.mismatch[b] = 0

        for b, i, j in ti.ndrange(self.N, self.H, self.W):
            if b < B:
                if self.state[curr, b, i, j] != self.state[prev, b, i, j]:
                    ti.atomic_max(self.mismatch[b], 1)

        for b in range(self.N):
            if b < B:
                if self.mismatch[b] == 0:
                    self.halted[b] = 1

    @ti.kernel
    def _decode(self, B: ti.i32):
        curr = self.bufid[None]

        for b in range(self.N):
            if b < B:
                self.L_count[b] = 0
                self.R_count[b] = 0
                self.force[b] = 0.0

        for b, m in ti.ndrange(self.N, self.mask_n):
            if b < B:
                lr = self.L_rows[m]
                lc = self.L_cols[m]
                rr = self.R_rows[m]
                rc = self.R_cols[m]

                ti.atomic_add(self.L_count[b], ti.cast(self.state[curr, b, lr, lc], ti.i32))
                ti.atomic_add(self.R_count[b], ti.cast(self.state[curr, b, rr, rc], ti.i32))

        for b in range(self.N):
            if b < B:
                L = ti.cast(self.L_count[b], ti.f32)
                R = ti.cast(self.R_count[b], ti.f32)

                f = self.gain * (R - L) / (R + L + self.bias)

                if f > self.F_clip:
                    f = self.F_clip
                if f < -self.F_clip:
                    f = -self.F_clip

                self.force[b] = f

    @ti.kernel
    def _gather_final(self, B: ti.i32):
        curr = self.bufid[None]

        for b, i, j in ti.ndrange(self.N, self.H, self.W):
            if b < B:
                self.final_state[b, i, j] = self.state[curr, b, i, j]


def rollout_population_taichi(genomes, cfg, layout, rng, sim):
    n = genomes.glider_pos.shape[0]

    state = init_cartpole_states(n, cfg, rng)
    alive = np.ones(n, dtype=bool)

    fitness_cp = np.zeros(n, dtype=np.float64)
    fitness_raw = np.zeros(n, dtype=np.float64)
    nonhalt_count = np.zeros(n, dtype=np.float64)
    force_abs_sum = np.zeros(n, dtype=np.float64)
    action_count = np.zeros(n, dtype=np.float64)

    for _ in range(cfg.max_control_steps):
        active_idx = np.where(alive)[0]

        if active_idx.size == 0:
            break

        active_state = state[active_idx]

        out = sim.run_control_batch(active_idx, active_state)
        force = out["force"]
        halted = out["halted"]

        next_state, done = cartpole_step(active_state, force, cfg)

        not_halted = (~halted).astype(np.float64)

        fitness_cp[active_idx] += 1.0
        fitness_raw[active_idx] += 1.0 - cfg.nonhalt_penalty * not_halted

        nonhalt_count[active_idx] += not_halted
        force_abs_sum[active_idx] += np.abs(force)
        action_count[active_idx] += 1.0

        state[active_idx] = next_state
        alive[active_idx] = ~done

    nonhalt_rate = nonhalt_count / np.maximum(action_count, 1.0)
    mean_abs_force = force_abs_sum / np.maximum(action_count, 1.0)

    return {
        "fitness_raw": fitness_raw,
        "fitness_cp": fitness_cp,
        "nonhalt_count": nonhalt_count,
        "nonhalt_rate": nonhalt_rate,
        "mean_abs_force": mean_abs_force,
        "action_count": action_count,
    }


def evaluate_population_taichi(genomes, cfg, layout, rng, n_rollouts, sim):
    n = genomes.glider_pos.shape[0]

    base_programs = render_programs(genomes, cfg, layout)
    sim.set_base_programs(base_programs)

    fitness_raw = np.zeros(n, dtype=np.float64)
    fitness_cp = np.zeros(n, dtype=np.float64)
    nonhalt_rate = np.zeros(n, dtype=np.float64)
    mean_abs_force = np.zeros(n, dtype=np.float64)
    action_count = np.zeros(n, dtype=np.float64)

    for _ in range(n_rollouts):
        out = rollout_population_taichi(genomes, cfg, layout, rng, sim)

        fitness_raw += out["fitness_raw"]
        fitness_cp += out["fitness_cp"]
        nonhalt_rate += out["nonhalt_rate"]
        mean_abs_force += out["mean_abs_force"]
        action_count += out["action_count"]

    fitness_raw /= n_rollouts
    fitness_cp /= n_rollouts
    nonhalt_rate /= n_rollouts
    mean_abs_force /= n_rollouts
    action_count /= n_rollouts

    return {
        "fitness_raw": fitness_raw,
        "fitness_cp": fitness_cp,
        "nonhalt_rate": nonhalt_rate,
        "mean_abs_force": mean_abs_force,
        "action_count": action_count,
    }


def rollout_population_cpu(genomes, cfg, layout, rng, base_programs=None):
    n = genomes.glider_pos.shape[0]

    if base_programs is None:
        base_programs = render_programs(genomes, cfg, layout)

    state = init_cartpole_states(n, cfg, rng)
    alive = np.ones(n, dtype=bool)

    fitness_cp = np.zeros(n, dtype=np.float64)
    fitness_raw = np.zeros(n, dtype=np.float64)
    nonhalt_count = np.zeros(n, dtype=np.float64)
    force_abs_sum = np.zeros(n, dtype=np.float64)
    action_count = np.zeros(n, dtype=np.float64)

    for _ in range(cfg.max_control_steps):
        active_idx = np.where(alive)[0]

        if active_idx.size == 0:
            break

        active_state = state[active_idx]

        S = base_programs[active_idx].copy()
        S = encode_inputs_cpu(S, active_state, cfg, layout)
        S, halted, _ = run_gol_until_stable_cpu(S, cfg)

        force, _, _ = decode_force_cpu(S, cfg, layout)
        next_state, done = cartpole_step(active_state, force, cfg)

        not_halted = (~halted).astype(np.float64)

        fitness_cp[active_idx] += 1.0
        fitness_raw[active_idx] += 1.0 - cfg.nonhalt_penalty * not_halted

        nonhalt_count[active_idx] += not_halted
        force_abs_sum[active_idx] += np.abs(force)
        action_count[active_idx] += 1.0

        state[active_idx] = next_state
        alive[active_idx] = ~done

    nonhalt_rate = nonhalt_count / np.maximum(action_count, 1.0)
    mean_abs_force = force_abs_sum / np.maximum(action_count, 1.0)

    return {
        "fitness_raw": fitness_raw,
        "fitness_cp": fitness_cp,
        "nonhalt_count": nonhalt_count,
        "nonhalt_rate": nonhalt_rate,
        "mean_abs_force": mean_abs_force,
        "action_count": action_count,
    }


def tournament_select(fitness, k, n, rng):
    N = fitness.shape[0]
    winners = np.empty(n, dtype=np.int32)

    for i in range(n):
        cand = rng.integers(0, N, size=k)
        winners[i] = cand[np.argmax(fitness[cand])]

    return winners


def mutate_genomes(genomes, cfg, rng):
    glider_pos = genomes.glider_pos.copy()
    glider_active = genomes.glider_active.copy()
    singletons = genomes.singletons.copy()

    pos_mask = rng.random(glider_pos.shape) < cfg.p_pos_mut
    delta = np.rint(
        cfg.student_t_scale
        * rng.standard_t(cfg.student_t_df, size=glider_pos.shape)
    ).astype(np.int32)

    glider_pos = np.clip(
        glider_pos + pos_mask.astype(np.int32) * delta,
        0,
        cfg.Lp - 1,
    ).astype(np.int32)

    active_mask = rng.random(glider_active.shape) < cfg.p_active_mut
    glider_active ^= active_mask.astype(np.uint8)

    singleton_mask = rng.random(singletons.shape) < cfg.p_singleton_mut
    singletons ^= singleton_mask.astype(np.uint8)

    return Genomes(
        glider_pos=glider_pos,
        glider_active=glider_active,
        singletons=singletons,
    )


def next_generation(genomes, fitness_raw, cfg, rng):
    n = genomes.glider_pos.shape[0]
    E = min(cfg.elitism_E, n)

    order = np.argsort(-fitness_raw)
    elites = select_genomes(genomes, order[:E])

    n_children = n - E
    parent_idx = tournament_select(fitness_raw, cfg.tournament_k, n_children, rng)
    children = select_genomes(genomes, parent_idx)
    children = mutate_genomes(children, cfg, rng)

    return Genomes(
        glider_pos=np.concatenate([elites.glider_pos, children.glider_pos], axis=0),
        glider_active=np.concatenate([elites.glider_active, children.glider_active], axis=0),
        singletons=np.concatenate([elites.singletons, children.singletons], axis=0),
    )


def summarize_generation(gen, metrics, elapsed):
    raw = metrics["fitness_raw"]
    cp = metrics["fitness_cp"]
    nonhalt = metrics["nonhalt_rate"]
    force = metrics["mean_abs_force"]
    actions = metrics["action_count"]

    best_idx = int(np.argmax(raw))

    return {
        "gen": int(gen),
        "best_idx": best_idx,
        "best_raw": float(raw[best_idx]),
        "mean_raw": float(raw.mean()),
        "best_cp": float(cp[best_idx]),
        "mean_cp": float(cp.mean()),
        "best_nonhalt_rate": float(nonhalt[best_idx]),
        "mean_nonhalt_rate": float(nonhalt.mean()),
        "best_steps": float(actions[best_idx]),
        "mean_steps": float(actions.mean()),
        "mean_abs_force": float(force.mean()),
        "elapsed": float(elapsed),
    }


def print_generation(row):
    print(
        f"gen={row['gen']:04d}  "
        f"best_raw={row['best_raw']:.3f}  "
        f"mean_raw={row['mean_raw']:.3f}  "
        f"best_cp={row['best_cp']:.3f}  "
        f"mean_cp={row['mean_cp']:.3f}  "
        f"best_nonhalt={row['best_nonhalt_rate']:.3f}  "
        f"mean_nonhalt={row['mean_nonhalt_rate']:.3f}  "
        f"best_steps={row['best_steps']:.2f}  "
        f"mean_steps={row['mean_steps']:.2f}  "
        f"mean_abs_force={row['mean_abs_force']:.3f}  "
        f"elapsed={row['elapsed']:.1f}s"
    )


def plot_training_live(history):
    if not history:
        return

    from IPython.display import clear_output, display

    xs = [row["gen"] for row in history]
    best_raw = [row["best_raw"] for row in history]
    mean_raw = [row["mean_raw"] for row in history]

    clear_output(wait=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, best_raw, label="best raw")
    ax.plot(xs, mean_raw, label="mean raw")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    display(fig)
    plt.close(fig)


def write_history_csv(path, history):
    path = Path(path)

    if not history:
        return

    keys = list(history[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_genome(path, genome, cfg=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg_json = ""
    if cfg is not None:
        cfg_dict = {k: _json_clean(v) for k, v in asdict(cfg).items()}
        cfg_json = json.dumps(cfg_dict)

    np.savez(
        path,
        glider_pos=genome.glider_pos,
        glider_active=genome.glider_active,
        singletons=genome.singletons,
        cfg_json=np.array(cfg_json),
    )


def load_genome(path):
    z = np.load(path, allow_pickle=True)

    return Genomes(
        glider_pos=z["glider_pos"].astype(np.int32),
        glider_active=z["glider_active"].astype(np.uint8),
        singletons=z["singletons"].astype(np.uint8),
    )


def train(genomes, cfg, layout, rng, run_dir):
    run_dir = Path(run_dir)
    history = []

    best_genome = None
    best_raw_seen = -np.inf

    report_every = max(1, int(cfg.n_gens * cfg.report_frac))
    t0 = time.perf_counter()

    sim = TaichiGoLSimulator(cfg, layout)

    for gen in range(1, cfg.n_gens + 1):
        metrics = evaluate_population_taichi(
            genomes=genomes,
            cfg=cfg,
            layout=layout,
            rng=rng,
            n_rollouts=cfg.n_rollouts_train,
            sim=sim,
        )

        elapsed = time.perf_counter() - t0
        row = summarize_generation(gen, metrics, elapsed)
        history.append(row)

        best_idx = row["best_idx"]

        if row["best_raw"] > best_raw_seen:
            best_raw_seen = row["best_raw"]
            best_genome = copy_genome(genomes, best_idx)
            save_genome(run_dir / "best_genome.npz", best_genome, cfg)

        write_history_csv(run_dir / "history.csv", history)

        should_report = gen == 1 or gen % report_every == 0 or gen == cfg.n_gens
        if should_report:
            if cfg.live_plot:
                plot_training_live(history)
            print_generation(row)

        if gen < cfg.n_gens:
            genomes = next_generation(genomes, metrics["fitness_raw"], cfg, rng)

    if best_genome is None:
        raise RuntimeError("No best genome was found.")

    return history, best_genome


def eval_genome(genome, cfg, layout, rng, n_rollouts=None):
    if n_rollouts is None:
        n_rollouts = cfg.n_rollouts_eval

    genomes = repeat_genome(genome, int(n_rollouts))
    base_programs = render_programs(genomes, cfg, layout)
    out = rollout_population_cpu(genomes, cfg, layout, rng, base_programs=base_programs)

    raw = out["fitness_raw"]
    cp = out["fitness_cp"]

    return {
        "n_rollouts": int(n_rollouts),
        "mean_raw": float(raw.mean()),
        "std_raw": float(raw.std()),
        "best_raw": float(raw.max()),
        "mean_cp": float(cp.mean()),
        "std_cp": float(cp.std()),
        "best_cp": float(cp.max()),
        "mean_nonhalt_rate": float(out["nonhalt_rate"].mean()),
        "mean_abs_force": float(out["mean_abs_force"].mean()),
    }


def trace_episode(genome, cfg, layout, rng):
    if genome.glider_pos.shape[0] != 1:
        raise ValueError("trace_episode expects a single genome.")

    state = init_cartpole_states(1, cfg, rng)

    states = [state[0].copy()]
    forces = []
    halted_flags = []
    gol_frames = []

    fitness_cp = 0.0
    fitness_raw = 0.0

    base_program = render_programs(genome, cfg, layout)

    for _ in range(cfg.max_control_steps):
        S = base_program.copy()
        S = encode_inputs_cpu(S, state, cfg, layout)

        gol_frames.append(S[0].copy())

        halted = np.zeros(1, dtype=bool)

        for t in range(1, cfg.T_max + 1):
            S_next = gol_step_cpu(S)

            if (t % cfg.halt_check_every == 0) or (t == cfg.T_max):
                stable = np.all(S_next == S, axis=(1, 2))
                halted |= stable

            S = S_next
            gol_frames.append(S[0].copy())

            if halted.all():
                break

        force, _, _ = decode_force_cpu(S, cfg, layout)
        state, done = cartpole_step(state, force, cfg)

        forces.append(float(force[0]))
        halted_flags.append(bool(halted[0]))
        states.append(state[0].copy())

        fitness_cp += 1.0
        fitness_raw += 1.0 - cfg.nonhalt_penalty * float(not halted[0])

        if bool(done[0]):
            break

    return {
        "states": np.array(states, dtype=np.float64),
        "forces": np.array(forces, dtype=np.float64),
        "halted": np.array(halted_flags, dtype=bool),
        "gol_frames": gol_frames,
        "fitness_cp": float(fitness_cp),
        "fitness_raw": float(fitness_raw),
    }


def _pad_rgb(img, m=16):
    h, w, c = img.shape
    H = ((h + m - 1) // m) * m
    W = ((w + m - 1) // m) * m

    if H == h and W == w:
        return img

    out = np.ones((H, W, c), dtype=np.uint8) * 255
    out[:h, :w] = img
    return out


def _pad_gray(img, m=16):
    h, w = img.shape
    H = ((h + m - 1) // m) * m
    W = ((w + m - 1) // m) * m

    if H == h and W == w:
        return img

    out = np.zeros((H, W), dtype=np.uint8)
    out[:h, :w] = img
    return out


def _render_cartpole_frame(state, force, cfg, width=480, height=300):
    x = state[0]
    theta = state[2]

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

    ax.add_patch(
        plt.Rectangle(
            (x - cart_w / 2.0, cart_y - cart_h / 2.0),
            cart_w,
            cart_h,
        )
    )

    pole_len = cfg.length * 2.0

    x0 = x
    y0 = cart_y + cart_h / 2.0
    x1 = x0 + pole_len * np.sin(theta)
    y1 = y0 + pole_len * np.cos(theta)

    ax.plot([x0, x1], [y0, y1], linewidth=4)

    ax.text(
        0.02,
        0.95,
        f"x={x:.2f}  theta={theta:.3f}  F={force:.2f}",
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


def make_videos(genome, cfg, layout, rng, run_dir, gol_scale=8, gol_fps=48, cp_fps=24):
    import imageio.v2 as imageio

    run_dir = Path(run_dir)
    trace = trace_episode(genome, cfg, layout, rng)

    gol_path = run_dir / "gol.mp4"
    cp_path = run_dir / "cartpole.mp4"

    with imageio.get_writer(gol_path, fps=gol_fps, codec="libx264", quality=8) as writer:
        for frame in trace["gol_frames"]:
            img = frame.astype(np.uint8) * 255
            img = np.repeat(np.repeat(img, gol_scale, axis=0), gol_scale, axis=1)
            img = _pad_gray(img)
            img = np.stack([img, img, img], axis=-1)
            writer.append_data(img)

    states = trace["states"]
    forces = trace["forces"]

    with imageio.get_writer(cp_path, fps=cp_fps, codec="libx264", quality=8) as writer:
        for i in range(len(forces)):
            img = _render_cartpole_frame(states[i], forces[i], cfg)
            img = _pad_rgb(img)
            writer.append_data(img)

    return {
        "gol": str(gol_path.resolve()),
        "cartpole": str(cp_path.resolve()),
        "fitness_cp": trace["fitness_cp"],
        "fitness_raw": trace["fitness_raw"],
    }


def run_self_tests(cfg, layout):
    validate_layout(layout, cfg)

    rng = np.random.default_rng(123)

    vals = np.array([-999.0, -cfg.Lx, 0.0, cfg.Lx, 999.0])
    bins = to_bin_idx(vals, cfg.Lx, cfg.B)

    assert bins[0] == 0
    assert bins[1] == 0
    assert bins[2] == cfg.B // 2
    assert bins[3] == cfg.B - 1
    assert bins[4] == cfg.B - 1

    S = np.zeros((3, layout.H, layout.W), dtype=np.uint8)
    states = np.array(
        [
            [-cfg.Lx, -cfg.Ldx, -cfg.Lth, -cfg.Ldth],
            [0.0, 0.0, 0.0, 0.0],
            [cfg.Lx, cfg.Ldx, cfg.Lth, cfg.Ldth],
        ],
        dtype=np.float64,
    )

    S = encode_inputs_cpu(S, states, cfg, layout)

    assert np.all(S.sum(axis=(1, 2)) == cfg.V * 4)
    assert not np.any(S[:, layout.L_mask])
    assert not np.any(S[:, layout.R_mask])

    Z = np.zeros((4, layout.H, layout.W), dtype=np.uint8)
    force, _, _ = decode_force_cpu(Z, cfg, layout)
    assert force[0] == 0.0

    left_cell = np.argwhere(layout.L_mask)[0]
    right_cell = np.argwhere(layout.R_mask)[0]

    Z[1, left_cell[0], left_cell[1]] = 1
    Z[2, right_cell[0], right_cell[1]] = 1
    Z[3, left_cell[0], left_cell[1]] = 1
    Z[3, right_cell[0], right_cell[1]] = 1

    force, _, _ = decode_force_cpu(Z, cfg, layout)

    assert force[1] < 0
    assert force[2] > 0
    assert force[3] == 0

    E = np.zeros((1, 8, 8), dtype=np.uint8)
    E2, halted, _ = run_gol_until_stable_cpu(E, cfg)
    assert halted[0]
    assert E2.sum() == 0

    block = np.zeros((1, 8, 8), dtype=np.uint8)
    block[0, 3:5, 3:5] = 1
    _, halted, _ = run_gol_until_stable_cpu(block, cfg)
    assert halted[0]

    single = np.zeros((1, 8, 8), dtype=np.uint8)
    single[0, 0, 0] = 1
    single_next = gol_step_cpu(single)
    assert single_next.sum() == 0

    blinker = np.zeros((1, 8, 8), dtype=np.uint8)
    blinker[0, 3, 2:5] = 1
    _, halted, _ = run_gol_until_stable_cpu(blinker, cfg)
    assert not halted[0]

    gp = np.array([[[cfg.Lp - 1, cfg.Lp - 1]] * cfg.n_gliders], dtype=np.int32)
    ga = np.zeros((1, cfg.n_gliders), dtype=np.uint8)
    sg = np.zeros((1, cfg.Lp, cfg.Lp), dtype=np.uint8)

    genome = Genomes(gp, ga, sg)
    S0 = render_programs(genome, cfg, layout)
    assert S0.sum() == 0

    genome.glider_active[0, 0] = 1
    S1 = render_programs(genome, cfg, layout)

    assert S1.sum() == 5
    assert not np.any(S1[:, layout.input_mask])
    assert not np.any(S1[:, layout.L_mask])
    assert not np.any(S1[:, layout.R_mask])

    sg2 = np.zeros((1, cfg.Lp, cfg.Lp), dtype=np.uint8)
    sg2[0, cfg.Lp - 1, cfg.Lp - 1] = 1

    genome2 = Genomes(gp.copy(), ga.copy(), sg2)
    S2 = render_programs(genome2, cfg, layout)

    assert S2[0, cfg.Lp - 1, cfg.Lp - 1] == (1 ^ S1[0, cfg.Lp - 1, cfg.Lp - 1])

    state = init_cartpole_states(5, cfg, rng)
    force = np.zeros(5, dtype=np.float64)
    state2, done = cartpole_step(state, force, cfg)

    assert np.isfinite(state2).all()
    assert done.shape == (5,)

    cfg_t = replace(cfg, N=4, T_max=min(30, cfg.T_max), halt_check_every=5)
    sim = TaichiGoLSimulator(cfg_t, layout)

    base = (rng.random((cfg_t.N, layout.H, layout.W)) < 0.2).astype(np.uint8)
    states_t = init_cartpole_states(cfg_t.N, cfg_t, rng)
    idx = np.arange(cfg_t.N, dtype=np.int32)

    S_cpu = encode_inputs_cpu(base.copy(), states_t, cfg_t, layout)
    S_cpu, halted_cpu, _ = run_gol_until_stable_cpu(S_cpu, cfg_t)
    force_cpu, L_cpu, R_cpu = decode_force_cpu(S_cpu, cfg_t, layout)

    sim.set_base_programs(base)
    out_t = sim.run_control_batch(idx, states_t, return_board=True)

    assert np.array_equal(S_cpu, out_t["board"])
    assert np.array_equal(halted_cpu, out_t["halted"])
    assert np.array_equal(L_cpu, out_t["L"])
    assert np.array_equal(R_cpu, out_t["R"])
    assert np.allclose(force_cpu, out_t["force"], atol=1e-5)

    fitness = np.zeros(2, dtype=np.float64)
    alive = np.array([True, False])
    active_idx = np.where(alive)[0]
    halted = np.array([False])

    fitness[active_idx] += 1.0
    fitness[active_idx] -= cfg.nonhalt_penalty * (~halted).astype(np.float64)

    assert fitness[1] == 0.0
    assert fitness.mean() == fitness.sum() / 2.0

    print("All self-tests passed.")