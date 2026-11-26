from pathlib import Path
import csv
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from matplotlib import animation
import taichi as ti


# -------------------------
# Fixed motifs / patterns
# -------------------------
GLIDER_BR = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)
GLIDER_LIVE_OFFSETS = [tuple(x) for x in np.argwhere(GLIDER_BR == 1).astype(np.int32)]

ISOLATED_BLOCK_6x6 = np.zeros((6, 6), dtype=np.uint8)
ISOLATED_BLOCK_6x6[2:4, 2:4] = 1

INTERACTION_RING_6x6 = np.ones((6, 6), dtype=np.uint8)
INTERACTION_RING_6x6[2:4, 2:4] = 0


# -------------------------
# Geometry helpers
# -------------------------
def board_size(Lp, Ls, Le):
    return int(Lp) + int(Ls) + int(Le)


def encoding_origin(Lp, Ls):
    return int(Lp) + int(Ls)


def _program_board_side(Lp):
    return int(Lp) + GLIDER_BR.shape[0] - 1


def _validate_board_large_enough(Lp, Ls, Le):
    L = board_size(Lp, Ls, Le)
    min_side = _program_board_side(Lp)
    if L < min_side:
        raise ValueError(
            f"board side L={L} too small for glider anchors in [0,{int(Lp) - 1}] "
            f"with glider size {GLIDER_BR.shape[0]}x{GLIDER_BR.shape[1]}; "
            f"need at least {min_side}"
        )


def _input_block_top_left(x, x0, s, L):
    p = int(x0) + int(s) * int(x)
    if p < 0 or p + 1 >= int(L):
        raise ValueError(
            f"input x={x} does not fit on board: top-left={p}, board side={L}"
        )
    return p


# -------------------------
# Random search population
# -------------------------
def init_population(pop_size, n_gliders, Lp, p_active_on, rng=None):
    if pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if n_gliders <= 0:
        raise ValueError("n_gliders must be positive")
    if Lp <= 0:
        raise ValueError("Lp must be positive")
    if not (0.0 <= p_active_on <= 1.0):
        raise ValueError("p_active_on must be in [0, 1]")

    rng = np.random.default_rng() if rng is None else rng

    positions = rng.integers(
        0,
        int(Lp),
        size=(int(pop_size), int(n_gliders), 2),
        dtype=np.int32,
    )
    active = (
        rng.random((int(pop_size), int(n_gliders))) < float(p_active_on)
    ).astype(np.uint8)

    return positions, active


def render_program_board(positions, active, Lp):
    positions = np.asarray(positions, dtype=np.int32)
    active = np.asarray(active, dtype=np.uint8)

    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(f"positions must have shape (G, 2), got {positions.shape}")
    if active.shape != (positions.shape[0],):
        raise ValueError(f"active must have shape {(positions.shape[0],)}, got {active.shape}")

    side = _program_board_side(Lp)
    board = np.zeros((side, side), dtype=np.uint8)

    for g in range(positions.shape[0]):
        if active[g] == 0:
            continue
        top = int(positions[g, 0])
        left = int(positions[g, 1])
        for dr, dc in GLIDER_LIVE_OFFSETS:
            board[top + dr, left + dc] = 1

    return board


def build_single_init_board(positions, active, x, Lp, Ls, Le, s=2, extra_margin=0):
    positions = np.asarray(positions, dtype=np.int32)
    active = np.asarray(active, dtype=np.uint8)

    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(f"positions must have shape (G, 2), got {positions.shape}")
    if active.shape != (positions.shape[0],):
        raise ValueError(f"active must have shape {(positions.shape[0],)}, got {active.shape}")

    _validate_board_large_enough(Lp, Ls, Le)

    x0 = encoding_origin(Lp, Ls)
    base_L = board_size(Lp, Ls, Le)
    base_p = int(x0) + int(s) * int(x)
    L = max(base_L, base_p + 2 + int(extra_margin))

    p = _input_block_top_left(x=x, x0=x0, s=s, L=L)

    board = np.zeros((L, L), dtype=np.uint8)

    for g in range(positions.shape[0]):
        if active[g] == 0:
            continue
        top = int(positions[g, 0])
        left = int(positions[g, 1])
        for dr, dc in GLIDER_LIVE_OFFSETS:
            board[top + dr, left + dc] = 1

    board[p:p + 2, p:p + 2] = 1
    return board


def build_input_interaction_mask(Lp, Ls, Le, x, s=2, board_side=None):
    _validate_board_large_enough(Lp, Ls, Le)

    L = board_size(Lp, Ls, Le) if board_side is None else int(board_side)
    x0 = encoding_origin(Lp, Ls)
    p = _input_block_top_left(x=x, x0=x0, s=s, L=L)

    mask = np.zeros((L, L), dtype=np.uint8)

    top = p - 2
    left = p - 2

    r0 = max(top, 0)
    c0 = max(left, 0)
    r1 = min(top + INTERACTION_RING_6x6.shape[0], L)
    c1 = min(left + INTERACTION_RING_6x6.shape[1], L)

    pr0 = r0 - top
    pc0 = c0 - left
    pr1 = pr0 + (r1 - r0)
    pc1 = pc0 + (c1 - c0)

    mask[r0:r1, c0:c1] = INTERACTION_RING_6x6[pr0:pr1, pc0:pc1]
    return mask


# -------------------------
# NumPy decode (reference semantics)
# -------------------------
def _isolated_output_matches(boards):
    boards = np.asarray(boards, dtype=np.uint8)
    if boards.ndim != 3:
        raise ValueError(f"boards must have shape (P, H, W), got {boards.shape}")
    windows = sliding_window_view(boards, (6, 6), axis=(1, 2))
    return np.all(windows == ISOLATED_BLOCK_6x6, axis=(-2, -1))


def decode_batch_outputs(boards, x0, s=2):
    boards = np.asarray(boards, dtype=np.uint8)

    matches = _isolated_output_matches(boards)

    row6_idx, col6_idx = np.indices(matches.shape[1:])
    x_inner = col6_idx + 2

    valid_matches = matches & (x_inner[None, :, :] >= int(x0))
    valid = valid_matches.any(axis=(1, 2))

    n_valid_blocks = valid_matches.sum(axis=(1, 2)).astype(np.int32)

    masked_x = np.where(valid_matches, x_inner[None, :, :], np.inf)
    decoded_x = np.full(boards.shape[0], -1, dtype=np.int32)
    decoded_x[valid] = masked_x.min(axis=(1, 2))[valid].astype(np.int32)

    preds = np.zeros(boards.shape[0], dtype=np.float32)
    preds[valid] = (decoded_x[valid] - int(x0)) / float(s)

    return preds, valid, decoded_x, n_valid_blocks


def find_valid_output_blocks(board, x0):
    board = np.asarray(board, dtype=np.uint8)
    if board.ndim != 2:
        raise ValueError(f"board must have shape (H, W), got {board.shape}")

    matches = _isolated_output_matches(board[None, :, :])[0]
    hits = np.argwhere(matches)

    out = []
    for row6, col6 in hits:
        y = int(row6 + 2)
        x = int(col6 + 2)
        if x >= int(x0):
            out.append((x, y))

    out.sort(key=lambda t: (t[0], t[1]))
    return out


# -------------------------
# Taichi batched simulator (training path unchanged)
# -------------------------
@ti.data_oriented
class TaichiGoLBatch:
    def __init__(self, B, Lp, Ls, Le, n_gliders, s=2):
        self.B = int(B)
        self.Lp = int(Lp)
        self.Ls = int(Ls)
        self.Le = int(Le)
        self.L = board_size(Lp, Ls, Le)
        self.x0 = encoding_origin(Lp, Ls)
        self.n_gliders = int(n_gliders)
        self.s = int(s)

        _validate_board_large_enough(Lp, Ls, Le)

        self.positions = ti.field(dtype=ti.i32, shape=(self.B, self.n_gliders, 2))
        self.active = ti.field(dtype=ti.u8, shape=(self.B, self.n_gliders))

        self.state = ti.field(dtype=ti.u8, shape=(2, self.B, self.L, self.L))
        self.bufid = ti.field(dtype=ti.i32, shape=())

        self.ring = ti.field(dtype=ti.u8, shape=(self.L, self.L))
        self.interacted = ti.field(dtype=ti.i32, shape=(self.B,))
        self.boundary_hit = ti.field(dtype=ti.i32, shape=(self.B,))
        self.end_mismatch = ti.field(dtype=ti.i32, shape=(self.B,))
        self.stable_end = ti.field(dtype=ti.i32, shape=(self.B,))

        self.decoded_x = ti.field(dtype=ti.i32, shape=(self.B,))
        self.n_valid_blocks = ti.field(dtype=ti.i32, shape=(self.B,))
        self.dec_valid = ti.field(dtype=ti.i32, shape=(self.B,))

        self.final_board = ti.field(dtype=ti.u8, shape=(self.B, self.L, self.L))

    def set_population(self, positions, active):
        positions = np.asarray(positions, dtype=np.int32)
        active = np.asarray(active, dtype=np.uint8)

        if positions.shape != (self.B, self.n_gliders, 2):
            raise ValueError(
                f"positions must have shape {(self.B, self.n_gliders, 2)}, got {positions.shape}"
            )
        if active.shape != (self.B, self.n_gliders):
            raise ValueError(
                f"active must have shape {(self.B, self.n_gliders)}, got {active.shape}"
            )

        self.positions.from_numpy(positions)
        self.active.from_numpy(active)

    @ti.kernel
    def _init_boards(self, block_p: ti.i32):
        self.bufid[None] = 0

        for b, i, j in ti.ndrange(self.B, self.L, self.L):
            self.state[0, b, i, j] = 0
            self.state[1, b, i, j] = 0

        for b in range(self.B):
            self.interacted[b] = 0
            self.boundary_hit[b] = 0
            self.end_mismatch[b] = 0
            self.stable_end[b] = 0
            self.decoded_x[b] = -1
            self.n_valid_blocks[b] = 0
            self.dec_valid[b] = 0

        for b, g in ti.ndrange(self.B, self.n_gliders):
            if self.active[b, g] != 0:
                top = self.positions[b, g, 0]
                left = self.positions[b, g, 1]
                for k in ti.static(range(len(GLIDER_LIVE_OFFSETS))):
                    dr = GLIDER_LIVE_OFFSETS[k][0]
                    dc = GLIDER_LIVE_OFFSETS[k][1]
                    self.state[0, b, top + dr, left + dc] = 1

        for b, di, dj in ti.ndrange(self.B, 2, 2):
            self.state[0, b, block_p + di, block_p + dj] = 1

    @ti.kernel
    def _set_ring(self, ring_np: ti.types.ndarray()):
        for i, j in ti.ndrange(self.L, self.L):
            self.ring[i, j] = ring_np[i, j]

    @ti.kernel
    def _mark_initial_flags(self):
        for b, i, j in ti.ndrange(self.B, self.L, self.L):
            if self.state[0, b, i, j] != 0:
                if i == self.L - 1 or j == self.L - 1:
                    ti.atomic_max(self.boundary_hit[b], 1)
                if self.ring[i, j] != 0:
                    ti.atomic_max(self.interacted[b], 1)

    @ti.kernel
    def _step(self):
        src = self.bufid[None]
        dst = src ^ 1

        for b, i, j in ti.ndrange(self.B, self.L, self.L):
            n = 0

            if i > 0 and j > 0:
                n += self.state[src, b, i - 1, j - 1]
            if i > 0:
                n += self.state[src, b, i - 1, j]
            if i > 0 and j + 1 < self.L:
                n += self.state[src, b, i - 1, j + 1]
            if j > 0:
                n += self.state[src, b, i, j - 1]
            if j + 1 < self.L:
                n += self.state[src, b, i, j + 1]
            if i + 1 < self.L and j > 0:
                n += self.state[src, b, i + 1, j - 1]
            if i + 1 < self.L:
                n += self.state[src, b, i + 1, j]
            if i + 1 < self.L and j + 1 < self.L:
                n += self.state[src, b, i + 1, j + 1]

            alive = self.state[src, b, i, j]
            if alive:
                self.state[dst, b, i, j] = 1 if (n == 2 or n == 3) else 0
            else:
                self.state[dst, b, i, j] = 1 if n == 3 else 0

        for b, i, j in ti.ndrange(self.B, self.L, self.L):
            if self.state[dst, b, i, j] != 0:
                if i == self.L - 1 or j == self.L - 1:
                    ti.atomic_max(self.boundary_hit[b], 1)
                if self.ring[i, j] != 0:
                    ti.atomic_max(self.interacted[b], 1)

        self.bufid[None] = dst

    @ti.kernel
    def _mark_end_mismatch(self):
        curr = self.bufid[None]
        prev = curr ^ 1

        for b, i, j in ti.ndrange(self.B, self.L, self.L):
            if self.state[curr, b, i, j] != self.state[prev, b, i, j]:
                ti.atomic_max(self.end_mismatch[b], 1)

    @ti.kernel
    def _finalize_stable(self):
        for b in range(self.B):
            self.stable_end[b] = 1 if self.end_mismatch[b] == 0 else 0

    @ti.kernel
    def _decode_final(self):
        curr = self.bufid[None]

        for b in range(self.B):
            self.decoded_x[b] = self.L + 1
            self.n_valid_blocks[b] = 0
            self.dec_valid[b] = 0

        col6_start = self.x0 - 2
        if col6_start < 0:
            col6_start = 0

        for b, row6, col6 in ti.ndrange(self.B, self.L - 5, self.L - 5):
            if col6 < col6_start:
                continue

            ok = 1
            for di, dj in ti.static(ti.ndrange(6, 6)):
                expected = 1 if (2 <= di < 4 and 2 <= dj < 4) else 0
                if self.state[curr, b, row6 + di, col6 + dj] != expected:
                    ok = 0

            if ok == 1:
                x = col6 + 2
                ti.atomic_add(self.n_valid_blocks[b], 1)
                ti.atomic_min(self.decoded_x[b], x)

        for b in range(self.B):
            if self.n_valid_blocks[b] > 0:
                self.dec_valid[b] = 1
            else:
                self.dec_valid[b] = 0
                self.decoded_x[b] = -1

    @ti.kernel
    def _gather_final_board(self):
        curr = self.bufid[None]
        for b, i, j in ti.ndrange(self.B, self.L, self.L):
            self.final_board[b, i, j] = self.state[curr, b, i, j]

    def simulate_input(self, x, T_max, return_final_board=False):
        block_p = _input_block_top_left(x=x, x0=self.x0, s=self.s, L=self.L)
        ring_np = build_input_interaction_mask(self.Lp, self.Ls, self.Le, x, self.s)

        t0 = time.perf_counter()
        self._set_ring(ring_np)
        self._init_boards(block_p)
        self._mark_initial_flags()
        ti.sync()
        t1 = time.perf_counter()

        for _ in range(int(T_max)):
            self._step()
        ti.sync()
        t2 = time.perf_counter()

        self._mark_end_mismatch()
        self._finalize_stable()
        self._decode_final()
        ti.sync()
        t3 = time.perf_counter()

        boundary_hit_np = self.boundary_hit.to_numpy().astype(np.bool_)
        interacted_np = self.interacted.to_numpy().astype(np.float32)
        stable_end_np = self.stable_end.to_numpy().astype(np.bool_)
        decoded_x_np = self.decoded_x.to_numpy().astype(np.int32)
        n_valid_blocks_np = self.n_valid_blocks.to_numpy().astype(np.int32)
        dec_valid_np = self.dec_valid.to_numpy().astype(np.bool_)
        t4 = time.perf_counter()

        out = {
            "boundary_hit": boundary_hit_np,
            "interacted": interacted_np,
            "stable_end": stable_end_np,
            "decoded_x": decoded_x_np,
            "n_valid_blocks": n_valid_blocks_np,
            "dec_valid": dec_valid_np,
            "times": {
                "encoding": t1 - t0,
                "sim": t2 - t1,
                "decoding": t3 - t2,
                "copy_small": t4 - t3,
            },
        }

        if return_final_board:
            self._gather_final_board()
            ti.sync()
            out["final_board"] = self.final_board.to_numpy()

        return out


# -------------------------
# Dataset evaluation (training path unchanged)
# -------------------------
def evaluate_population_on_input(simulator, x, y, T_max):
    out = simulator.simulate_input(
        x=x,
        T_max=T_max,
        return_final_board=False,
    )

    boundary_hit = out["boundary_hit"]
    interacted = out["interacted"]
    stable_end = out["stable_end"]
    decoded_x = out["decoded_x"]
    n_valid_blocks = out["n_valid_blocks"]
    dec_valid = out["dec_valid"]

    preds = np.zeros(simulator.B, dtype=np.float32)
    valid_dec = dec_valid & (decoded_x >= 0)
    preds[valid_dec] = (decoded_x[valid_dec] - simulator.x0) / float(simulator.s)

    valid = (~boundary_hit) & stable_end & dec_valid

    abs_err = np.zeros(simulator.B, dtype=np.float32)
    abs_err[valid] = np.abs(preds[valid] - float(y))

    invalid_flag = (~valid).astype(np.float32)

    exact = np.zeros(simulator.B, dtype=np.float32)
    exact[valid] = (preds[valid] == float(y)).astype(np.float32)

    multi = np.maximum(n_valid_blocks - 1, 0).astype(np.float32)

    return abs_err, invalid_flag, exact, interacted, multi, {
        "encoding": out["times"]["encoding"],
        "sim": out["times"]["sim"],
        "decoding": out["times"]["decoding"] + out["times"]["copy_small"],
    }


def evaluate_population_dataset(
    simulator,
    positions,
    active,
    xs,
    ys,
    T_max,
    k_invalid=50.0,
    k_interact=0.0,
    k_exact=0.1,
    k_multi=0.05,
):
    positions = np.asarray(positions, dtype=np.int32)
    active = np.asarray(active, dtype=np.uint8)
    xs = list(xs)
    ys = np.asarray(ys, dtype=np.float32)

    if len(xs) != ys.shape[0]:
        raise ValueError(f"xs and ys must have same length, got {len(xs)} and {ys.shape[0]}")

    simulator.set_population(positions, active)

    P = simulator.B
    sum_abs_err = np.zeros(P, dtype=np.float32)
    sum_invalid = np.zeros(P, dtype=np.float32)
    sum_exact = np.zeros(P, dtype=np.float32)
    sum_interact = np.zeros(P, dtype=np.float32)
    sum_multi = np.zeros(P, dtype=np.float32)

    times = {
        "encoding": 0.0,
        "sim": 0.0,
        "decoding": 0.0,
    }

    for x, y in zip(xs, ys):
        abs_err, invalid_flag, exact, interacted, multi, time_parts = evaluate_population_on_input(
            simulator=simulator,
            x=x,
            y=y,
            T_max=T_max,
        )
        sum_abs_err += abs_err
        sum_invalid += invalid_flag
        sum_exact += exact
        sum_interact += interacted
        sum_multi += multi

        times["encoding"] += time_parts["encoding"]
        times["sim"] += time_parts["sim"]
        times["decoding"] += time_parts["decoding"]

    n = float(len(xs))
    mae = sum_abs_err / n
    invalid = sum_invalid / n
    exact = sum_exact / n
    interact = sum_interact / n
    multi = sum_multi / n

    loss = (
        mae
        + float(k_invalid) * invalid
        - float(k_interact) * interact
        - float(k_exact) * exact
        + float(k_multi) * multi
    )

    return {
        "loss": loss.astype(np.float32),
        "mae": mae.astype(np.float32),
        "invalid": invalid.astype(np.float32),
        "exact": exact.astype(np.float32),
        "interact": interact.astype(np.float32),
        "multi": multi.astype(np.float32),
    }, times


# -------------------------
# Save helpers (training unchanged)
# -------------------------
def save_genome(path, positions, active, metrics, config, Lp):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    program_board = render_program_board(positions, active, Lp)

    np.savez(
        path,
        positions=np.asarray(positions, dtype=np.int32),
        active=np.asarray(active, dtype=np.uint8),
        program_board=np.asarray(program_board, dtype=np.uint8),
        best_loss=np.float32(metrics["loss"]),
        best_mae=np.float32(metrics["mae"]),
        best_invalid=np.float32(metrics["invalid"]),
        best_exact=np.float32(metrics["exact"]),
        best_interact=np.float32(metrics["interact"]),
        best_multi=np.float32(metrics["multi"]),
        config=np.array(config, dtype=object),
    )


def write_csv_rows(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# -------------------------
# New default CPU eval (self-extending)
# -------------------------
def gol_step_single(board):
    board = np.asarray(board, dtype=np.uint8)
    H, W = board.shape
    padded = np.pad(board, ((1, 1), (1, 1)), mode="constant")

    neighbors = (
        padded[0:H,     0:W]     + padded[0:H,     1:W + 1] + padded[0:H,     2:W + 2] +
        padded[1:H + 1, 0:W]                                 + padded[1:H + 1, 2:W + 2] +
        padded[2:H + 2, 0:W]     + padded[2:H + 2, 1:W + 1] + padded[2:H + 2, 2:W + 2]
    )

    alive = board == 1
    nxt = (alive & ((neighbors == 2) | (neighbors == 3))) | ((~alive) & (neighbors == 3))
    return nxt.astype(np.uint8)


def _needs_extension(board, margin_trigger=3):
    H, W = board.shape
    if H != W:
        raise ValueError(f"board must be square, got shape {board.shape}")
    trigger_idx = H - int(margin_trigger)
    if trigger_idx < 0:
        raise ValueError("margin_trigger is too large for board size")
    return bool((board[trigger_idx, :] != 0).any() or (board[:, trigger_idx] != 0).any())


def _extend_square(arr, growth=10):
    arr = np.asarray(arr)
    H, W = arr.shape
    if H != W:
        raise ValueError(f"array must be square, got shape {arr.shape}")
    new_side = H + int(growth)
    out = np.zeros((new_side, new_side), dtype=arr.dtype)
    out[:H, :W] = arr
    return out


def _apply_eval_layout(ax, board_side, Lp, Ls, s):
    x0 = encoding_origin(Lp, Ls)
    L = int(board_side)

    # Program overlay
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            Lp,
            Lp,
            facecolor="gray",
            alpha=0.18,
            edgecolor="none",
        )
    )

    # Separator overlays
    ax.add_patch(
        plt.Rectangle(
            (Lp - 0.5, -0.5),
            Ls,
            x0,
            facecolor="green",
            alpha=0.12,
            edgecolor="none",
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (-0.5, Lp - 0.5),
            Lp,
            Ls,
            facecolor="green",
            alpha=0.12,
            edgecolor="none",
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (Lp - 0.5, Lp - 0.5),
            Ls,
            Ls,
            facecolor="green",
            alpha=0.12,
            edgecolor="none",
        )
    )

    # Delimiter lines only inside upper-left prep region
    for xv in [Lp - 0.5, x0 - 0.5]:
        ax.plot([xv, xv], [-0.5, x0 - 0.5], color="black", linewidth=1.0, alpha=0.7)
    for yv in [Lp - 0.5, x0 - 0.5]:
        ax.plot([-0.5, x0 - 0.5], [yv, yv], color="black", linewidth=1.0, alpha=0.7)

    # X ticks in encoding coordinates, centered on 2x2 markers
    enc_tick_labels = np.arange(0, ((L - 1 - x0) // s) + 1, 5)
    enc_tick_pos = x0 + 0.5 + s * enc_tick_labels
    mask = enc_tick_pos < L

    ax.set_xticks(enc_tick_pos[mask])
    ax.set_xticklabels([str(v) for v in enc_tick_labels[mask]], fontsize=8)
    ax.tick_params(axis="x", which="major", pad=6, length=6, width=1.0)

    # No y ticks
    ax.set_yticks([])

    # Grid every 2 cells
    grid2 = np.arange(-0.5, L, 2)
    ax.set_xticks(grid2, minor=True)
    ax.set_yticks(grid2, minor=True)
    ax.grid(which="minor", color="steelblue", linewidth=0.5, alpha=0.28)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="y", which="minor", left=False)

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(L - 0.5, -0.5)


def plot_board(board, title=None, ax=None, *, Lp=None, Ls=None, s=2):
    board = np.asarray(board, dtype=np.uint8)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.imshow(
        board,
        cmap="binary",
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    if Lp is not None and Ls is not None:
        _apply_eval_layout(ax, board.shape[0], int(Lp), int(Ls), int(s))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_trajectory_strip(frames, cols=5, figsize_per_panel=3, *, Lp=None, Ls=None, s=2):
    n = len(frames)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_panel, rows * figsize_per_panel),
        squeeze=False,
    )

    for i, ax in enumerate(axes.ravel()):
        if i < n:
            plot_board(
                frames[i],
                title=f"t={i}",
                ax=ax,
                Lp=Lp,
                Ls=Ls,
                s=s,
            )
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    return fig, axes


def animate_trajectory(frames, interval=200, figsize=(5, 5), *, Lp=None, Ls=None, s=2):
    fig, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(
        np.asarray(frames[0], dtype=np.uint8),
        cmap="binary",
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    if Lp is not None and Ls is not None:
        _apply_eval_layout(ax, np.asarray(frames[0]).shape[0], int(Lp), int(Ls), int(s))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("t=0")

    def update(i):
        image.set_data(np.asarray(frames[i], dtype=np.uint8))
        ax.set_title(f"t={i}")
        return (image,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=True,
    )
    return fig, ani


def save_animation_gif(ani, path, fps=5):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))


def save_frames_pngs(frames, out_dir, prefix="frame", *, Lp=None, Ls=None, s=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        fig, ax = plot_board(frame, title=f"t={i}", Lp=Lp, Ls=Ls, s=s)
        fig.savefig(out_dir / f"{prefix}_{i:04d}.png", bbox_inches="tight")
        plt.close(fig)


def run_single_input_cpu(
    positions,
    active,
    x,
    target,
    Lp,
    Ls,
    Le,
    T_max,
    s=2,
    *,
    growth=10,
    margin_trigger=3,
    plot=False,
    store_frames=None,
):
    if store_frames is None:
        store_frames = bool(plot)

    x0 = encoding_origin(Lp, Ls)

    board = build_single_init_board(
        positions=positions,
        active=active,
        x=x,
        Lp=Lp,
        Ls=Ls,
        Le=Le,
        s=s,
        extra_margin=growth,
    )
    ring = build_input_interaction_mask(
        Lp=Lp,
        Ls=Ls,
        Le=Le,
        x=x,
        s=s,
        board_side=board.shape[0],
    )

    frames = [board.copy()] if store_frames else None

    interacted = bool(((board & ring) != 0).any())
    n_extensions = 0
    steps_run = 0
    stable_end = False

    curr = board.copy()
    prev = curr.copy()

    for t in range(1, int(T_max) + 1):
        nxt = gol_step_single(curr)
        interacted = interacted or bool(((nxt & ring) != 0).any())

        if _needs_extension(nxt, margin_trigger=margin_trigger):
            nxt = _extend_square(nxt, growth=growth)
            ring = _extend_square(ring, growth=growth)
            n_extensions += 1

        if store_frames:
            frames.append(nxt.copy())

        steps_run = t
        prev = curr
        curr = nxt

        if np.array_equal(curr, prev):
            stable_end = True
            break

    valid_blocks = find_valid_output_blocks(curr, x0)
    if valid_blocks:
        chosen_block = valid_blocks[0]
        pred = (chosen_block[0] - x0) / float(s)
    else:
        chosen_block = None
        pred = None

    dec_valid = pred is not None
    valid = stable_end and dec_valid
    exact = bool(valid and pred == float(target))

    result = {
        "frames": frames,
        "final_board": curr,
        "prev_board": prev,
        "stable_end": stable_end,
        "boundary_hit": bool(n_extensions > 0),  # compatibility field
        "interacted": interacted,
        "pred": pred,
        "chosen_block": chosen_block,
        "valid_blocks": valid_blocks,
        "n_valid_blocks": len(valid_blocks),
        "valid": valid,
        "exact": exact,
        "accuracy": float(exact),
        "steps_run": int(steps_run),
        "n_extensions": int(n_extensions),
        "final_side": int(curr.shape[0]),
    }

    if plot:
        plot_board(
            result["final_board"],
            title=f"x={x} | pred={result['pred']} | exact={result['exact']}",
            Lp=Lp,
            Ls=Ls,
            s=s,
        )
        plt.show()

        if store_frames and result["frames"] is not None:
            plot_trajectory_strip(
                result["frames"],
                cols=5,
                figsize_per_panel=3,
                Lp=Lp,
                Ls=Ls,
                s=s,
            )
            plt.show()

    return result


def evaluate_genome_dataset_cpu(
    positions,
    active,
    xs,
    ys,
    Lp,
    Ls,
    Le,
    T_max,
    s=2,
    *,
    growth=10,
    margin_trigger=3,
    plot=False,
    progress=True,
):
    xs = list(xs)
    ys = np.asarray(ys, dtype=np.float32)

    if len(xs) != ys.shape[0]:
        raise ValueError(f"xs and ys must have same length, got {len(xs)} and {ys.shape[0]}")

    records = []

    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        result = run_single_input_cpu(
            positions=positions,
            active=active,
            x=x,
            target=float(y),
            Lp=Lp,
            Ls=Ls,
            Le=Le,
            T_max=T_max,
            s=s,
            growth=growth,
            margin_trigger=margin_trigger,
            plot=plot,
            store_frames=plot,
        )

        rec = {
            "idx": i,
            "x": int(x),
            "target": float(y),
            "pred": result["pred"],
            "exact": bool(result["exact"]),
            "valid": bool(result["valid"]),
            "stable_end": bool(result["stable_end"]),
            "n_extensions": int(result["n_extensions"]),
            "final_side": int(result["final_side"]),
            "steps_run": int(result["steps_run"]),
            "n_valid_blocks": int(result["n_valid_blocks"]),
        }
        records.append(rec)

        if progress:
            print(
                f"[{i}/{len(xs)}] "
                f"x={rec['x']}  "
                f"target={rec['target']}  "
                f"pred={rec['pred']}  "
                f"exact={rec['exact']}  "
                f"valid={rec['valid']}  "
                f"stable={rec['stable_end']}  "
                f"ext={rec['n_extensions']}  "
                f"side={rec['final_side']}  "
                f"steps={rec['steps_run']}"
            )

    valid_mask = np.array([r["valid"] for r in records], dtype=bool)
    exact_mask = np.array([r["exact"] for r in records], dtype=bool)

    preds_valid = np.array([r["pred"] for r in records if r["valid"]], dtype=np.float32)
    ys_valid = np.array([r["target"] for r in records if r["valid"]], dtype=np.float32)

    if preds_valid.size > 0:
        mae_valid = float(np.mean(np.abs(preds_valid - ys_valid)))
    else:
        mae_valid = np.nan

    summary = {
        "n": int(len(records)),
        "valid_rate": float(valid_mask.mean()) if len(records) > 0 else np.nan,
        "exact_rate": float(exact_mask.mean()) if len(records) > 0 else np.nan,
        "mae_valid": mae_valid,
        "mean_extensions": float(np.mean([r["n_extensions"] for r in records])) if len(records) > 0 else np.nan,
        "max_final_side": int(max(r["final_side"] for r in records)) if len(records) > 0 else None,
        "max_steps_run": int(max(r["steps_run"] for r in records)) if len(records) > 0 else None,
    }

    return summary, records