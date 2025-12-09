from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from tqdm import tqdm


# ----------------------------
# Config & paths
# ----------------------------

GRID_SIZE: int = 32
TIME_FRAMES: int = 100
SAMPLES_PER_CLASS: int = 100
RADIUS_RANGE: Tuple[float, float] = (4.0, 10.0)
RADIUS_DRAW_RANGE: Tuple[float, float] = (1.0, 6.0)
LABEL_NAMES: List[str] = [
    "left_right",
    "right_left",
    "top_bottom",
    "bottom_top",
    "clockwise",
    "counter_clockwise",
    "random",
]

DATA_DIR = Path("dati")
OUTPUT_FILE = DATA_DIR / "trajectory_spike_encoded.npz"

# Optional: set to an int for reproducibility, or None for stochastic runs
SEED: int | None = 42


# ----------------------------
# Drawing & noise
# ----------------------------

def draw_object(
    frame: np.ndarray,
    x: int,
    y: int,
    radius: float,
    shape: str = "circle",
    rng: Generator | None = None,
) -> np.ndarray:
    """Draw a circle or ellipse on a binary frame."""
    rng = rng or default_rng()
    xx, yy = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="xy")

    if shape == "circle":
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    else:  # ellipse
        rx = radius
        ry = max(1.0, radius - rng.integers(0, 5))
        mask = ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2 <= 1.0

    frame[mask] = 1
    return frame


def add_position_noise(x: float, y: float, amount: float, rng: Generator) -> Tuple[float, float]:
    """Jitter the position by uniform noise in [-amount, amount]."""
    return x + rng.uniform(-amount, amount), y + rng.uniform(-amount, amount)


def add_frame_noise(frame: np.ndarray, prob: float, rng: Generator) -> np.ndarray:
    """Flip a small random subset of pixels to 1 with given probability."""
    noise = rng.random(frame.shape) < prob
    return np.clip(frame + noise.astype(frame.dtype), 0, 1)


def wrap_coordinates(x: float, y: float, size: int) -> Tuple[float, float]:
    """Wrap coordinates on a toroidal grid."""
    return x % size, y % size


# ----------------------------
# Trajectories
# ----------------------------

def linear_trajectory(
    start: Tuple[float, float],
    direction: Tuple[float, float],
    frames: int,
    noise_std: float,
    size: int,
    rng: Generator,
) -> List[Tuple[float, float]]:
    """Generate a noisy linear path across the grid."""
    x0, y0 = start
    dx, dy = direction
    path: List[Tuple[float, float]] = []
    for t in range(frames):
        x = x0 + dx * t + rng.normal(0.0, noise_std)
        y = y0 + dy * t + rng.normal(0.0, noise_std)
        x, y = wrap_coordinates(x, y, size)
        path.append((x, y))
    return path


def circular_trajectory(
    center: Tuple[float, float],
    radius: float,
    frames: int,
    clockwise: bool,
    noise_std: float,
    size: int,
    rng: Generator,
) -> List[Tuple[float, float]]:
    """Generate a noisy circular (clockwise/counter-clockwise) path."""
    angles = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    if not clockwise:
        angles = -angles

    cx, cy = center
    path: List[Tuple[float, float]] = []
    for a in angles:
        x = cx + radius * np.cos(a) + rng.normal(0.0, noise_std)
        y = cy + radius * np.sin(a) + rng.normal(0.0, noise_std)
        x, y = wrap_coordinates(x, y, size)
        path.append((x, y))
    return path


def random_trajectory(frames: int, size: int, rng: Generator) -> List[Tuple[float, float]]:
    """Sample independent random positions across the grid."""
    return [(rng.uniform(0, size - 1), rng.uniform(0, size - 1)) for _ in range(frames)]


# ----------------------------
# Sample & dataset generation
# ----------------------------

def generate_sample(
    trajectory_type: str,
    grid_size: int = GRID_SIZE,
    frames: int = TIME_FRAMES,
    radius_range: Tuple[float, float] = RADIUS_RANGE,
    rng: Generator | None = None,
) -> np.ndarray:
    """Generate a single (T, H, W) binary sample for the given trajectory type."""
    rng = rng or default_rng()
    sample = np.zeros((frames, grid_size, grid_size), dtype=np.uint8)

    radius = rng.uniform(*radius_range)
    radius_draw = rng.uniform(*RADIUS_DRAW_RANGE)
    shape = rng.choice(["circle", "ellipse"], p=[0.7, 0.3])
    margin = int(np.ceil(radius))

    if trajectory_type in {"left_right", "right_left", "top_bottom", "bottom_top"}:
        x = rng.integers(margin, grid_size - margin)
        y = rng.integers(margin, grid_size - margin)

    if trajectory_type == "left_right":
        path = linear_trajectory((x, y), (1, 0), frames, noise_std=1.2, size=grid_size, rng=rng)
    elif trajectory_type == "right_left":
        path = linear_trajectory((x, y), (-1, 0), frames, noise_std=1.2, size=grid_size, rng=rng)
    elif trajectory_type == "top_bottom":
        path = linear_trajectory((x, y), (0, 1), frames, noise_std=1.2, size=grid_size, rng=rng)
    elif trajectory_type == "bottom_top":
        path = linear_trajectory((x, y), (0, -1), frames, noise_std=1.2, size=grid_size, rng=rng)
    elif trajectory_type == "clockwise":
        cx = rng.integers(margin, grid_size - margin)
        cy = rng.integers(margin, grid_size - margin)
        path = circular_trajectory((cx, cy), radius, frames, clockwise=True, noise_std=0.8, size=grid_size, rng=rng)
    elif trajectory_type == "counter_clockwise":
        cx = rng.integers(margin, grid_size - margin)
        cy = rng.integers(margin, grid_size - margin)
        path = circular_trajectory((cx, cy), radius, frames, clockwise=False, noise_std=0.8, size=grid_size, rng=rng)
    elif trajectory_type == "random":
        path = random_trajectory(frames, grid_size, rng)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # Optional random start delay (blank frames at the beginning)
    start_delay = int(rng.integers(0, 11))
    path = path[: frames - start_delay]

    for t, (x, y) in enumerate(path):
        nx, ny = add_position_noise(x, y, amount=1.2, rng=rng)
        xi = int(np.clip(round(nx), 0, grid_size - 1))
        yi = int(np.clip(round(ny), 0, grid_size - 1))

        frame = np.zeros((grid_size, grid_size), dtype=np.uint8)
        frame = draw_object(frame, xi, yi, radius_draw, shape=shape, rng=rng)
        frame = add_frame_noise(frame, prob=0.003, rng=rng)
        sample[start_delay + t] = frame

    return sample


def generate_dataset(
    labels: List[str] = LABEL_NAMES,
    samples_per_class: int = SAMPLES_PER_CLASS,
    grid_size: int = GRID_SIZE,
    frames: int = TIME_FRAMES,
    radius_range: Tuple[float, float] = RADIUS_RANGE,
    seed: int | None = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (N, T, H, W) dataset + label array (N,) over all classes."""
    rng = default_rng(seed)
    data: List[np.ndarray] = []
    lab: List[int] = []

    for label_idx, trajectory in enumerate(labels):
        desc = f"Generating '{trajectory}'"
        for _ in tqdm(range(samples_per_class), desc=desc):
            sample = generate_sample(
                trajectory,
                grid_size=grid_size,
                frames=frames,
                radius_range=radius_range,
                rng=rng,
            )
            data.append(sample)
            lab.append(label_idx)

    data_arr = np.asarray(data, dtype=np.uint8)
    labels_arr = np.asarray(lab, dtype=np.uint8)
    return data_arr, labels_arr


def save_dataset(data: np.ndarray, labels: np.ndarray, out_path: Path = OUTPUT_FILE) -> None:
    """Save compressed NPZ with 'data' and 'labels' arrays."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=data, labels=labels)
    print(f"Dataset saved to: {out_path}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data, labels = generate_dataset()

    # data: (N, T, H, W) -> (N, H, W, T) -> (N, H*W, T)
    N, T, H, W = data.shape
    data = data.transpose(0, 2, 3, 1).reshape(N, H * W, T)

    save_dataset(data, labels)

if __name__ == "__main__":
    main()
