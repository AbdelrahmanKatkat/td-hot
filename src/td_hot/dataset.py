"""Train/val splitting for fAIr chips (numpy only; safe to import at serving time)."""

import re

import numpy as np

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


def spatial_split(
    chip_names: list[str], val_ratio: float, seed: int, *, block_size: int = 4
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; whole `(x // K, y // K)` blocks go to one side.

    Neighbouring chips share trees cut at chip borders and similar canopy, so a random
    chip split leaks. Chips not named `OAM-x-y-z.ext` are shuffled and only fill leftover
    validation slots. Returns sorted `(train, val)` name lists.
    """
    matched: dict[tuple[int, int], list[str]] = {}
    unmatched: list[str] = []
    for name in chip_names:
        m = OAM_TILE_RE.match(name)
        if m is None:
            unmatched.append(name)
            continue
        matched.setdefault((int(m.group(1)) // block_size, int(m.group(2)) // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(matched)
    rng.shuffle(blocks)

    n_total = sum(len(v) for v in matched.values()) + len(unmatched)
    n_val_target = max(1, int(n_total * val_ratio)) if n_total else 0

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        (val if len(val) < n_val_target else train).extend(matched[block])

    if unmatched:
        leftover = sorted(unmatched)
        rng.shuffle(leftover)
        remaining = max(0, n_val_target - len(val))
        val.extend(leftover[:remaining])
        train.extend(leftover[remaining:])
    return sorted(train), sorted(val)
