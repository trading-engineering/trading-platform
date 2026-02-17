"""
Synthetic hftbacktest market data generator.

This script generates deterministic synthetic Level-2 depth snapshots,
incremental depth updates, and trades in a single unified event stream
compatible with hftbacktest v2.x.
"""

from __future__ import annotations

import os

import numpy as np
from hftbacktest import (
    BUY_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
)

# ---------------------------------------------------------------------------
# dtype used by hftbacktest v2.x
# px / qty MUST be integer tick / lot units
# ---------------------------------------------------------------------------

DTYPE = np.dtype(
    [
        ("ev", "u8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
        ("order_id", "u8"),
        ("ival", "i8"),
        ("fval", "f8"),
    ],
    align=True,
)


def make_synthetic_snapshot(
    exch_ts: int,
    local_ts: int,
    mid_price: float = 60000.0,
    tick_size: float = 0.1,
    lot_size: float = 0.001,
    n_levels: int = 5,
) -> np.ndarray:
    """
    Build a simple Level-2 snapshot: n bid levels and n ask levels around a mid price.
    Timestamps = exch_ts / local_ts.
    """
    n_rows = 2 * n_levels
    data = np.zeros(n_rows, dtype=DTYPE)

    bid_flag = EXCH_EVENT | DEPTH_SNAPSHOT_EVENT | BUY_EVENT
    ask_flag = EXCH_EVENT | DEPTH_SNAPSHOT_EVENT | SELL_EVENT

    # Bid levels
    for i in range(n_levels):
        idx = i
        price = mid_price - (i + 1) * tick_size
        qty = (i + 1) * lot_size

        data["ev"][idx] = bid_flag
        data["exch_ts"][idx] = exch_ts
        data["local_ts"][idx] = local_ts
        data["px"][idx] = price
        data["qty"][idx] = qty
        data["order_id"][idx] = 0
        data["ival"][idx] = 0
        data["fval"][idx] = 0.0

    # Ask levels
    for i in range(n_levels):
        idx = n_levels + i
        price = mid_price + (i + 1) * tick_size
        qty = (i + 1) * lot_size

        data["ev"][idx] = ask_flag
        data["exch_ts"][idx] = exch_ts
        data["local_ts"][idx] = local_ts
        data["px"][idx] = price
        data["qty"][idx] = qty
        data["order_id"][idx] = 0
        data["ival"][idx] = 0
        data["fval"][idx] = 0.0

    return data


def _depth_profile_qty(
    level: int,
    base_qty: float,
    profile: str = "linear",
) -> float:
    """
    Compute quantity for a given depth level.
    """
    if profile == "flat":
        return base_qty
    if profile == "linear":
        return base_qty * (level + 1)
    if profile == "exp":
        return base_qty * np.exp(-level / 4.0)
    return base_qty


def make_synthetic_day(
    n_steps: int = 1_000,
    interval_ns: int = 100_000_000,
    tick_size: float = 0.1,
    lot_size: float = 0.001,
    base_price: float = 60_000.0,
    *,
    n_levels: int = 10,
    base_exch_ts_ns: int = 1_723_161_256_000_000_000,
    feed_latency_ns: int = 1_000_000,
    trade_prob: float = 0.7,
    max_trades_per_step: int = 2,
    depth_profile: str = "linear",
) -> np.ndarray:
    """
    Synthetic trading session producing depth updates and trades.
    """
    rng = np.random.default_rng(42)

    max_events_per_step = 2 * n_levels + max_trades_per_step
    buffer = np.zeros(n_steps * max_events_per_step, dtype=DTYPE)
    ptr = 0

    mid_price = base_price

    for step in range(n_steps):
        exch_ts = base_exch_ts_ns + step * interval_ns
        local_ts = exch_ts + feed_latency_ns

        mid_price += rng.normal(scale=0.2) * tick_size
        mid_price = max(tick_size, mid_price)

        mid_rounded = round(mid_price / tick_size) * tick_size
        best_bid = mid_rounded - tick_size
        best_ask = mid_rounded + tick_size

        for level in range(n_levels):
            bid_px = best_bid - level * tick_size
            ask_px = best_ask + level * tick_size
            qty = _depth_profile_qty(level, lot_size, depth_profile)

            buffer["ev"][ptr] = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | BUY_EVENT
            buffer["exch_ts"][ptr] = exch_ts
            buffer["local_ts"][ptr] = local_ts
            buffer["px"][ptr] = bid_px
            buffer["qty"][ptr] = qty
            ptr += 1

            buffer["ev"][ptr] = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | SELL_EVENT
            buffer["exch_ts"][ptr] = exch_ts
            buffer["local_ts"][ptr] = local_ts
            buffer["px"][ptr] = ask_px
            buffer["qty"][ptr] = qty
            ptr += 1

        trades = 0
        while trades < max_trades_per_step and rng.random() < trade_prob:
            is_buy = rng.random() < 0.5
            side_flag = BUY_EVENT if is_buy else SELL_EVENT

            buffer["ev"][ptr] = EXCH_EVENT | LOCAL_EVENT | TRADE_EVENT | side_flag
            buffer["exch_ts"][ptr] = exch_ts
            buffer["local_ts"][ptr] = local_ts
            buffer["px"][ptr] = best_ask if is_buy else best_bid
            buffer["qty"][ptr] = rng.integers(1, 10) * lot_size

            ptr += 1
            trades += 1

    data = buffer[:ptr]
    order = np.lexsort((data["local_ts"], data["exch_ts"]))
    return data[order]


def save_npz(filename: str, data: np.ndarray) -> None:
    """Save event stream in hftbacktest-compatible NPZ format."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, data=data)
    print(f"Saved {filename} with shape {data.shape}")


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Example usages
    # -----------------------------------------------------------------

    output_dir = "tests/data/parts"
    os.makedirs(output_dir, exist_ok=True)

    tick_size = 0.1
    lot_size = 0.01

    snapshot_data = make_synthetic_snapshot(
        exch_ts=1_723_161_256_000_000_000,
        local_ts=1_723_161_256_001_000_000,
        mid_price=60_000.0,
        tick_size=tick_size,
        lot_size=lot_size,
        n_levels=10,
    )

    day_data = make_synthetic_day(
        n_steps=100_000,
        interval_ns=100_000_000,
        tick_size=tick_size,
        lot_size=lot_size,
        base_price=60_000.0,
        base_exch_ts_ns=1_723_161_256_000_000_000,
    )

    unified_data = np.concatenate((snapshot_data, day_data))
    unified_data = unified_data[
        np.lexsort((unified_data["local_ts"], unified_data["exch_ts"]))
    ]

    save_npz(
        os.path.join(output_dir, "part-000.npz"),
        unified_data,
    )
