"""
Synthetic hftbacktest market data generator.

This script generates deterministic synthetic Level-2 depth and trade data
compatible with hftbacktest v2.x. It is intended for testing and benchmarking
purposes and therefore prioritizes explicitness and configurability over
minimal function signatures.
"""

# pylint: disable=line-too-long
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-statements
# pylint: disable=redefined-outer-name,no-else-return
# pylint: disable=invalid-name
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
        ("ev", "u8"),       # uint64 event flags
        ("exch_ts", "i8"),  # venue timestamp (ns)
        ("local_ts", "i8"), # local receive timestamp (ns)
        ("px", "f8"),       # price in tick units
        ("qty", "f8"),      # quantity in lot units
        ("order_id", "u8"), # uint64 (used only for L3 feeds)
        ("ival", "i8"),     # auxiliary integer
        ("fval", "f8"),     # auxiliary float
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

    profile:
        - "flat": same qty on all levels
        - "linear": grows linearly with depth
        - "exp": decays exponentially with depth
    """
    if profile == "flat":
        return base_qty
    if profile == "linear":
        return base_qty * (level + 1)
    if profile == "exp":
        return base_qty * np.exp(-level / 4.0)
    return base_qty


def make_synthetic_day(
    n_steps: int = 1000,
    interval_ns: int = 100_000_000,
    tick_size: float = 0.1,
    lot_size: float = 0.001,
    base_price: float = 60_000.0,
    *,
    n_levels: int = 10,
    base_exch_ts_ns: int = 1_723_161_256_000_000_000,
    feed_latency_ns: int = 1_000_000,
    regime_switch_prob: float = 0.01,
    trend_strength_ticks: float = 0.1,
    mean_reversion_strength: float = 0.02,
    sigma_low_ticks: float = 0.2,
    sigma_high_ticks: float = 0.7,
    volatility_switch_prob: float = 0.03,
    depth_profile: str = "linear",
    trade_prob: float = 0.7,
    max_trades_per_step: int = 2,
) -> np.ndarray:
    """
    Synthetic trading day generator producing L2 depth and trades.
    All prices and quantities are stored in integer tick / lot units.
    """
    rng = np.random.default_rng(42)

    max_events_per_step = 2 * n_levels + max_trades_per_step
    data = np.zeros(n_steps * max_events_per_step, dtype=DTYPE)
    ptr = 0

    mid = base_price
    regimes = np.array(["mean_revert", "trend_up", "trend_down"])
    regime = "mean_revert"
    vol_state = "low"

    for step in range(n_steps):
        if rng.random() < regime_switch_prob:
            regime = rng.choice(regimes)

        if rng.random() < volatility_switch_prob:
            vol_state = "high" if vol_state == "low" else "low"

        sigma_ticks = sigma_low_ticks if vol_state == "low" else sigma_high_ticks

        if regime == "mean_revert":
            diff_ticks = (mid - base_price) / tick_size
            drift_ticks = -mean_reversion_strength * diff_ticks
        elif regime == "trend_up":
            drift_ticks = trend_strength_ticks
        else:
            drift_ticks = -trend_strength_ticks

        mid += (drift_ticks + rng.normal(scale=sigma_ticks)) * tick_size
        mid = max(tick_size, mid)

        exch_ts = base_exch_ts_ns + step * interval_ns
        local_ts = exch_ts + feed_latency_ns

        mid_rounded = np.round(mid / tick_size) * tick_size
        best_bid = mid_rounded - tick_size
        best_ask = best_bid + tick_size

        for level in range(n_levels):
            bid_px = best_bid - level * tick_size
            bid_qty = _depth_profile_qty(level, lot_size, depth_profile)

            data["ev"][ptr] = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | BUY_EVENT
            data["exch_ts"][ptr] = exch_ts
            data["local_ts"][ptr] = local_ts
            data["px"][ptr] = bid_px
            data["qty"][ptr] = bid_qty
            data["order_id"][ptr] = 0
            data["ival"][ptr] = 0
            data["fval"][ptr] = 0.0
            ptr += 1

            ask_px = best_ask + level * tick_size
            ask_qty = _depth_profile_qty(level, lot_size, depth_profile)

            data["ev"][ptr] = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | SELL_EVENT
            data["exch_ts"][ptr] = exch_ts
            data["local_ts"][ptr] = local_ts
            data["px"][ptr] = ask_px
            data["qty"][ptr] = ask_qty
            data["order_id"][ptr] = 0
            data["ival"][ptr] = 0
            data["fval"][ptr] = 0.0
            ptr += 1

        n_trades = 0
        while n_trades < max_trades_per_step and rng.random() < trade_prob:
            side_is_buy = rng.random() < 0.5
            flag_side = BUY_EVENT if side_is_buy else SELL_EVENT
            ev_flag = EXCH_EVENT | LOCAL_EVENT | TRADE_EVENT | flag_side

            trade_px = best_ask if side_is_buy else best_bid
            trade_qty = rng.integers(1, 11) * lot_size

            data["ev"][ptr] = ev_flag
            data["exch_ts"][ptr] = exch_ts
            data["local_ts"][ptr] = local_ts
            data["px"][ptr] = trade_px
            data["qty"][ptr] = trade_qty
            data["order_id"][ptr] = 0
            data["ival"][ptr] = 0
            data["fval"][ptr] = 0.0

            ptr += 1
            n_trades += 1

    data = data[:ptr]
    sort_idx = np.lexsort((data["local_ts"], data["exch_ts"]))
    return data[sort_idx]


def save_npz(filename: str, data: np.ndarray, compress: bool = True) -> None:
    """Save data in hftbacktest-compatible NPZ format."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if compress:
        np.savez_compressed(filename, data=data)
    else:
        np.savez(filename, data=data)
    print(f"Saved {filename} with shape {data.shape}")


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Example usages
    # -----------------------------------------------------------------

    out_dir = "tests/data/parts"
    os.makedirs(out_dir, exist_ok=True)

    tick_size = 0.1
    lot_size = 0.01

    snapshot = make_synthetic_snapshot(
        exch_ts=1_723_161_256_000_000_000,
        local_ts=1_723_161_256_001_000_000,
        mid_price=60000.0,
        tick_size=tick_size,
        lot_size=lot_size,
        n_levels=10,
    )
    save_npz(os.path.join(out_dir, "part-000-eod.npz"), snapshot)

    day_data = make_synthetic_day(
        n_steps=1000,
        interval_ns=100_000_000,
        tick_size=tick_size,
        lot_size=lot_size,
        base_price=60000.0,
        base_exch_ts_ns=1_723_161_256_000_000_000,
    )
    save_npz(os.path.join(out_dir, "part-000.npz"), day_data)
