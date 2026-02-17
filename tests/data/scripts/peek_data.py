"""Utilities for inspecting and visualizing hftbacktest NPZ data.

This module provides small helper functions to peek into raw market-data
and statistics NPZ files and to plot simple price series for exploratory
analysis and debugging.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
from __future__ import annotations

import os
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from hftbacktest import (
    BUY_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
)

# ---------------------------------------------------------------------
# Event filter configuration
# ---------------------------------------------------------------------
# Mapping from human-friendly string names to bit masks that can be used
# to filter rows by their event flags (column "ev").
#
# Convention:
#   - "trade"         : any trade event (buy or sell aggressor)
#   - "trade_buy"     : trade with BUY side flag set (buy aggressor)
#   - "trade_sell"    : trade with SELL side flag set (sell aggressor)
#   - "depth"         : any depth (order book) event
#   - "depth_bid"     : depth event on bid side
#   - "depth_ask"     : depth event on ask side
#   - "snapshot"      : any snapshot event
#   - "snapshot_bid"  : snapshot on bid side
#   - "snapshot_ask"  : snapshot on ask side
EVENT_FILTERS: Mapping[str, int] = {
    "trade": TRADE_EVENT,
    "trade_buy": TRADE_EVENT | BUY_EVENT,
    "trade_sell": TRADE_EVENT | SELL_EVENT,
    "depth": DEPTH_EVENT,
    "depth_bid": DEPTH_EVENT | BUY_EVENT,
    "depth_ask": DEPTH_EVENT | SELL_EVENT,
    "snapshot": DEPTH_SNAPSHOT_EVENT,
    "snapshot_bid": DEPTH_SNAPSHOT_EVENT | BUY_EVENT,
    "snapshot_ask": DEPTH_SNAPSHOT_EVENT | SELL_EVENT,
}


def _filter_by_event(data, filter_name):
    """
    Accepts either a single filter name (string) or multiple filter names (list/tuple of strings).
    Returns a boolean mask selecting all rows that match ANY of the filters.
    """
    if isinstance(filter_name, str):
        filter_names = [filter_name]
    else:
        filter_names = list(filter_name)

    masks = []
    for name in filter_names:
        if name not in EVENT_FILTERS:
            available = ", ".join(sorted(EVENT_FILTERS.keys()))
            raise ValueError(f"Unknown filter '{name}'. Available: {available}")
        bits = EVENT_FILTERS[name]
        masks.append((data["ev"] & bits) == bits)

    # OR-combine all masks → row is included if it matches ANY filter
    return np.logical_or.reduce(masks)


def peek_data(
    path: str = "data/btcusdt_20240809.npz",
    key: str = "data",
    n: int = 15,
    ) -> None:
    """
    Print basic information and the first `n` rows of an NPZ dataset.

    Parameters
    ----------
    path : str, optional
        Path to the NPZ file, by default "data/btcusdt_20240809.npz".
    key : str, optional
        Array key inside the NPZ file, by default "data".
    n : int, optional
        Number of rows to print from the start of the array, by default 15.
    """
    npz = np.load(path)
    if key not in npz.files:
        raise KeyError(
            f"Key '{key}' not found in NPZ file. "
            f"Available keys: {list(npz.files)}"
        )

    data = npz[key]
    print("file:", path)
    print("dtype:", data.dtype)
    print("shape:", data.shape)
    print(f"first {n} rows:")
    print(data[:n])

    # --- Time range inspection (timestamps) ---
    if "exch_ts" in data.dtype.names:
        ts = data["exch_ts"]

        start_ns = ts[0]
        end_ns = ts[-1]
        duration_sec = (end_ns - start_ns) / 1e9

        print("\nTime range (exch_ts):")
        print(f"  start ns : {start_ns}")
        print(f"  end   ns : {end_ns}")

        if duration_sec < 120:
            print(f"  duration : {duration_sec:.2f} seconds")
        elif duration_sec < 7200:
            print(f"  duration : {duration_sec/60:.2f} minutes")
        else:
            print(f"  duration : {duration_sec/3600:.2f} hours")

def inspect_stats(path: str = "stats_example.npz", n: int = 10) -> None:
    """
    Inspect a stats NPZ file produced by `Recorder.to_npz()`.

    This is intended for quick manual inspection of an example stats file.

    Parameters
    ----------
    path : str, optional
        Path to the stats NPZ file, by default "stats_example.npz".
    """
    npz = np.load(path)
    print("Keys in stats NPZ:", npz.files)

    # By convention, backtest stats are stored under key "0"
    if "0" not in npz.files:
        raise KeyError(
            f"Key '0' not found in stats NPZ. "
            f"Available keys: {list(npz.files)}"
        )

    stats = npz["0"]
    print("Stats dtype:", stats.dtype)
    print("Stats shape:", stats.shape)
    if stats.shape[0] > 0:
        print("First rows:")
        print(stats[:n])


def plot_price_series(
    path: str = "data/btcusdt_20240809.npz",
    key: str = "data",
    ts_field: str = "exch_ts",
    px_field: str = "px",
    event_filter: str | None = None,
    output_path: str | None = None,
    ) -> None:
    """
    Plot a price series from a raw market-data NPZ file.

    The function optionally filters rows by logical event type (e.g. only trades),
    converts timestamps from nanoseconds to seconds (relative to the first event),
    and saves the resulting plot as a PNG.

    Expected dtype of the `key` array (typical hftbacktest format):
        [
            ('ev', 'u8'),       # event flags
            ('exch_ts', 'i8'),  # venue timestamp (ns)
            ('local_ts', 'i8'), # local receive timestamp (ns)
            ('px', 'f8'),       # price
            ('qty', 'f8'),      # quantity
            ('order_id', 'u8'), # order ID (for L3 feeds)
            ('ival', 'i8'),     # auxiliary integer
            ('fval', 'f8'),     # auxiliary float
        ]

    Parameters
    ----------
    path : str, optional
        Path to the NPZ file, by default "data/btcusdt_20240809.npz".
    key : str, optional
        Array key inside the NPZ file, by default "data".
    ts_field : str, optional
        Name of the timestamp field, by default "exch_ts".
    px_field : str, optional
        Name of the price field, by default "px".
    event_filter : str or None, optional
        Logical event filter name (see EVENT_FILTERS). If None, all rows are used.
    output_path : str or None, optional
        Output PNG filename. If None, a name is derived from `path` and `event_filter`.
    """
    npz = np.load(path)
    if key not in npz.files:
        raise KeyError(
            f"Key '{key}' not found in NPZ file. "
            f"Available keys: {list(npz.files)}"
        )

    data = npz[key]

    if ts_field not in data.dtype.names:
        raise KeyError(
            f"Timestamp field '{ts_field}' not found. "
            f"Available fields: {data.dtype.names}"
        )
    if px_field not in data.dtype.names:
        raise KeyError(
            f"Price field '{px_field}' not found. "
            f"Available fields: {data.dtype.names}"
        )

    # Optionally filter by event type using human-friendly name
    if event_filter is not None:
        mask = _filter_by_event(data, event_filter)
        data = data[mask]
        if data.size == 0:
            raise ValueError(
                f"No rows match event filter '{event_filter}'. "
                "Try a different filter or inspect EVENT_FILTERS."
            )

    if data.size == 0:
        raise ValueError("No rows in data array, nothing to plot.")

    ts = data[ts_field]
    px = data[px_field]

    # Convert nanoseconds to seconds and make it relative to the first timestamp
    t_sec = (ts - ts[0]) / 1e9

    # Auto-generate output filename if none is provided
    if output_path is None:
        base = os.path.splitext(os.path.basename(path))[0]
        suffix = f"_{event_filter}" if event_filter else ""
        output_path = f"{base}_price{suffix}.png"

    # Build a human-readable title
    filter_label = event_filter if event_filter is not None else "all events"
    title = f"Price series ({px_field}) from {os.path.basename(path)}\nfilter: {filter_label}"

    # Create and save the plot
    plt.figure(figsize=(14, 6))
    plt.plot(t_sec, px)
    plt.title(title)
    plt.xlabel("Time (s, relative to first event)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved price plot to: {output_path}")


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Example usages
    # -----------------------------------------------------------------

    out_dir_parts = "tests/data/parts"
    out_dir_res = "tests/data/results"

    parts_file_0 = f"{out_dir_parts}/part-000.npz"
    parts_file_1 = f"{out_dir_parts}/part-001.npz"
    stats_file = f"{out_dir_res}/stats.npz"

    # Quick peek at the raw market data file
    peek_data(parts_file_0, n=30)

    # Plot full price series (all events)
    plot_price_series(parts_file_0, event_filter=["trade"], output_path=f"{out_dir_res}/part-000.png")

    # Plot only buy-side trade prices (aggressive buys)
    plot_price_series(parts_file_1, event_filter="trade_buy", output_path=f"{out_dir_res}/part-001.png")

    # Inspect a stats file produced by Recorder.to_npz()
    inspect_stats(stats_file, n=20)
