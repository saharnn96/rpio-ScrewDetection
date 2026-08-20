#!/usr/bin/env python3
"""Fake TC output publisher, to exercise tc_dashboard.py without a real checker.

Publishes the same property booleans a Trustworthiness Checker would emit --
one redis pub/sub channel per `out` stream, payload `true` / `false` -- so the
dashboard's bars move without the MAPLE-K loop or the RV monitor running.

This is a test fixture, not part of the specification.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import redis

from tc_dashboard import PROPERTY_SIGNALS

MODES = ("wave", "random", "all-ok", "all-bad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish fake TC property booleans to redis for tc_dashboard.py"
    )
    parser.add_argument(
        "--redis-host", default="127.0.0.1", help="Redis host (never 'localhost' on Windows)"
    )
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis TCP port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis DB index")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="wave",
        help="wave: one violation walks across the properties; "
        "random: each signal flips with --violation-prob; "
        "all-ok / all-bad: hold every signal false / true",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Seconds between publish rounds"
    )
    parser.add_argument(
        "--violation-prob",
        type=float,
        default=0.05,
        help="Per-signal probability of being true in --mode random",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Stop after N rounds (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--dsrv-file",
        type=Path,
        default=None,
        help="Optional: cross-check the signal list against a .dsrv file's out streams",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not print a per-round summary"
    )
    return parser.parse_args()


def signals_in_order() -> list[str]:
    """All property signals, kept in P1..P15 order so --mode wave sweeps top-down."""
    ordered: list[str] = []
    for property_signals in PROPERTY_SIGNALS.values():
        ordered.extend(property_signals)
    return ordered


def check_against_dsrv(dsrv_file: Path, signals: list[str]) -> None:
    out_decl = re.compile(r"^\s*out\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")
    declared = {
        match.group(1)
        for line in dsrv_file.read_text(encoding="utf-8").splitlines()
        if (match := out_decl.match(line))
    }
    missing = [signal for signal in signals if signal not in declared]
    if missing:
        raise SystemExit(
            f"{dsrv_file} does not declare: {', '.join(missing)}"
        )
    print(f"[check] all {len(signals)} signals are declared in {dsrv_file}")


def values_for_round(mode: str, signals: list[str], round_idx: int, prob: float) -> dict[str, bool]:
    if mode == "all-ok":
        return {signal: False for signal in signals}
    if mode == "all-bad":
        return {signal: True for signal in signals}
    if mode == "random":
        return {signal: random.random() < prob for signal in signals}
    # wave: exactly one signal true, advancing one step per round
    active = signals[round_idx % len(signals)]
    return {signal: signal == active for signal in signals}


def main() -> None:
    args = parse_args()
    signals = signals_in_order()

    if args.dsrv_file is not None:
        check_against_dsrv(args.dsrv_file, signals)

    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db)
    client.ping()
    print(
        f"[publish] {len(signals)} signals -> {args.redis_host}:{args.redis_port}/{args.redis_db} "
        f"mode={args.mode} interval={args.interval}s (Ctrl+C to stop)"
    )

    round_idx = 0
    try:
        while args.rounds == 0 or round_idx < args.rounds:
            values = values_for_round(args.mode, signals, round_idx, args.violation_prob)
            for signal, value in values.items():
                client.publish(signal, json.dumps(value))

            if not args.quiet:
                active = [signal for signal, value in values.items() if value]
                summary = ", ".join(active) if active else "none"
                print(f"[round {round_idx:>4}] true: {summary}")

            round_idx += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[publish] stopped")


if __name__ == "__main__":
    main()
