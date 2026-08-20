#!/usr/bin/env python3
"""Redis-only Dash dashboard for TC property activation (P1-P15)."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dash
import redis
from dash import Input, Output, dcc, html


PROPERTY_SIGNALS: dict[str, list[str]] = {
    "P1 loop liveness": ["v_capture_timeout", "v_detection_timeout"],
    "P2 entropy window analysis": [
        "v_window_overflow",
        "v_entropy_range",
        "v_avg_range",
        "v_avg_mismatch",
    ],
    "P3 sensor brightness": ["w_brightness_range"],
    "P4 anomaly soundness": ["v_anomaly_unsound"],
    "P5 missed anomaly": ["v_missed_anomaly"],
    "P6 protocol conformance": [
        "v_overlapping_anomaly",
        "v_unsolicited_plan",
        "v_unsolicited_verdict",
        "v_unlegitimated_execution",
    ],
    "P7 per-step deadlines": [
        "v_plan_timeout",
        "v_legit_timeout",
        "v_execute_timeout",
    ],
    "P8 whole-episode bound": ["v_episode_timeout"],
    "P9 candidate validity": ["v_candidate_unknown", "v_candidate_self_swap"],
    "P10 legitimation consistency": [
        "v_unjustified_accept",
        "v_unjustified_reject",
        "v_verdict_flag_mismatch",
    ],
    "P11 replan budget": ["v_replan_budget", "v_counter_leak"],
    "P12 execute consistency": ["v_execute_mismatch"],
    "P13 post-swap reset": ["v_post_reset"],
    "P14 adaptation effectiveness": ["v_ineffective_adaptation"],
    "P15 swap rate and ping-pong": ["v_thrashing", "v_ping_pong"],
}


@dataclass
class DashboardState:
    signal_values: dict[str, bool] = field(default_factory=dict)
    signal_seen_at: dict[str, float] = field(default_factory=dict)
    property_history: dict[str, list[tuple[float, bool | None]]] = field(
        default_factory=dict
    )
    total_messages: int = 0
    last_message_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dash dashboard for TC property booleans from Redis output topics"
    )
    parser.add_argument(
        "--dsrv-file",
        type=Path,
        default=Path("examples/screw_detection_maple_full.dsrv"),
        help="DSRV file used to validate required output variables",
    )
    parser.add_argument(
        "--redis-host", default="127.0.0.1", help="Redis host used by TC"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis TCP port used by TC"
    )
    parser.add_argument("--redis-db", type=int, default=0, help="Redis DB index")
    parser.add_argument("--host", default="127.0.0.1", help="Dash bind address")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port")
    parser.add_argument(
        "--refresh-ms", type=int, default=500, help="UI refresh interval in milliseconds"
    )
    parser.add_argument(
        "--history-window-sec",
        type=float,
        default=10.0,
        help="Seconds of status history shown in each moving timeline bar",
    )
    parser.add_argument(
        "--history-bins",
        type=int,
        default=60,
        help="Number of time slices in each timeline bar",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print matched Redis channel/value updates to stdout",
    )
    return parser.parse_args()


def parse_dsrv_outputs(dsrv_file: Path) -> set[str]:
    outputs: set[str] = set()
    out_decl = re.compile(r"^\s*out\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")
    for line in dsrv_file.read_text(encoding="utf-8").splitlines():
        match = out_decl.match(line)
        if match:
            outputs.add(match.group(1))
    return outputs


def decode_bool_payload(raw_payload: Any) -> bool | None:
    payload = (
        raw_payload.decode("utf-8", errors="replace")
        if isinstance(raw_payload, bytes)
        else str(raw_payload)
    )
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        lowered = payload.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return None

    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("value"), bool):
        # Defensive support for wrapped payloads.
        return parsed["value"]
    return None


def resolve_signal(channel: str, expected_signals: set[str]) -> str | None:
    if channel in expected_signals:
        return channel

    # Support routed/prefixed channels, e.g. "nodeA/v_capture_timeout".
    last_segment = channel.split("/")[-1].split(":")[-1].split(".")[-1]
    if last_segment in expected_signals:
        return last_segment

    # Final fallback: exact suffix match against known signals.
    for signal in expected_signals:
        if channel.endswith(signal):
            return signal

    return None


def redis_listener(
    redis_client: redis.Redis,
    expected_signals: set[str],
    state: DashboardState,
    verbose: bool,
) -> None:
    pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
    pubsub.psubscribe("*")

    for message in pubsub.listen():
        if message.get("type") not in {"message", "pmessage"}:
            continue

        topic_raw = message.get("channel", b"")
        topic = topic_raw.decode("utf-8") if isinstance(topic_raw, bytes) else str(topic_raw)
        signal = resolve_signal(topic, expected_signals)
        if signal is None:
            continue

        value = decode_bool_payload(message.get("data"))
        if value is None:
            continue

        now = time.time()
        with state.lock:
            state.signal_values[signal] = value
            state.signal_seen_at[signal] = now
            state.total_messages += 1
            state.last_message_at = now

        if verbose:
            print(f"[redis] channel={topic} signal={signal} value={value}")


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 1:
        return "<1s ago"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    minutes = int(seconds // 60)
    return f"{minutes}m {int(seconds % 60)}s ago"


def property_state(
    signals: list[str], values: dict[str, bool], seen_at: dict[str, float]
) -> tuple[str, bool | None, int]:
    seen_count = sum(1 for signal in signals if signal in seen_at)
    if seen_count == 0:
        return "WAITING", None, 0

    active = any(values.get(signal, False) for signal in signals if signal in seen_at)
    suffix = "" if seen_count == len(signals) else " (partial)"
    return ("TRUE" if active else "FALSE") + suffix, active, seen_count


def update_property_history(
    history: dict[str, list[tuple[float, bool | None]]],
    property_name: str,
    state_value: bool | None,
    now: float,
    history_window_sec: float,
) -> None:
    entries = history.setdefault(property_name, [])

    if not entries or entries[-1][1] != state_value:
        entries.append((now, state_value))

    cutoff = now - history_window_sec - 2.0
    while len(entries) > 1 and entries[1][0] < cutoff:
        entries.pop(0)


def color_for_state(state_value: bool | None) -> str:
    if state_value is None:
        return "#808080"
    return "#b00020" if state_value else "#0b7a32"


def history_background(
    entries: list[tuple[float, bool | None]], now: float, window_sec: float, bins: int
) -> str:
    if bins < 2:
        bins = 2

    if not entries:
        return "#808080"

    start_time = now - window_sec
    colors: list[str] = []
    idx = 0
    current_state: bool | None = None

    for bin_idx in range(bins):
        sample_t = start_time + ((bin_idx + 0.5) / bins) * window_sec
        while idx < len(entries) and entries[idx][0] <= sample_t:
            current_state = entries[idx][1]
            idx += 1
        colors.append(color_for_state(current_state))

    segments: list[str] = []
    run_start = 0
    for i in range(1, bins + 1):
        if i == bins or colors[i] != colors[run_start]:
            left = (run_start / bins) * 100.0
            right = (i / bins) * 100.0
            segments.append(f"{colors[run_start]} {left:.2f}% {right:.2f}%")
            run_start = i

    return "linear-gradient(to right, " + ", ".join(segments) + ")"


def property_card(
    property_name: str,
    signals: list[str],
    values: dict[str, bool],
    seen_at: dict[str, float],
    history: list[tuple[float, bool | None]],
    now: float,
    history_window_sec: float,
    history_bins: int,
) -> html.Div:
    status, _, _ = property_state(signals, values, seen_at)
    timeline_bg = history_background(
        entries=history,
        now=now,
        window_sec=history_window_sec,
        bins=history_bins,
    )

    last_seen = max((seen_at.get(signal, 0.0) for signal in signals), default=0.0)
    age = format_age(now - last_seen if last_seen > 0 else None)

    signal_text = []
    for signal in signals:
        if signal not in seen_at:
            signal_text.append(f"{signal}=?")
        else:
            signal_text.append(f"{signal}={values.get(signal, False)}")

    return html.Div(
        style={
            "border": "1px solid #d7d7d7",
            "borderRadius": "8px",
            "padding": "6px 8px",
            "marginBottom": "6px",
            "backgroundColor": "#ffffff",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                },
                children=[
                    html.Div(
                        property_name,
                        title=", ".join(signal_text),
                        style={
                            "minWidth": "240px",
                            "maxWidth": "240px",
                            "fontWeight": "700",
                            "fontSize": "13px",
                            "whiteSpace": "nowrap",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                    ),
                    html.Div(
                        style={
                            "height": "14px",
                            "flex": "1",
                            "borderRadius": "6px",
                            "backgroundColor": "#ececec",
                            "overflow": "hidden",
                            "background": timeline_bg,
                            "border": "1px solid #d0d0d0",
                        },
                    ),
                    html.Div(
                        status,
                        style={
                            "minWidth": "96px",
                            "textAlign": "right",
                            "fontSize": "12px",
                            "fontWeight": "700",
                        },
                    ),
                    html.Div(
                        age,
                        style={
                            "minWidth": "74px",
                            "textAlign": "right",
                            "fontSize": "11px",
                            "color": "#666666",
                        },
                    ),
                    html.Div(
                        f"{int(history_window_sec)}s",
                        style={
                            "minWidth": "30px",
                            "textAlign": "right",
                            "fontSize": "11px",
                            "color": "#777777",
                        },
                    ),
                ],
            ),
        ],
    )


def main() -> None:
    args = parse_args()

    declared_outputs = parse_dsrv_outputs(args.dsrv_file)
    missing = {
        name: [signal for signal in signals if signal not in declared_outputs]
        for name, signals in PROPERTY_SIGNALS.items()
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        lines = ["DSRV file is missing expected output signals:"]
        for prop, values in missing.items():
            lines.append(f"- {prop}: {', '.join(values)}")
        raise SystemExit("\n".join(lines))

    all_signals = sorted({signal for signals in PROPERTY_SIGNALS.values() for signal in signals})
    expected_signals = set(all_signals)

    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=False,
    )
    state = DashboardState()

    thread = threading.Thread(
        target=redis_listener,
        args=(redis_client, expected_signals, state, args.verbose),
        daemon=True,
    )
    thread.start()

    app = dash.Dash(__name__)
    app.title = "TC Property Dashboard"

    app.layout = html.Div(
        style={
            "maxWidth": "1320px",
            "margin": "0 auto",
            "padding": "12px 14px",
            "background": "linear-gradient(180deg, #f6f7f8 0%, #ffffff 45%)",
            "minHeight": "100vh",
            "fontFamily": "'Trebuchet MS', 'Segoe UI', sans-serif",
        },
        children=[
            html.H2("Trustworthiness Checker Property Dashboard", style={"margin": "2px 0 6px 0"}),
            html.P(
                "Rolling timeline per property: green=False, red=True, gray=waiting. New status appears on the right and shifts left over 10 seconds.",
                style={"color": "#4e4e4e", "fontSize": "12px", "margin": "0 0 8px 0"},
            ),
            html.Div(id="summary", style={"marginBottom": "8px", "fontWeight": "600", "fontSize": "13px"}),
            html.Div(id="property-cards"),
            dcc.Interval(id="refresh", interval=args.refresh_ms, n_intervals=0),
        ],
    )

    @app.callback(
        Output("summary", "children"),
        Output("property-cards", "children"),
        Input("refresh", "n_intervals"),
    )
    def refresh_dashboard(_: int):
        now = time.time()
        with state.lock:
            values = dict(state.signal_values)
            seen_at = dict(state.signal_seen_at)
            total_messages = state.total_messages
            last_message_at = state.last_message_at

            for property_name, signals in PROPERTY_SIGNALS.items():
                _, state_value, _ = property_state(signals, values, seen_at)
                update_property_history(
                    history=state.property_history,
                    property_name=property_name,
                    state_value=state_value,
                    now=now,
                    history_window_sec=args.history_window_sec,
                )

            history_snapshot = {
                name: list(entries) for name, entries in state.property_history.items()
            }

        ready_count = 0
        active_count = 0
        cards = []
        for property_name, signals in PROPERTY_SIGNALS.items():
            _, state_value, seen_count = property_state(signals, values, seen_at)
            if seen_count > 0:
                ready_count += 1
                if state_value is True:
                    active_count += 1

            cards.append(
                property_card(
                    property_name=property_name,
                    signals=signals,
                    values=values,
                    seen_at=seen_at,
                    history=history_snapshot.get(property_name, []),
                    now=now,
                    history_window_sec=args.history_window_sec,
                    history_bins=args.history_bins,
                )
            )

        age = format_age(now - last_message_at if last_message_at else None)
        summary = (
            f"Publishing properties: {ready_count}/{len(PROPERTY_SIGNALS)} | "
            f"Active violations: {active_count}/{len(PROPERTY_SIGNALS)} | "
            f"Redis messages: {total_messages} | Last message: {age}"
        )
        return summary, cards

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
