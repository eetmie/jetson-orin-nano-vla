"""Summarize ONNX Runtime trace events as measured execution placement."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def summarize_profile(path: str | Path) -> dict:
    events = json.loads(Path(path).read_text())
    if not isinstance(events, list):
        raise ValueError(f"ORT profile must be a JSON event list: {path}")

    by_provider: dict[str, dict] = {}
    unassigned = 0
    for event in events:
        if not isinstance(event, dict) or str(event.get("cat", "")).lower() != "node":
            continue
        args = event.get("args") or {}
        provider = args.get("provider") or args.get("execution_provider")
        if not provider:
            unassigned += 1
            continue
        item = by_provider.setdefault(str(provider), {
            "event_count": 0,
            "duration_us": 0.0,
            "_nodes": set(),
            "_ops": Counter(),
        })
        item["event_count"] += 1
        item["duration_us"] += float(event.get("dur") or 0.0)
        item["_nodes"].add(str(event.get("name") or "<unnamed>"))
        item["_ops"][str(args.get("op_name") or "<unknown>")] += 1

    providers = {}
    for provider, item in sorted(by_provider.items()):
        providers[provider] = {
            "event_count": item["event_count"],
            "unique_node_count": len(item["_nodes"]),
            "duration_ms": round(item["duration_us"] / 1000.0, 6),
            "ops": dict(sorted(item["_ops"].items())),
        }
    return {
        "event_count": len(events),
        "node_event_count": sum(v["event_count"] for v in providers.values()),
        "unassigned_node_event_count": unassigned,
        "providers": providers,
    }


def placement_verdict(per_graph: dict[str, dict], expected_graphs: list[str]) -> dict:
    missing_profiles = sorted(set(expected_graphs) - set(per_graph))
    missing_trt = []
    fallbacks = {}
    for graph in expected_graphs:
        providers = (per_graph.get(graph) or {}).get("providers") or {}
        if not providers.get("TensorrtExecutionProvider", {}).get("event_count"):
            missing_trt.append(graph)
        graph_fallbacks = {
            provider: stats
            for provider, stats in providers.items()
            if provider != "TensorrtExecutionProvider" and stats.get("event_count")
        }
        if graph_fallbacks:
            fallbacks[graph] = graph_fallbacks
    return {
        "status": "pass" if not missing_profiles and not missing_trt else "fail",
        "expected_graphs": len(expected_graphs),
        "profiled_graphs": len(per_graph),
        "graphs_with_trt_nodes": len(expected_graphs) - len(missing_trt),
        "missing_profiles": missing_profiles,
        "graphs_without_trt_nodes": missing_trt,
        "fallbacks": fallbacks,
    }
