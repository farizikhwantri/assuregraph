import argparse
import json
import os
import re
import textwrap  
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colors as mcolors           
import colorsys     

import textwrap


def wrap_label(text: str, width: int = 18, max_lines: int = 3) -> str:
    """
    Wrap a label into multiple lines (by words).
    width = characters per line, max_lines = maximum number of lines to render.
    """
    lines = textwrap.wrap(str(text), width=width, break_long_words=False)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        # Add ellipsis if we truncated
        if lines:
            lines[-1] = lines[-1].rstrip(".") + " ..."
    return "\n".join(lines)

# --------------------- IO --------------------- #
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------- Helpers --------------------- #
TYPE_COLORS = {
    "Goal": "#1f77b4",
    "Strategy": "#ff7f0e",
    "Context": "#2ca02c",
    "Justification": "#9467bd",
    "Assumption": "#8c564b",
    "Solution": "#e377c2",
    # Fallbacks/unknown
    "Unknown": "#7f7f7f",
}

def boost_hex_color(hex_color: str, sat: float = 1.3, val: float = 1.1):
    """
    Make a color 'stronger' by increasing saturation and brightness in HSV space.
    Returns an RGB tuple usable by Matplotlib.
    """
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = max(0.0, min(1.0, s * sat))
    v = max(0.0, min(1.0, v * val))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2)

def wrap_label(text: str, width: int = 18, max_lines: int = 3) -> str:
    """
    Wrap a label into multiple lines (by words).
    width = characters per line, max_lines = maximum number of lines to render.
    """
    lines = textwrap.wrap(str(text), width=width, break_long_words=False)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        # Add ellipsis if we truncated
        if lines:
            lines[-1] = lines[-1].rstrip(".") + " ..."
    return "\n".join(lines)

def first_n_words(text: str, n: int = 5) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    n = max(1, min(n, 5))
    out = " ".join(words[:n])
    if len(words) > n:
        out += " ..."
    return out


def normalize_type(node_type: Optional[str]) -> str:
    if not node_type or not isinstance(node_type, str):
        return "Unknown"
    t = node_type.strip()
    # common typos/variants
    mapping = {
        "goal": "Goal",
        "strategy": "Strategy",
        "context": "Context",
        "justification": "Justification",
        "assumption": "Assumption",
        "solution": "Solution",
    }
    return mapping.get(t.lower(), t)


def record_valid(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    pc = record.get("parent_child", {})
    nd = record.get("nodes", {})
    return isinstance(pc, dict) and isinstance(nd, dict) and (len(pc) > 0 or len(nd) > 0)


def match_llm_docname(docname: str, tag_substr: str) -> bool:
    # Case-insensitive substring match for tags like "ACAS XU_2_0"
    return tag_substr.lower() in str(docname).lower()


def pick_human_record(records: List[Dict[str, Any]], human_docname: str) -> Optional[Dict[str, Any]]:
    for r in records:
        if r.get("model_name") == "human":
            if str(r.get("docname", "")).lower() == human_docname.lower() and record_valid(r):
                return r
    # fallback: first human record with content
    for r in records:
        if r.get("model_name") == "human" and record_valid(r):
            return r
    return None


def pick_llm_record(records: List[Dict[str, Any]], llm_tag: str) -> Optional[Dict[str, Any]]:
    # Prefer exact tag match in docname first
    for r in records:
        if r.get("model_name", "").lower() != "human":
            if match_llm_docname(str(r.get("docname", "")), llm_tag) and record_valid(r):
                return r
    # fallback: any non-human record with content
    for r in records:
        if r.get("model_name", "").lower() != "human" and record_valid(r):
            return r
    return None


# --------------------- Graph build --------------------- #
def build_graph(record: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[str, Dict[str, Any]]]:
    """
    Build a DiGraph from parent_child, attach node attributes:
      - type (normalized)
      - description
      - label (ID + short description)
    """
    pc: Dict[str, List[str]] = record.get("parent_child", {}) or {}
    nodes_meta: Dict[str, Any] = record.get("nodes", {}) or {}

    # Collect all node ids from nodes and parent_child
    ids = set(nodes_meta.keys())
    for p, childs in pc.items():
        ids.add(str(p))
        for c in (childs or []):
            ids.add(str(c))

    G = nx.DiGraph()
    # Add nodes with attributes
    for nid in ids:
        meta = nodes_meta.get(nid, {}) or {}
        ntype = normalize_type(meta.get("type"))
        desc = meta.get("description", "")
        desc = wrap_label(first_n_words(desc, 3), width=30, max_lines=2)
        label = f"{nid}: {desc}" if desc else nid
        # label = nid
        G.add_node(nid, type=ntype, description=desc, label=label)

    # Add edges (parent -> child)
    for parent, childs in pc.items():
        p = str(parent)
        for child in (childs or []):
            c = str(child)
            if p in G and c in G:
                G.add_edge(p, c)

    return G, nx.get_node_attributes(G, "type")


def layered_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Simple DAG layered layout:
    - Compute levels by distance from roots (in-degree 0)
    - Place nodes per level left-to-right
    Falls back to spring_layout if graph has cycles.
    """
    try:
        if not nx.is_directed_acyclic_graph(G):
            raise nx.NetworkXUnfeasible

        # Roots: nodes with no predecessors
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if not roots:
            roots = list(G.nodes)[:1]

        # BFS levels
        level: Dict[str, int] = {}
        from collections import deque

        dq = deque()
        for r in roots:
            level[r] = 0
            dq.append(r)

        while dq:
            u = dq.popleft()
            for v in G.successors(u):
                cand = level[u] + 1
                if v not in level or cand > level[v]:
                    level[v] = cand
                    dq.append(v)

        # Group nodes by level
        layers: Dict[int, List[str]] = {}
        for n, lv in level.items():
            layers.setdefault(lv, []).append(n)
        # Add any nodes that were not reached
        for n in G.nodes:
            if n not in level:
                lv = 0
                level[n] = lv
                layers.setdefault(lv, []).append(n)

        # Coordinates
        pos: Dict[str, Tuple[float, float]] = {}
        for lv, nodes in sorted(layers.items()):
            nodes_sorted = sorted(nodes)
            count = len(nodes_sorted)
            for i, n in enumerate(nodes_sorted):
                # x spreads within the level, y goes top-down by level
                x = i - (count - 1) / 2.0
                y = -lv
                pos[n] = (x, y)

        return pos
    except Exception:
        # Fallback: spring layout
        return nx.spring_layout(G, seed=42)

# def draw_case(
#     ax: plt.Axes,
#     G: nx.DiGraph,
#     record: Dict[str, Any],
#     title: str,
#     *,
#     label_width: int = 18,       # characters per line
#     max_label_lines: int = 3,    # max wrapped lines
#     label_fontsize: int = 10,
# ):
#     pos = layered_layout(G)

#     # Node colors by type
#     types = nx.get_node_attributes(G, "type")
#     node_colors = [TYPE_COLORS.get(types.get(n, "Unknown"), TYPE_COLORS["Unknown"]) for n in G.nodes()]

#     # Draw nodes and edges
#     nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600, edgecolors="black", linewidths=0.5)
#     nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=10, width=1.2, edge_color="#444444")

#     # Wrapped labels (replace draw_networkx_labels)
#     labels = nx.get_node_attributes(G, "label")
#     for n, (x, y) in pos.items():
#         text = labels.get(n, str(n))
#         wrapped = wrap_label(text, width=label_width, max_lines=max_label_lines)
#         ax.text(
#             x, y, wrapped,
#             ha="center", va="center",
#             fontsize=label_fontsize, fontweight="bold",
#             bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.2"),
#         )

#     ax.set_title(title, fontsize=18, fontweight="bold")
#     ax.axis("off")
def draw_case(
    ax: plt.Axes,
    G: nx.DiGraph,
    record: Dict[str, Any],
    title: str,
    *,
    label_width: int = 18,       # characters per line
    max_label_lines: int = 3,    # max wrapped lines
    label_fontsize: int = 10,
    node_sat: float = 1.3,       # NEW: saturation multiplier
    node_val: float = 1.1,       # NEW: brightness multiplier
    node_size: int = 700,        # NEW: node size
    edge_width: float = 1.6,     # NEW: edge width
):
    pos = layered_layout(G)

    # Node colors by type (boosted saturation/brightness)
    types = nx.get_node_attributes(G, "type")
    node_colors = [
        boost_hex_color(
            TYPE_COLORS.get(types.get(n, "Unknown"), TYPE_COLORS["Unknown"]),
            sat=node_sat,
            val=node_val,
        )
        for n in G.nodes()
    ]

    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_size, edgecolors="black", linewidths=1.0
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12, width=edge_width, edge_color="#333333"
    )

    # Wrapped labels (replace draw_networkx_labels)
    labels = nx.get_node_attributes(G, "label")
    for n, (x, y) in pos.items():
        text = labels.get(n, str(n))
        wrapped = wrap_label(text, width=label_width, max_lines=max_label_lines)
        ax.text(
            x, y, wrapped,
            ha="center", va="center",
            fontsize=label_fontsize, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.25"),
        )

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.axis("off")

# --------------------- CLI --------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare assurance cases (LLM vs. human) from document_aware.json")
    p.add_argument("--json", type=str, required=True, help="Path to document_aware.json")
    p.add_argument("--human-docname", type=str, default="acas_xu.txt", help="Human docname to pick (exact match)")
    p.add_argument("--llm-tag", type=str, required=True, help='Substring tag to match LLM docname, e.g., "ACAS XU_2_0"')
    p.add_argument("--group", type=str, default=None, help="Optional group_docname filter (e.g., ACAS_XU)")
    p.add_argument("--out", type=str, default="assurance_case_compare.png", help="Output image path")
    # label wrapping controls
    p.add_argument("--label-width", type=int, default=18, help="Characters per line for node labels")
    p.add_argument("--label-lines", type=int, default=3, help="Max lines per node label")
    p.add_argument("--label-fontsize", type=int, default=10, help="Font size for node labels")
    # visual strength controls
    p.add_argument("--node-sat", type=float, default=1.3, help="Node color saturation multiplier (e.g., 1.3)")
    p.add_argument("--node-val", type=float, default=1.1, help="Node color brightness multiplier (e.g., 1.1)")
    p.add_argument("--node-size", type=int, default=2000, help="Node size")
    p.add_argument("--edge-width", type=float, default=1.6, help="Edge line width")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_json(args.json)

    records: List[Dict[str, Any]] = data.get("val", [])
    if args.group:
        records = [r for r in records if r.get("group_docname") == args.group]

    human_rec = pick_human_record(records, args.human_docname)
    llm_rec = pick_llm_record(records, args.llm_tag)

    if not human_rec:
        raise SystemExit(f"Human record not found for docname='{args.human_docname}' within group={args.group}.")
    if not llm_rec:
        raise SystemExit(f"LLM record not found for tag '{args.llm_tag}' within group={args.group}.")

    # Build graphs
    Gh, _ = build_graph(human_rec)
    Gl, _ = build_graph(llm_rec)

    # Titles
    ht = f"Human: {human_rec.get('docname', 'unknown')} ({human_rec.get('requirement', '')})"
    lt = f"LLM ({llm_rec.get('model_name','')}): {llm_rec.get('docname', 'unknown')}"

    # # Plot
    # fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    # draw_case(
    #     axes[0], Gh, human_rec, ht,
    #     label_width=args.label_width,
    #     max_label_lines=args.label_lines,
    #     label_fontsize=args.label_fontsize,
    # )
    # draw_case(
    #     axes[1], Gl, llm_rec, lt,
    #     label_width=args.label_width,
    #     max_label_lines=args.label_lines,
    #     label_fontsize=args.label_fontsize,
    # )
    # plt.tight_layout()
    # os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # plt.savefig(args.out, dpi=200)
    # print(f"Saved comparison to {args.out}")
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    draw_case(
        axes[0], Gh, human_rec, ht,
        label_width=args.label_width,
        max_label_lines=args.label_lines,
        label_fontsize=args.label_fontsize,
        node_sat=args.node_sat,
        node_val=args.node_val,
        node_size=args.node_size,
        edge_width=args.edge_width,
    )
    draw_case(
        axes[1], Gl, llm_rec, lt,
        label_width=args.label_width,
        max_label_lines=args.label_lines,
        label_fontsize=args.label_fontsize,
        node_sat=args.node_sat,
        node_val=args.node_val,
        node_size=args.node_size,
        edge_width=args.edge_width,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Saved comparison to {args.out}")


if __name__ == "__main__":
    main()