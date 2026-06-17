#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Node:
    node_id: int
    parent_id: int
    scalar: float
    children: List[int] = field(default_factory=list)


def parse_tree(path: Path) -> Dict[int, Node]:
    nodes: Dict[int, Node] = {}
    with path.open("r", encoding="utf-8") as f:
      for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        node_id = int(parts[0])
        parent_id = int(parts[1])
        scalar = float(parts[2])
        nodes[node_id] = Node(node_id=node_id, parent_id=parent_id, scalar=scalar)

    if not nodes:
        raise ValueError(f"Tree file is empty: {path}")

    for node in nodes.values():
        if node.parent_id >= 0 and node.parent_id != node.node_id:
            if node.parent_id not in nodes:
                raise ValueError(f"Parent {node.parent_id} not found for node {node.node_id}")
            nodes[node.parent_id].children.append(node.node_id)

    for node in nodes.values():
        node.children.sort(key=lambda cid: nodes[cid].scalar)

    return nodes


def find_roots(nodes: Dict[int, Node]) -> List[int]:
    roots = [node.node_id for node in nodes.values() if node.parent_id < 0 or node.parent_id == node.node_id]
    if not roots:
        raise ValueError("Expected at least one root.")
    roots.sort(key=lambda nid: nodes[nid].scalar)
    return roots


def assign_x_positions(nodes: Dict[int, Node], root_ids: List[int]) -> Dict[int, float]:
    x_positions: Dict[int, float] = {}
    next_x = 0.0

    def dfs(node_id: int) -> float:
        nonlocal next_x
        node = nodes[node_id]
        if not node.children:
            x_positions[node_id] = next_x
            next_x += 1.0
            return x_positions[node_id]

        child_x = [dfs(child_id) for child_id in node.children]
        x_positions[node_id] = sum(child_x) / len(child_x)
        return x_positions[node_id]

    for root_id in root_ids:
        dfs(root_id)
        next_x += 1.5
    return x_positions


def classify_node(node: Node) -> str:
    if node.parent_id < 0 or node.parent_id == node.node_id:
        return "root"
    if not node.children:
        return "leaf"
    if len(node.children) > 1:
        return "branch"
    return "regular"


def draw_tree(nodes: Dict[int, Node], root_ids: List[int], output_path: Path, title: str) -> None:
    x_pos = assign_x_positions(nodes, root_ids)
    y_pos = {node_id: node.scalar for node_id, node in nodes.items()}

    categories = defaultdict(list)
    for node in nodes.values():
        categories[classify_node(node)].append(node.node_id)

    fig_w = max(14, min(28, len(nodes) / 18))
    fig_h = 12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)

    for node in nodes.values():
        if node.parent_id < 0 or node.parent_id == node.node_id:
            continue
        parent = nodes[node.parent_id]
        ax.plot(
            [x_pos[node.node_id], x_pos[parent.node_id]],
            [y_pos[node.node_id], y_pos[parent.node_id]],
            color="#6c757d",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )

    styles = {
        "root": {"color": "#d62828", "size": 55, "label": "root"},
        "branch": {"color": "#1d3557", "size": 20, "label": "branch"},
        "leaf": {"color": "#2a9d8f", "size": 14, "label": "leaf"},
        "regular": {"color": "#b0b8c1", "size": 8, "label": "regular"},
    }

    for key, node_ids in categories.items():
        style = styles[key]
        ax.scatter(
            [x_pos[nid] for nid in node_ids],
            [y_pos[nid] for nid in node_ids],
            s=style["size"],
            c=style["color"],
            label=f'{style["label"]} ({len(node_ids)})',
            edgecolors="none",
            zorder=2 if key != "regular" else 1.5,
        )

    for root_id in root_ids:
        root_scalar = nodes[root_id].scalar
        ax.axhline(root_scalar, color="#d62828", linewidth=0.6, linestyle="--", alpha=0.15)

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Tree Layout Position")
    ax.set_ylabel("Scalar")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    ax.margins(x=0.02, y=0.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an input merge tree from tree_output.txt.")
    parser.add_argument("tree_path", type=Path, help="Path to tree_output.txt")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", type=str, default="", help="Figure title")
    args = parser.parse_args()

    nodes = parse_tree(args.tree_path)
    root_ids = find_roots(nodes)
    title = args.title or f"Merge Tree Visualization: {args.tree_path.stem}"
    draw_tree(nodes, root_ids, args.output, title)
    print(f"Saved tree visualization to {args.output}")


if __name__ == "__main__":
    main()
