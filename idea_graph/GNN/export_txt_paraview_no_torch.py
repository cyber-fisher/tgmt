#!/usr/bin/env python3
"""Export dataset_txt GNN result to ParaView files without importing torch."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


GridEdge = Tuple[int, int]


@dataclass
class Node:
    node_id: int
    parent_id: int
    scalar: float
    children: List[int] = field(default_factory=list)
    region: Set[int] = field(default_factory=set)


@dataclass
class ActiveSubgraph:
    edges: List[GridEdge]
    vertices: Set[int]


def read_image(path: Path, scalar_name: str) -> Tuple[vtk.vtkImageData, np.ndarray]:
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = vtk.vtkImageData()
    image.DeepCopy(reader.GetOutput())
    point_data = image.GetPointData()
    array = point_data.GetArray(scalar_name)
    if array is None:
        array = point_data.GetScalars()
    if array is None and point_data.GetNumberOfArrays() > 0:
        array = point_data.GetArray(0)
    if array is None:
        raise ValueError(f"no scalar array found in {path}")
    return image, vtk_to_numpy(array).astype(np.float64, copy=False)


def read_tree(path: Path) -> Dict[int, Node]:
    nodes: Dict[int, Node] = {}
    with path.open() as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            node_id = int(parts[0])
            parent_id = int(parts[1])
            scalar = float(parts[2])
            nodes[node_id] = Node(node_id, parent_id, scalar)
    for node in nodes.values():
        if node.parent_id != node.node_id and node.parent_id >= 0:
            nodes[node.parent_id].children.append(node.node_id)
    roots = [node_id for node_id, node in nodes.items() if node.parent_id == node_id or node.parent_id < 0]
    if len(roots) != 1:
        raise ValueError(f"expected one root, got {len(roots)}")
    order: List[int] = []
    stack = [roots[0]]
    while stack:
        node_id = stack.pop()
        order.append(node_id)
        stack.extend(nodes[node_id].children)
    for node_id in reversed(order):
        region = {node_id}
        for child_id in nodes[node_id].children:
            region.update(nodes[child_id].region)
        nodes[node_id].region = region
    return nodes


def build_chains(nodes: Dict[int, Node], tree_type: int) -> List[List[int]]:
    parents: Dict[int, List[int]] = {node_id: [] for node_id in nodes}
    for node in nodes.values():
        if node.parent_id != node.node_id and node.parent_id >= 0:
            parents[node.node_id].append(node.parent_id)
    endpoints = [
        node_id for node_id, node in nodes.items()
        if len(parents[node_id]) + len(node.children) != 2
    ]
    visited: Set[Tuple[int, int]] = set()

    def key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    chains: List[List[int]] = []
    for start in endpoints:
        neighbors = parents[start] + nodes[start].children
        for nxt in neighbors:
            if key(start, nxt) in visited:
                continue
            chain = [start]
            prev = start
            cur = nxt
            while True:
                visited.add(key(prev, cur))
                chain.append(cur)
                cur_neighbors = parents[cur] + nodes[cur].children
                if len(cur_neighbors) != 2:
                    break
                candidates = [node_id for node_id in cur_neighbors if node_id != prev]
                if not candidates:
                    break
                prev, cur = cur, candidates[0]
            reverse = (tree_type == 0 and nodes[chain[0]].scalar > nodes[chain[-1]].scalar) or (
                tree_type != 0 and nodes[chain[0]].scalar < nodes[chain[-1]].scalar
            )
            if reverse:
                chain.reverse()
            if len(chain) >= 2:
                chains.append(chain)
    return chains


def bbox_axis_edges(dims: Tuple[int, int, int], imin: int, imax: int, jmin: int, jmax: int, kmin: int, kmax: int, axis: int):
    dx, dy, dz = dims
    if axis == 0:
        edge_imax = min(imax, dx - 2)
        if imin > edge_imax:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        k = np.arange(kmin, kmax + 1, dtype=np.int64)[:, None, None]
        j = np.arange(jmin, jmax + 1, dtype=np.int64)[None, :, None]
        i = np.arange(imin, edge_imax + 1, dtype=np.int64)[None, None, :]
        first = k * dy * dx + j * dx + i
        return first.ravel(), (first + 1).ravel()
    if axis == 1:
        edge_jmax = min(jmax, dy - 2)
        if jmin > edge_jmax:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        k = np.arange(kmin, kmax + 1, dtype=np.int64)[:, None, None]
        j = np.arange(jmin, edge_jmax + 1, dtype=np.int64)[None, :, None]
        i = np.arange(imin, imax + 1, dtype=np.int64)[None, None, :]
        first = k * dy * dx + j * dx + i
        return first.ravel(), (first + dx).ravel()
    edge_kmax = min(kmax, dz - 2)
    if kmin > edge_kmax:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    k = np.arange(kmin, edge_kmax + 1, dtype=np.int64)[:, None, None]
    j = np.arange(jmin, jmax + 1, dtype=np.int64)[None, :, None]
    i = np.arange(imin, imax + 1, dtype=np.int64)[None, None, :]
    first = k * dy * dx + j * dx + i
    return first.ravel(), (first + dx * dy).ravel()


def active_subgraph(node: Node, image: vtk.vtkImageData, scalar_values: np.ndarray) -> ActiveSubgraph:
    dims = image.GetDimensions()
    region = node.region
    ids = np.asarray(list(region), dtype=np.int64)
    i = ids % dims[0]
    j = (ids // dims[0]) % dims[1]
    k = ids // (dims[0] * dims[1])
    bbox = (int(i.min()), int(i.max()), int(j.min()), int(j.max()), int(k.min()), int(k.max()))
    region_values = ids
    edge_chunks: List[np.ndarray] = []
    vertex_chunks: List[np.ndarray] = []
    for axis in range(3):
        first, second = bbox_axis_edges(dims, *bbox, axis)
        if first.size == 0:
            continue
        v1 = scalar_values[first]
        v2 = scalar_values[second]
        mask = ((v1 <= node.scalar) & (v2 > node.scalar)) | ((v2 <= node.scalar) & (v1 > node.scalar))
        mask &= np.isin(first, region_values) | np.isin(second, region_values)
        if np.any(mask):
            a = first[mask].astype(np.int64, copy=False)
            b = second[mask].astype(np.int64, copy=False)
            edge_chunks.append(np.stack([a, b], axis=1))
            vertex_chunks.extend([a, b])
    if not edge_chunks:
        return ActiveSubgraph([], set())
    edge_array = np.concatenate(edge_chunks, axis=0)
    edge_array.sort(axis=1)
    edge_array = np.unique(edge_array, axis=0)
    vertices = set(int(v) for v in np.unique(np.concatenate(vertex_chunks)))
    return ActiveSubgraph([(int(a), int(b)) for a, b in edge_array], vertices)


def read_kept_candidates(path: Path) -> Dict[int, float]:
    kept: Dict[int, float] = {}
    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            if int(row["keep"]) == 1:
                kept[int(row["candidateNodeId"])] = float(row["distGNN"])
    return kept


def kept_node_set(nodes: Dict[int, Node], chains: Iterable[List[int]], kept_candidates: Dict[int, float]) -> Set[int]:
    kept = set(kept_candidates)
    for chain in chains:
        kept.add(chain[0])
        kept.add(chain[-1])
    return kept


def add_array(image: vtk.vtkImageData, values: np.ndarray, name: str) -> None:
    arr = numpy_to_vtk(values, deep=True)
    arr.SetName(name)
    image.GetPointData().AddArray(arr)


def write_volume(path: Path, image_in: vtk.vtkImageData, nodes: Dict[int, Node], kept_nodes: Set[int], kept_candidates: Dict[int, float], scalar_values: np.ndarray) -> Tuple[int, int]:
    image = vtk.vtkImageData()
    image.DeepCopy(image_in)
    mask = np.zeros(image.GetNumberOfPoints(), dtype=np.uint8)
    count = np.zeros(image.GetNumberOfPoints(), dtype=np.int32)
    max_dist = np.full(image.GetNumberOfPoints(), -1.0, dtype=np.float32)
    edge_count = 0
    for node_id in sorted(kept_nodes):
        sg = active_subgraph(nodes[node_id], image, scalar_values)
        if not sg.vertices:
            continue
        ids = np.asarray(list(sg.vertices), dtype=np.int64)
        mask[ids] = 1
        count[ids] += 1
        max_dist[ids] = np.maximum(max_dist[ids], kept_candidates.get(node_id, 0.0))
        edge_count += len(sg.edges)
    add_array(image, mask, "GNNKeptActiveVertex")
    add_array(image, count, "GNNKeptActiveNodeCount")
    add_array(image, max_dist, "GNNKeptMaxDistGNN")
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.Write()
    return int(mask.sum()), edge_count


def write_nodes(path: Path, image: vtk.vtkImageData, nodes: Dict[int, Node], kept_nodes: Set[int], kept_candidates: Dict[int, float]) -> None:
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    node_ids = vtk.vtkIntArray(); node_ids.SetName("NodeId")
    scalars = vtk.vtkDoubleArray(); scalars.SetName("Scalar")
    reasons = vtk.vtkIntArray(); reasons.SetName("GNNKeepReason")
    dists = vtk.vtkDoubleArray(); dists.SetName("distGNN")
    for idx, node_id in enumerate(sorted(kept_nodes)):
        points.InsertNextPoint(image.GetPoint(node_id))
        cells.InsertNextCell(1); cells.InsertCellPoint(idx)
        node_ids.InsertNextValue(node_id)
        scalars.InsertNextValue(nodes[node_id].scalar)
        reasons.InsertNextValue(1 if node_id in kept_candidates else 0)
        dists.InsertNextValue(kept_candidates.get(node_id, 0.0))
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_VERTEX, cells)
    for arr in [node_ids, scalars, reasons, dists]:
        grid.GetPointData().AddArray(arr)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.Write()


def write_edges(path: Path, image: vtk.vtkImageData, nodes: Dict[int, Node], kept_candidates: Dict[int, float], scalar_values: np.ndarray) -> int:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    source = vtk.vtkIntArray(); source.SetName("SourceNodeId")
    dists = vtk.vtkDoubleArray(); dists.SetName("distGNN")
    offset = 0
    count = 0
    for node_id, dist in sorted(kept_candidates.items()):
        sg = active_subgraph(nodes[node_id], image, scalar_values)
        for a, b in sg.edges:
            points.InsertNextPoint(image.GetPoint(a))
            points.InsertNextPoint(image.GetPoint(b))
            lines.InsertNextCell(2)
            lines.InsertCellPoint(offset)
            lines.InsertCellPoint(offset + 1)
            source.InsertNextValue(node_id)
            dists.InsertNextValue(dist)
            offset += 2
            count += 1
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_LINE, lines)
    grid.GetCellData().AddArray(source)
    grid.GetCellData().AddArray(dists)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.Write()
    return count


def write_values(path: Path, nodes: Dict[int, Node], kept_nodes: Set[int]) -> None:
    with path.open("w") as fp:
        fp.write("# nodeId,scalar\n")
        for node_id in sorted(kept_nodes, key=lambda nid: nodes[nid].scalar):
            fp.write(f"{node_id},{nodes[node_id].scalar:.17g}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-root", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("-d", "--distances", required=True, type=Path)
    parser.add_argument("-o", "--output-prefix", required=True, type=Path)
    parser.add_argument("--scalar", default="v")
    parser.add_argument("--tree-type", default=0, type=int, choices=[0, 1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tree = args.txt_root / "results" / args.case / "tree_output.txt"
    image_path = args.txt_root / "datas" / f"{args.case}.vti"
    image, scalar_values = read_image(image_path, args.scalar)
    nodes = read_tree(tree)
    chains = build_chains(nodes, args.tree_type)
    kept_candidates = read_kept_candidates(args.distances)
    kept_nodes = kept_node_set(nodes, chains, kept_candidates)
    volume = args.output_prefix.with_name(args.output_prefix.name + "_volume_for_contour.vti")
    node_file = args.output_prefix.with_name(args.output_prefix.name + "_kept_nodes.vtu")
    edge_file = args.output_prefix.with_name(args.output_prefix.name + "_kept_active_edges.vtu")
    values_file = args.output_prefix.with_name(args.output_prefix.name + "_contour_values.csv")
    active_points, all_edges = write_volume(volume, image, nodes, kept_nodes, kept_candidates, scalar_values)
    write_nodes(node_file, image, nodes, kept_nodes, kept_candidates)
    kept_edges = write_edges(edge_file, image, nodes, kept_candidates, scalar_values)
    write_values(values_file, nodes, kept_nodes)
    print(f"kept nodes: {len(kept_nodes)}")
    print(f"kept by GNN distance: {len(kept_candidates)}")
    print(f"active points marked in volume: {active_points}")
    print(f"active edges across all kept nodes: {all_edges}")
    print(f"active edges for GNN-kept candidates: {kept_edges}")
    print(f"wrote: {volume}")
    print(f"wrote: {node_file}")
    print(f"wrote: {edge_file}")
    print(f"wrote: {values_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

