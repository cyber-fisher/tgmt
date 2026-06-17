#!/usr/bin/env python3
"""Vectorize active merge-tree subgraphs with a deterministic 2-layer GCN.

The implementation mirrors the C++ active_graph_simplifier pipeline:

1. Read merge-tree nodes/arcs from VTU files and scalar/SegmentationId arrays
   from a VTI file.
2. For each merge-tree node, use the node scalar as an isovalue threshold.
3. Extract local active grid edges crossing that threshold, constrained by the
   incident segmentation ids around the current node.
4. Build per-vertex features and run an untrained deterministic 2-layer GCN.
5. Pool node embeddings into a fixed-length graph vector.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy


GridEdge = Tuple[int, int]


@dataclass(frozen=True)
class TreeNode:
    node_id: int
    scalar: float
    critical_type: int
    vertex_id: int
    position: Tuple[float, float, float]


@dataclass(frozen=True)
class TreeArc:
    arc_id: int
    segmentation_id: int
    down_node_id: int
    up_node_id: int


@dataclass
class TopologicalChain:
    chain_id: int
    node_ids: List[int]


@dataclass
class ActiveSubgraph:
    threshold: float
    active_edges: List[GridEdge]
    active_vertices: Set[int]
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    diag_length: float


@dataclass
class GraphTensors:
    features: torch.Tensor
    edge_index: List[GridEdge]
    graph_stats: torch.Tensor


def _require_array(attributes: vtk.vtkDataSetAttributes, name: str) -> vtk.vtkDataArray:
    array = attributes.GetArray(name)
    if array is None:
        raise ValueError(f"missing required VTK array: {name}")
    return array


def _optional_array(attributes: vtk.vtkDataSetAttributes, name: str) -> Optional[vtk.vtkDataArray]:
    return attributes.GetArray(name)


class MergeTreeData:
    def __init__(self, nodes_path: Path, arcs_path: Path, segmentation_path: Path, scalar_name: str):
        self.nodes_path = nodes_path
        self.arcs_path = arcs_path
        self.segmentation_path = segmentation_path
        self.scalar_name = scalar_name

        self.nodes: Dict[int, TreeNode] = {}
        self.arcs: Dict[int, TreeArc] = {}
        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = {}
        self.incident_seg_ids: Dict[int, Set[int]] = {}
        self.image_data: vtk.vtkImageData
        self.scalar_array: vtk.vtkDataArray
        self.segmentation_array: vtk.vtkDataArray
        self.scalar_values: np.ndarray
        self.segmentation_values: np.ndarray
        self.local_regions: Dict[int, Set[int]] = {}

    def load(self) -> None:
        self._read_nodes()
        self._read_arcs()
        self._read_segmentation()
        self._build_adjacency()

    def _read_nodes(self) -> None:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(self.nodes_path))
        reader.Update()
        grid = reader.GetOutput()
        if grid is None or grid.GetNumberOfPoints() == 0:
            raise ValueError(f"failed to read tree nodes: {self.nodes_path}")

        point_data = grid.GetPointData()
        node_ids = _require_array(point_data, "NodeId")
        scalars = _require_array(point_data, "Scalar")
        critical_types = _optional_array(point_data, "CriticalType")
        vertex_ids = _optional_array(point_data, "VertexId")

        self.nodes.clear()
        for point_idx in range(grid.GetNumberOfPoints()):
            node_id = int(node_ids.GetComponent(point_idx, 0))
            pos = grid.GetPoint(point_idx)
            self.nodes[node_id] = TreeNode(
                node_id=node_id,
                scalar=float(scalars.GetComponent(point_idx, 0)),
                critical_type=int(critical_types.GetComponent(point_idx, 0)) if critical_types else 0,
                vertex_id=int(vertex_ids.GetComponent(point_idx, 0)) if vertex_ids else -1,
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
            )

    def _read_arcs(self) -> None:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(self.arcs_path))
        reader.Update()
        grid = reader.GetOutput()
        if grid is None:
            raise ValueError(f"failed to read tree arcs: {self.arcs_path}")

        cell_data = grid.GetCellData()
        down_ids = _require_array(cell_data, "downNodeId")
        up_ids = _require_array(cell_data, "upNodeId")
        seg_ids = _optional_array(cell_data, "SegmentationId")

        self.arcs.clear()
        for cell_idx in range(grid.GetNumberOfCells()):
            self.arcs[cell_idx] = TreeArc(
                arc_id=cell_idx,
                segmentation_id=int(seg_ids.GetComponent(cell_idx, 0)) if seg_ids else -1,
                down_node_id=int(down_ids.GetComponent(cell_idx, 0)),
                up_node_id=int(up_ids.GetComponent(cell_idx, 0)),
            )

    def _read_segmentation(self) -> None:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(self.segmentation_path))
        reader.Update()
        image = reader.GetOutput()
        if image is None or image.GetNumberOfPoints() == 0:
            raise ValueError(f"failed to read image data: {self.segmentation_path}")

        point_data = image.GetPointData()
        self.scalar_array = _optional_array(point_data, self.scalar_name)
        if self.scalar_array is None:
            self.scalar_array = point_data.GetScalars()
        if self.scalar_array is None and point_data.GetNumberOfArrays() > 0:
            self.scalar_array = point_data.GetArray(0)
        if self.scalar_array is None:
            raise ValueError(f"missing required scalar VTK array: {self.scalar_name}")
        self.scalar_name = self.scalar_array.GetName() or self.scalar_name
        self.segmentation_array = _optional_array(point_data, "SegmentationId")
        self.scalar_values = vtk_to_numpy(self.scalar_array).astype(np.float64, copy=False)
        if self.segmentation_array is not None:
            self.segmentation_values = vtk_to_numpy(self.segmentation_array).astype(np.int64, copy=False)
        else:
            self.segmentation_values = np.zeros(image.GetNumberOfPoints(), dtype=np.int64)
        self.image_data = image

    def _build_adjacency(self) -> None:
        self.children = {node_id: [] for node_id in self.nodes}
        self.parents = {node_id: [] for node_id in self.nodes}
        self.incident_seg_ids = {node_id: set() for node_id in self.nodes}

        for arc in self.arcs.values():
            self.children.setdefault(arc.down_node_id, []).append(arc.up_node_id)
            self.parents.setdefault(arc.up_node_id, []).append(arc.down_node_id)
            if arc.segmentation_id >= 0:
                self.incident_seg_ids.setdefault(arc.down_node_id, set()).add(arc.segmentation_id)
                self.incident_seg_ids.setdefault(arc.up_node_id, set()).add(arc.segmentation_id)

    def build_topological_chains(self) -> List[TopologicalChain]:
        chains: List[TopologicalChain] = []
        critical_nodes = []
        for node_id, node in self.nodes.items():
            in_degree = len(self.parents.get(node_id, []))
            out_degree = len(self.children.get(node_id, []))
            if node.critical_type != 0 or in_degree == 0 or out_degree == 0 or out_degree > 1:
                critical_nodes.append(node_id)

        chain_id = 0
        for start_node_id in critical_nodes:
            for next_node_id in self.children.get(start_node_id, []):
                node_ids = [start_node_id]
                current = next_node_id
                while True:
                    node_ids.append(current)
                    if (
                        self.nodes[current].critical_type != 0
                        or len(self.children.get(current, [])) != 1
                        or len(self.parents.get(current, [])) != 1
                    ):
                        break
                    current = self.children[current][0]

                if len(node_ids) >= 2:
                    node_ids.sort(key=lambda nid: self.nodes[nid].scalar)
                    chains.append(TopologicalChain(chain_id=chain_id, node_ids=node_ids))
                    chain_id += 1

        return chains


class TxtTreeData(MergeTreeData):
    def __init__(self, tree_path: Path, image_path: Path, scalar_name: str, tree_type: int = 0):
        super().__init__(tree_path, Path(), image_path, scalar_name)
        self.tree_path = tree_path
        self.tree_type = tree_type

    def load(self) -> None:
        self._read_segmentation()
        self._read_tree_txt()
        self._build_adjacency()
        self._compute_local_regions()

    def _read_tree_txt(self) -> None:
        self.nodes.clear()
        with self.tree_path.open() as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                node_id = int(parts[0])
                parent_id = int(parts[1])
                scalar = float(parts[2])
                if node_id < 0 or node_id >= self.image_data.GetNumberOfPoints():
                    raise ValueError(f"tree node id {node_id} is outside VTI point range")
                pos = self.image_data.GetPoint(node_id)
                self.nodes[node_id] = TreeNode(
                    node_id=node_id,
                    scalar=scalar,
                    critical_type=0,
                    vertex_id=node_id,
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                )
                self.parents[node_id] = [] if parent_id == node_id or parent_id < 0 else [parent_id]

        if not self.nodes:
            raise ValueError(f"empty tree txt: {self.tree_path}")

        self.arcs.clear()
        for node_id in sorted(self.nodes):
            parent_ids = self.parents.get(node_id, [])
            if not parent_ids:
                continue
            parent_id = parent_ids[0]
            if parent_id not in self.nodes:
                raise ValueError(f"parent id {parent_id} not found for node {node_id}")
            arc_id = len(self.arcs)
            self.arcs[arc_id] = TreeArc(
                arc_id=arc_id,
                segmentation_id=-1,
                down_node_id=node_id,
                up_node_id=parent_id,
            )

    def _build_adjacency(self) -> None:
        self.children = {node_id: [] for node_id in self.nodes}
        normalized_parents = {node_id: [] for node_id in self.nodes}
        self.incident_seg_ids = {node_id: set() for node_id in self.nodes}

        for arc in self.arcs.values():
            self.children.setdefault(arc.up_node_id, []).append(arc.down_node_id)
            normalized_parents.setdefault(arc.down_node_id, []).append(arc.up_node_id)

        self.parents = normalized_parents
        for node_id, node in list(self.nodes.items()):
            child_count = len(self.children.get(node_id, []))
            parent_count = len(self.parents.get(node_id, []))
            critical_type = 3 if parent_count == 0 else (0 if child_count == 0 else (2 if child_count > 1 else 1))
            self.nodes[node_id] = TreeNode(
                node_id=node.node_id,
                scalar=node.scalar,
                critical_type=critical_type,
                vertex_id=node.vertex_id,
                position=node.position,
            )

    def _compute_local_regions(self) -> None:
        roots = [node_id for node_id in self.nodes if not self.parents.get(node_id)]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root in txt tree, got {len(roots)}")

        postorder: List[int] = []
        stack = [roots[0]]
        while stack:
            node_id = stack.pop()
            postorder.append(node_id)
            stack.extend(self.children.get(node_id, []))

        self.local_regions = {}
        for node_id in reversed(postorder):
            region = {node_id}
            for child_id in self.children.get(node_id, []):
                region.update(self.local_regions[child_id])
            self.local_regions[node_id] = region

    def build_topological_chains(self) -> List[TopologicalChain]:
        chains: List[TopologicalChain] = []
        endpoints = [
            node_id
            for node_id in self.nodes
            if len(self.parents.get(node_id, [])) + len(self.children.get(node_id, [])) != 2
        ]
        visited_edges: Set[Tuple[int, int]] = set()

        def edge_key(a: int, b: int) -> Tuple[int, int]:
            return (a, b) if a < b else (b, a)

        branch_id = 0
        for start_id in endpoints:
            neighbors = self.parents.get(start_id, []) + self.children.get(start_id, [])
            for next_id in neighbors:
                key = edge_key(start_id, next_id)
                if key in visited_edges:
                    continue
                node_ids = [start_id]
                prev_id = start_id
                current_id = next_id
                while True:
                    visited_edges.add(edge_key(prev_id, current_id))
                    node_ids.append(current_id)
                    degree = len(self.parents.get(current_id, [])) + len(self.children.get(current_id, []))
                    if degree != 2:
                        break
                    current_neighbors = self.parents.get(current_id, []) + self.children.get(current_id, [])
                    next_candidates = [nid for nid in current_neighbors if nid != prev_id]
                    if not next_candidates:
                        break
                    prev_id, current_id = current_id, next_candidates[0]

                first_scalar = self.nodes[node_ids[0]].scalar
                last_scalar = self.nodes[node_ids[-1]].scalar
                reverse = (self.tree_type == 0 and first_scalar > last_scalar) or (
                    self.tree_type != 0 and first_scalar < last_scalar
                )
                if reverse:
                    node_ids.reverse()
                if len(node_ids) >= 2:
                    chains.append(TopologicalChain(branch_id, node_ids))
                    branch_id += 1
        return chains


class ActiveSubgraphExtractor:
    def __init__(self, data: MergeTreeData):
        self.data = data
        self.dims = data.image_data.GetDimensions()
        self.origin = np.asarray(data.image_data.GetOrigin(), dtype=np.float64)
        self.spacing = np.asarray(data.image_data.GetSpacing(), dtype=np.float64)
        self.scalar_min, self.scalar_max = self._compute_scalar_range()
        self._segmentation_bboxes: Optional[Dict[int, Tuple[int, int, int, int, int, int]]] = None

    def _compute_scalar_range(self) -> Tuple[float, float]:
        return float(np.min(self.data.scalar_values)), float(np.max(self.data.scalar_values))

    def _point_id(self, i: int, j: int, k: int) -> int:
        return int(k * self.dims[1] * self.dims[0] + j * self.dims[0] + i)

    def _ijk_from_point_id(self, point_id: int) -> Tuple[int, int, int]:
        i = point_id % self.dims[0]
        j = (point_id // self.dims[0]) % self.dims[1]
        k = point_id // (self.dims[0] * self.dims[1])
        return int(i), int(j), int(k)

    def _build_segmentation_bboxes(self) -> Dict[int, Tuple[int, int, int, int, int, int]]:
        if self._segmentation_bboxes is not None:
            return self._segmentation_bboxes

        bboxes: Dict[int, Tuple[int, int, int, int, int, int]] = {}
        needed_seg_ids = {
            arc.segmentation_id for arc in self.data.arcs.values() if arc.segmentation_id >= 0
        }
        for seg_id in needed_seg_ids:
            point_ids = np.flatnonzero(self.data.segmentation_values == seg_id)
            if point_ids.size == 0:
                continue
            i = point_ids % self.dims[0]
            j = (point_ids // self.dims[0]) % self.dims[1]
            k = point_ids // (self.dims[0] * self.dims[1])
            bboxes[seg_id] = (
                int(np.min(i)),
                int(np.max(i)),
                int(np.min(j)),
                int(np.max(j)),
                int(np.min(k)),
                int(np.max(k)),
            )

        self._segmentation_bboxes = bboxes
        return self._segmentation_bboxes

    def _merged_bbox(self, seg_ids: Set[int]) -> Tuple[int, int, int, int, int, int]:
        if not seg_ids:
            return 0, self.dims[0] - 1, 0, self.dims[1] - 1, 0, self.dims[2] - 1

        bboxes = self._build_segmentation_bboxes()
        ranges = [bboxes[seg_id] for seg_id in seg_ids if seg_id in bboxes]
        if not ranges:
            return 1, 0, 1, 0, 1, 0

        return (
            min(b[0] for b in ranges),
            max(b[1] for b in ranges),
            min(b[2] for b in ranges),
            max(b[3] for b in ranges),
            min(b[4] for b in ranges),
            max(b[5] for b in ranges),
        )

    def extract(self, node_id: int, seg_ids: Optional[Set[int]] = None) -> ActiveSubgraph:
        node = self.data.nodes[node_id]
        threshold = node.scalar
        local_region = self.data.local_regions.get(node_id)
        if local_region is not None:
            return self._extract_from_local_region(threshold, local_region)

        seg_ids = set(seg_ids if seg_ids is not None else self.data.incident_seg_ids.get(node_id, set()))
        imin, imax, jmin, jmax, kmin, kmax = self._merged_bbox(seg_ids)

        edge_chunks: List[np.ndarray] = []
        vertex_chunks: List[np.ndarray] = []
        if imax >= imin and jmax >= jmin and kmax >= kmin:
            for axis in range(3):
                first, second = self._bbox_axis_edges(imin, imax, jmin, jmax, kmin, kmax, axis)
                if first.size == 0:
                    continue
                values_first = self.data.scalar_values[first]
                values_second = self.data.scalar_values[second]
                mask = ((values_first <= threshold) & (values_second > threshold)) | (
                    (values_second <= threshold) & (values_first > threshold)
                )
                if seg_ids:
                    seg_first = self.data.segmentation_values[first]
                    seg_second = self.data.segmentation_values[second]
                    allowed = np.isin(seg_first, list(seg_ids)) | np.isin(seg_second, list(seg_ids))
                    mask &= allowed
                if np.any(mask):
                    active_first = first[mask].astype(np.int64, copy=False)
                    active_second = second[mask].astype(np.int64, copy=False)
                    edge_chunks.append(np.stack([active_first, active_second], axis=1))
                    vertex_chunks.append(active_first)
                    vertex_chunks.append(active_second)

        edges: Set[GridEdge]
        vertices: Set[int]
        if edge_chunks:
            edge_array = np.concatenate(edge_chunks, axis=0)
            edge_array.sort(axis=1)
            edge_array = np.unique(edge_array, axis=0)
            edges = {(int(a), int(b)) for a, b in edge_array}
            vertices = set(int(v) for v in np.unique(np.concatenate(vertex_chunks)))
        else:
            edges = set()
            vertices = set()
        centroid, bbox_min, bbox_max, diag_length = self._geometry(vertices)
        return ActiveSubgraph(
            threshold=threshold,
            active_edges=sorted(edges),
            active_vertices=vertices,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            diag_length=diag_length,
        )

    def _extract_from_local_region(self, threshold: float, local_region: Set[int]) -> ActiveSubgraph:
        if not local_region:
            zeros = np.zeros(3, dtype=np.float64)
            return ActiveSubgraph(threshold, [], set(), zeros, zeros.copy(), zeros.copy(), 0.0)

        point_ids = np.asarray(list(local_region), dtype=np.int64)
        i = point_ids % self.dims[0]
        j = (point_ids // self.dims[0]) % self.dims[1]
        k = point_ids // (self.dims[0] * self.dims[1])
        imin, imax = int(np.min(i)), int(np.max(i))
        jmin, jmax = int(np.min(j)), int(np.max(j))
        kmin, kmax = int(np.min(k)), int(np.max(k))

        edge_chunks: List[np.ndarray] = []
        vertex_chunks: List[np.ndarray] = []
        region_values = np.fromiter(local_region, dtype=np.int64)
        for axis in range(3):
            first, second = self._bbox_axis_edges(imin, imax, jmin, jmax, kmin, kmax, axis)
            if first.size == 0:
                continue
            values_first = self.data.scalar_values[first]
            values_second = self.data.scalar_values[second]
            mask = ((values_first <= threshold) & (values_second > threshold)) | (
                (values_second <= threshold) & (values_first > threshold)
            )
            mask &= np.isin(first, region_values) | np.isin(second, region_values)
            if np.any(mask):
                active_first = first[mask].astype(np.int64, copy=False)
                active_second = second[mask].astype(np.int64, copy=False)
                edge_chunks.append(np.stack([active_first, active_second], axis=1))
                vertex_chunks.append(active_first)
                vertex_chunks.append(active_second)

        if edge_chunks:
            edge_array = np.concatenate(edge_chunks, axis=0)
            edge_array.sort(axis=1)
            edge_array = np.unique(edge_array, axis=0)
            edges = {(int(a), int(b)) for a, b in edge_array}
            vertices = set(int(v) for v in np.unique(np.concatenate(vertex_chunks)))
        else:
            edges = set()
            vertices = set()

        centroid, bbox_min, bbox_max, diag_length = self._geometry(vertices)
        return ActiveSubgraph(
            threshold=threshold,
            active_edges=sorted(edges),
            active_vertices=vertices,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            diag_length=diag_length,
        )

    def _bbox_axis_edges(
        self,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
        kmin: int,
        kmax: int,
        axis: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if axis == 0:
            edge_imax = min(imax, self.dims[0] - 2)
            if imin > edge_imax:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
            k = np.arange(kmin, kmax + 1, dtype=np.int64)[:, None, None]
            j = np.arange(jmin, jmax + 1, dtype=np.int64)[None, :, None]
            i = np.arange(imin, edge_imax + 1, dtype=np.int64)[None, None, :]
            first = k * self.dims[1] * self.dims[0] + j * self.dims[0] + i
            return first.ravel().astype(np.int64, copy=False), (first + 1).ravel().astype(np.int64, copy=False)
        if axis == 1:
            edge_jmax = min(jmax, self.dims[1] - 2)
            if jmin > edge_jmax:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
            k = np.arange(kmin, kmax + 1, dtype=np.int64)[:, None, None]
            j = np.arange(jmin, edge_jmax + 1, dtype=np.int64)[None, :, None]
            i = np.arange(imin, imax + 1, dtype=np.int64)[None, None, :]
            first = k * self.dims[1] * self.dims[0] + j * self.dims[0] + i
            return first.ravel().astype(np.int64, copy=False), (first + self.dims[0]).ravel().astype(np.int64, copy=False)
        edge_kmax = min(kmax, self.dims[2] - 2)
        if kmin > edge_kmax:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        k = np.arange(kmin, edge_kmax + 1, dtype=np.int64)[:, None, None]
        j = np.arange(jmin, jmax + 1, dtype=np.int64)[None, :, None]
        i = np.arange(imin, imax + 1, dtype=np.int64)[None, None, :]
        first = k * self.dims[1] * self.dims[0] + j * self.dims[0] + i
        return first.ravel().astype(np.int64, copy=False), (
            first + self.dims[0] * self.dims[1]
        ).ravel().astype(np.int64, copy=False)

    def _geometry(self, vertices: Set[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if not vertices:
            zeros = np.zeros(3, dtype=np.float64)
            return zeros, zeros.copy(), zeros.copy(), 0.0

        point_ids = np.asarray(list(vertices), dtype=np.int64)
        points = self.points_from_ids(point_ids)
        centroid = points.mean(axis=0)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        diag_length = float(np.linalg.norm(bbox_max - bbox_min))
        return centroid, bbox_min, bbox_max, diag_length

    def points_from_ids(self, point_ids: np.ndarray) -> np.ndarray:
        i = point_ids % self.dims[0]
        j = (point_ids // self.dims[0]) % self.dims[1]
        k = point_ids // (self.dims[0] * self.dims[1])
        ijk = np.stack([i, j, k], axis=1).astype(np.float64, copy=False)
        return self.origin + ijk * self.spacing


class GcnGraphEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64, seed: int = 17, trainable: bool = False):
        super().__init__()
        self.hidden_dim = max(1, int(hidden_dim))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        self.layer1 = torch.nn.Linear(11, self.hidden_dim, bias=False)
        self.layer2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self._init_layer(self.layer1, generator)
        self._init_layer(self.layer2, generator)

        for param in self.parameters():
            param.requires_grad_(trainable)

    @staticmethod
    def _init_layer(layer: torch.nn.Linear, generator: torch.Generator) -> None:
        fan_in = layer.weight.shape[1]
        fan_out = layer.weight.shape[0]
        scale = math.sqrt(2.0 / max(1, fan_in + fan_out))
        with torch.no_grad():
            layer.weight.copy_((torch.rand(layer.weight.shape, generator=generator) * 2.0 - 1.0) * scale)

    @staticmethod
    def _gcn_aggregate(features: torch.Tensor, edge_index: Sequence[GridEdge], vertex_count: int) -> torch.Tensor:
        degree = torch.ones(vertex_count, dtype=features.dtype, device=features.device)
        if edge_index:
            edge_tensor = torch.tensor(edge_index, dtype=torch.long, device=features.device)
            src = edge_tensor[:, 0]
            dst = edge_tensor[:, 1]
            degree.index_add_(0, src, torch.ones_like(src, dtype=features.dtype))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=features.dtype))

        output = features / degree[:, None]
        if not edge_index:
            return output

        edge_tensor = torch.tensor(edge_index, dtype=torch.long, device=features.device)
        src = edge_tensor[:, 0]
        dst = edge_tensor[:, 1]
        norm = torch.rsqrt(degree[src] * degree[dst])

        messages_src_to_dst = features[src] * norm[:, None]
        messages_dst_to_src = features[dst] * norm[:, None]
        output.index_add_(0, dst, messages_src_to_dst)
        output.index_add_(0, src, messages_dst_to_src)
        return output

    def forward(self, features: torch.Tensor, edge_index: Sequence[GridEdge], vertex_count: int, graph_stats: torch.Tensor) -> torch.Tensor:
        if vertex_count == 0:
            return torch.zeros(self.hidden_dim * 2 + 2, dtype=torch.float32, device=features.device)

        hidden = torch.relu(self.layer1(self._gcn_aggregate(features, edge_index, vertex_count)))
        hidden = torch.relu(self.layer2(self._gcn_aggregate(hidden, edge_index, vertex_count)))
        pooled = torch.cat([hidden.mean(dim=0), hidden.max(dim=0).values, graph_stats], dim=0)
        return torch.nn.functional.normalize(pooled, p=2.0, dim=0)


class ActiveSubgraphVectorizer:
    def __init__(self, data: MergeTreeData, extractor: ActiveSubgraphExtractor, encoder: GcnGraphEncoder):
        self.data = data
        self.extractor = extractor
        self.encoder = encoder
        self.scalar_range = max(1.0e-12, extractor.scalar_max - extractor.scalar_min)

    def encode_node(self, node_id: int, seg_ids: Optional[Set[int]] = None) -> Tuple[ActiveSubgraph, np.ndarray]:
        subgraph = self.extractor.extract(node_id, seg_ids)
        graph = self.build_graph_tensors(subgraph)
        with torch.no_grad():
            embedding = self.encoder(
                graph.features,
                graph.edge_index,
                graph.features.shape[0],
                graph.graph_stats,
            )
        return subgraph, embedding.cpu().numpy().astype(np.float64)

    def build_graph_tensors(self, subgraph: ActiveSubgraph) -> GraphTensors:
        vertex_ids = sorted(subgraph.active_vertices)
        local_index = {point_id: idx for idx, point_id in enumerate(vertex_ids)}
        neighbors = [[] for _ in vertex_ids]
        local_edges: List[GridEdge] = []

        for a, b in subgraph.active_edges:
            if a not in local_index or b not in local_index:
                continue
            local_a = local_index[a]
            local_b = local_index[b]
            local_edges.append((local_a, local_b))
            neighbors[local_a].append(local_b)
            neighbors[local_b].append(local_a)

        spatial_scale = max(1.0e-12, subgraph.diag_length)
        features = np.zeros((len(vertex_ids), 11), dtype=np.float32)
        if vertex_ids:
            point_ids = np.asarray(vertex_ids, dtype=np.int64)
            scalars = self.data.scalar_values[point_ids]
            seg_ids = self.data.segmentation_values[point_ids].astype(np.float64, copy=False)
            points = self.extractor.points_from_ids(point_ids)
            centered = (points - subgraph.centroid[None, :]) / spatial_scale
            degrees = np.asarray([len(nbrs) for nbrs in neighbors], dtype=np.float32)

            features[:, 0] = ((scalars - self.extractor.scalar_min) / self.scalar_range).astype(np.float32)
            features[:, 1] = ((scalars - subgraph.threshold) / self.scalar_range).astype(np.float32)
            features[:, 2] = (np.abs(scalars - subgraph.threshold) / self.scalar_range).astype(np.float32)
            features[:, 3:6] = centered.astype(np.float32)
            features[:, 6] = degrees / 6.0
            features[:, 7] = np.sin(seg_ids * math.pi / 180.0).astype(np.float32)
            features[:, 8] = np.cos(seg_ids * math.pi / 180.0).astype(np.float32)
            features[:, 9] = np.sin(seg_ids / 1024.0).astype(np.float32)
            features[:, 10] = np.cos(seg_ids / 1024.0).astype(np.float32)

        graph_stats = torch.tensor(
            [math.log1p(len(vertex_ids)), math.log1p(len(local_edges))],
            dtype=torch.float32,
        )
        return GraphTensors(torch.from_numpy(features), local_edges, graph_stats)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1.0e-12 and norm_b <= 1.0e-12:
        return 0.0
    if norm_a <= 1.0e-12 or norm_b <= 1.0e-12:
        return 1.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return 1.0 - max(-1.0, min(1.0, cosine))


def write_embeddings(path: Path, embeddings: Dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = len(next(iter(embeddings.values()))) if embeddings else 0
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["nodeId", *[f"z{i}" for i in range(dim)]])
        for node_id in sorted(embeddings):
            writer.writerow([node_id, *[f"{value:.10g}" for value in embeddings[node_id]]])


def write_chain_distances(
    path: Path,
    data: MergeTreeData,
    vectorizer: ActiveSubgraphVectorizer,
    chains: Iterable[TopologicalChain],
    epsilon: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["chainId", "refNodeId", "candidateNodeId", "distGNN", "keep"])

        for chain in chains:
            if len(chain.node_ids) <= 2:
                continue
            ref_node_id = chain.node_ids[0]
            _, ref_embedding = vectorizer.encode_node(ref_node_id, data.incident_seg_ids.get(ref_node_id, set()))
            for candidate_node_id in chain.node_ids[1:-1]:
                _, candidate_embedding = vectorizer.encode_node(
                    candidate_node_id,
                    data.incident_seg_ids.get(candidate_node_id, set()),
                )
                distance = cosine_distance(ref_embedding, candidate_embedding)
                keep = distance >= epsilon
                writer.writerow([chain.chain_id, ref_node_id, candidate_node_id, f"{distance:.10g}", int(keep)])
                if keep:
                    ref_node_id = candidate_node_id
                    ref_embedding = candidate_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize active merge-tree subgraphs with a 2-layer GCN.")
    parser.add_argument("-n", "--nodes", type=Path, help="merge tree nodes VTU path")
    parser.add_argument("-a", "--arcs", type=Path, help="merge tree arcs VTU path")
    parser.add_argument("-s", "--segmentation", type=Path, help="scalar/segmentation VTI path")
    parser.add_argument("--txt-root", type=Path, help="dataset_txt root containing datas/ and results/")
    parser.add_argument("--case", help="case name under dataset_txt, e.g. case17_12x7x5")
    parser.add_argument("--tree-txt", type=Path, help="direct tree_output.txt path")
    parser.add_argument("--image", type=Path, help="direct VTI image path for --tree-txt")
    parser.add_argument("--tree-type", default=0, type=int, choices=[0, 1], help="txt tree type: 0=JT, 1=ST")
    parser.add_argument("-o", "--output-prefix", required=True, type=Path, help="output file prefix")
    parser.add_argument("--scalar", default="v", help="scalar field name in VTI, default: v")
    parser.add_argument("--hidden", default=64, type=int, help="GCN hidden dimension, default: 64")
    parser.add_argument("--seed", default=17, type=int, help="deterministic random weight seed, default: 17")
    parser.add_argument("--epsilon", default=0.15, type=float, help="chain keep threshold for GNN distance")
    parser.add_argument("--model", type=Path, help="trained GCN checkpoint path")
    return parser.parse_args()


def resolve_txt_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.txt_root is not None:
        if not args.case:
            raise ValueError("--case is required when using --txt-root")
        tree_path = args.txt_root / "results" / args.case / "tree_output.txt"
        image_path = args.txt_root / "datas" / f"{args.case}.vti"
        return tree_path, image_path
    if args.tree_txt is not None:
        if args.image is None:
            raise ValueError("--image is required when using --tree-txt")
        return args.tree_txt, args.image
    raise ValueError("txt paths were not provided")


def create_data_from_args(args: argparse.Namespace) -> MergeTreeData:
    if args.txt_root is not None or args.tree_txt is not None:
        tree_path, image_path = resolve_txt_paths(args)
        data = TxtTreeData(tree_path, image_path, args.scalar, args.tree_type)
    else:
        if args.nodes is None or args.arcs is None or args.segmentation is None:
            raise ValueError("VTU mode requires --nodes, --arcs and --segmentation")
        data = MergeTreeData(args.nodes, args.arcs, args.segmentation, args.scalar)
    data.load()
    return data


def load_encoder(hidden_dim: int, seed: int, model_path: Optional[Path] = None, trainable: bool = False) -> GcnGraphEncoder:
    if model_path is None:
        return GcnGraphEncoder(hidden_dim=hidden_dim, seed=seed, trainable=trainable)

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint_hidden = int(checkpoint.get("hidden_dim", hidden_dim))
    checkpoint_seed = int(checkpoint.get("seed", seed))
    encoder = GcnGraphEncoder(
        hidden_dim=checkpoint_hidden,
        seed=checkpoint_seed,
        trainable=trainable,
    )
    encoder.load_state_dict(checkpoint["model_state_dict"])
    return encoder


def main() -> int:
    args = parse_args()
    data = create_data_from_args(args)

    extractor = ActiveSubgraphExtractor(data)
    encoder = load_encoder(args.hidden, args.seed, args.model)
    vectorizer = ActiveSubgraphVectorizer(data, extractor, encoder)

    embeddings: Dict[int, np.ndarray] = {}
    for node_id in sorted(data.nodes):
        _, embedding = vectorizer.encode_node(node_id, data.incident_seg_ids.get(node_id, set()))
        embeddings[node_id] = embedding

    chains = data.build_topological_chains()
    embeddings_path = args.output_prefix.with_name(args.output_prefix.name + "_embeddings.csv")
    distances_path = args.output_prefix.with_name(args.output_prefix.name + "_chain_distances.csv")
    write_embeddings(embeddings_path, embeddings)
    write_chain_distances(distances_path, data, vectorizer, chains, args.epsilon)

    print(f"nodes: {len(data.nodes)}")
    print(f"arcs: {len(data.arcs)}")
    print(f"topological chains: {len(chains)}")
    print(f"embedding dimension: {encoder.hidden_dim * 2 + 2}")
    if args.model:
        print(f"model: {args.model}")
    print(f"wrote: {embeddings_path}")
    print(f"wrote: {distances_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
