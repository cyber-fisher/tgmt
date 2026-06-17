#!/usr/bin/env python3
"""Export GNN simplification results as ParaView-friendly VTK files.

The main output is a vtkImageData (.vti) copy of the original volume with extra
point-data arrays. Select this .vti in ParaView before applying Contour.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from vectorize_active_subgraphs import ActiveSubgraphExtractor, create_data_from_args


def read_kept_candidates(distances_path: Path) -> Dict[int, float]:
    kept: Dict[int, float] = {}
    with distances_path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            if int(row["keep"]) == 1:
                kept[int(row["candidateNodeId"])] = float(row["distGNN"])
    return kept


def build_simplified_node_set(data: MergeTreeData, kept_candidates: Dict[int, float]) -> Set[int]:
    kept: Set[int] = set(kept_candidates)
    for chain in data.build_topological_chains():
        if chain.node_ids:
            kept.add(chain.node_ids[0])
            kept.add(chain.node_ids[-1])
    return kept


def add_numpy_array(image: vtk.vtkImageData, values: np.ndarray, name: str) -> None:
    vtk_array = numpy_to_vtk(values, deep=True)
    vtk_array.SetName(name)
    image.GetPointData().AddArray(vtk_array)


def write_volume_vti(
    output_path: Path,
    data,
    extractor: ActiveSubgraphExtractor,
    kept_nodes: Iterable[int],
    kept_candidates: Dict[int, float],
) -> Tuple[int, int]:
    image = vtk.vtkImageData()
    image.DeepCopy(data.image_data)

    point_count = image.GetNumberOfPoints()
    active_vertex_mask = np.zeros(point_count, dtype=np.uint8)
    active_node_count = np.zeros(point_count, dtype=np.int32)
    max_dist = np.full(point_count, -1.0, dtype=np.float32)

    total_edges = 0
    for node_id in sorted(kept_nodes):
        subgraph = extractor.extract(node_id, data.incident_seg_ids.get(node_id, set()))
        if not subgraph.active_vertices:
            continue
        point_ids = np.asarray(list(subgraph.active_vertices), dtype=np.int64)
        active_vertex_mask[point_ids] = 1
        active_node_count[point_ids] += 1
        dist = kept_candidates.get(node_id, 0.0)
        max_dist[point_ids] = np.maximum(max_dist[point_ids], dist)
        total_edges += len(subgraph.active_edges)

    add_numpy_array(image, active_vertex_mask, "GNNKeptActiveVertex")
    add_numpy_array(image, active_node_count, "GNNKeptActiveNodeCount")
    add_numpy_array(image, max_dist, "GNNKeptMaxDistGNN")

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(image)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write {output_path}")

    return int(np.count_nonzero(active_vertex_mask)), total_edges


def write_kept_nodes_vtu(
    output_path: Path,
    data,
    kept_nodes: Iterable[int],
    kept_candidates: Dict[int, float],
) -> None:
    kept_ids = sorted(kept_nodes)

    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    node_ids = vtk.vtkIntArray()
    node_ids.SetName("NodeId")
    scalars = vtk.vtkDoubleArray()
    scalars.SetName("Scalar")
    keep_reason = vtk.vtkIntArray()
    keep_reason.SetName("GNNKeepReason")
    dist_array = vtk.vtkDoubleArray()
    dist_array.SetName("distGNN")

    for idx, node_id in enumerate(kept_ids):
        node = data.nodes[node_id]
        points.InsertNextPoint(node.position)
        cells.InsertNextCell(1)
        cells.InsertCellPoint(idx)
        node_ids.InsertNextValue(node_id)
        scalars.InsertNextValue(node.scalar)
        keep_reason.InsertNextValue(1 if node_id in kept_candidates else 0)
        dist_array.InsertNextValue(kept_candidates.get(node_id, 0.0))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_VERTEX, cells)
    grid.GetPointData().AddArray(node_ids)
    grid.GetPointData().AddArray(scalars)
    grid.GetPointData().AddArray(keep_reason)
    grid.GetPointData().AddArray(dist_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write {output_path}")


def write_active_edges_vtu(
    output_path: Path,
    data,
    extractor: ActiveSubgraphExtractor,
    kept_candidates: Dict[int, float],
) -> int:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    source_node = vtk.vtkIntArray()
    source_node.SetName("SourceNodeId")
    dist_array = vtk.vtkDoubleArray()
    dist_array.SetName("distGNN")

    point_offset = 0
    line_count = 0
    for node_id, dist in sorted(kept_candidates.items()):
        subgraph = extractor.extract(node_id, data.incident_seg_ids.get(node_id, set()))
        for a, b in subgraph.active_edges:
            pa = data.image_data.GetPoint(a)
            pb = data.image_data.GetPoint(b)
            points.InsertNextPoint(pa)
            points.InsertNextPoint(pb)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(point_offset)
            lines.InsertCellPoint(point_offset + 1)
            source_node.InsertNextValue(node_id)
            dist_array.InsertNextValue(dist)
            point_offset += 2
            line_count += 1

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_LINE, lines)
    grid.GetCellData().AddArray(source_node)
    grid.GetCellData().AddArray(dist_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write {output_path}")
    return line_count


def write_contour_values(output_path: Path, data, kept_nodes: Iterable[int]) -> None:
    with output_path.open("w") as fp:
        fp.write("# nodeId,scalar\n")
        for node_id in sorted(kept_nodes, key=lambda nid: data.nodes[nid].scalar):
            fp.write(f"{node_id},{data.nodes[node_id].scalar:.17g}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GNN result VTK files for ParaView.")
    parser.add_argument("-n", "--nodes", type=Path, help="merge tree nodes VTU path")
    parser.add_argument("-a", "--arcs", type=Path, help="merge tree arcs VTU path")
    parser.add_argument("-s", "--segmentation", type=Path, help="original scalar/segmentation VTI path")
    parser.add_argument("--txt-root", type=Path, help="dataset_txt root containing datas/ and results/")
    parser.add_argument("--case", help="case name under dataset_txt, e.g. case17_12x7x5")
    parser.add_argument("--tree-txt", type=Path, help="direct tree_output.txt path")
    parser.add_argument("--image", type=Path, help="direct VTI image path for --tree-txt")
    parser.add_argument("--tree-type", default=0, type=int, choices=[0, 1], help="txt tree type: 0=JT, 1=ST")
    parser.add_argument("-d", "--distances", required=True, type=Path, help="GNN chain_distances.csv path")
    parser.add_argument("-o", "--output-prefix", required=True, type=Path, help="output file prefix")
    parser.add_argument("--scalar", default="v", help="scalar field name in VTI, default: v")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = create_data_from_args(args)
    extractor = ActiveSubgraphExtractor(data)
    kept_candidates = read_kept_candidates(args.distances)
    kept_nodes = build_simplified_node_set(data, kept_candidates)

    volume_path = args.output_prefix.with_name(args.output_prefix.name + "_volume_for_contour.vti")
    nodes_path = args.output_prefix.with_name(args.output_prefix.name + "_kept_nodes.vtu")
    edges_path = args.output_prefix.with_name(args.output_prefix.name + "_kept_active_edges.vtu")
    values_path = args.output_prefix.with_name(args.output_prefix.name + "_contour_values.csv")

    active_point_count, total_active_edges = write_volume_vti(
        volume_path, data, extractor, kept_nodes, kept_candidates
    )
    write_kept_nodes_vtu(nodes_path, data, kept_nodes, kept_candidates)
    kept_candidate_edge_count = write_active_edges_vtu(edges_path, data, extractor, kept_candidates)
    write_contour_values(values_path, data, kept_nodes)

    print(f"kept nodes: {len(kept_nodes)}")
    print(f"kept by GNN distance: {len(kept_candidates)}")
    print(f"active points marked in volume: {active_point_count}")
    print(f"active edges across all kept nodes: {total_active_edges}")
    print(f"active edges for GNN-kept candidates: {kept_candidate_edge_count}")
    print(f"wrote: {volume_path}")
    print(f"wrote: {nodes_path}")
    print(f"wrote: {edges_path}")
    print(f"wrote: {values_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
