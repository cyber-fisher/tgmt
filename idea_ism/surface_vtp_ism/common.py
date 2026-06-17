from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math
from collections import defaultdict, deque

import vtk


@dataclass
class NodeRecord:
    node_id: int
    parent_id: int
    scalar: float
    point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    children: List[int] = field(default_factory=list)
    incident_arc_ids: List[int] = field(default_factory=list)
    degree: int = 0
    critical_type: int = 0
    region_size: float = 1.0
    region_span: float = 0.0


@dataclass
class ArcRecord:
    arc_id: int
    down_node_id: int
    up_node_id: int
    region_size: float
    region_span: float


@dataclass
class BranchRecord:
    branch_id: int
    node_ids: List[int]
    arc_ids: List[int]
    keep_indices: List[int] = field(default_factory=list)


@dataclass
class NewArcRecord:
    new_arc_id: int
    down_node_id: int
    up_node_id: int
    region_size: float
    region_span: float
    merged_arc_ids: List[int]


@dataclass
class SimplifiedTree:
    kept_node_ids: List[int]
    new_arcs: List[NewArcRecord]
    node_keep_mask: Dict[int, int]


@dataclass
class DistanceRecord:
    branch_id: int
    ref_node_id: int
    candidate_node_id: int
    dist_ism: float
    dist_pos: float
    dist_bbox: float
    kept: int


@dataclass
class SimplificationResult:
    tree: SimplifiedTree
    distances: List[DistanceRecord]


@dataclass
class SurfaceDescriptor:
    node_id: int
    iso_value: float
    active_cell_ids: List[int]
    distance_field: List[float]
    centroid: Tuple[float, float, float]
    bbox_diagonal: float


@dataclass
class ProgramOptions:
    tree_path: Path
    surface_path: Path
    output_prefix: Path
    scalar_array_name: str
    epsilon_ism: float
    epsilon_pos: float
    epsilon_bbox: float
    histogram_bins: int
    tree_type: int
    single_component_only: bool
    mode_name: str


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def parse_args(description: str, mode_name: str, single_component_only: bool) -> ProgramOptions:
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-t", "--tree", required=True, type=Path)
    parser.add_argument("-s", "--surface", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-e", "--epsilon-ism", type=float, default=0.15)
    parser.add_argument("-p", "--epsilon-pos", type=float, default=1.0)
    parser.add_argument("-b", "--epsilon-bbox", type=float, default=0.10)
    parser.add_argument("-n", "--hist-bins", type=int, default=32)
    parser.add_argument("-T", "--tree-type", type=int, default=0)
    parser.add_argument("-a", "--scalar-array", type=str, default="")
    args = parser.parse_args()

    return ProgramOptions(
        tree_path=args.tree,
        surface_path=args.surface,
        output_prefix=args.output,
        scalar_array_name=args.scalar_array,
        epsilon_ism=args.epsilon_ism,
        epsilon_pos=args.epsilon_pos,
        epsilon_bbox=args.epsilon_bbox,
        histogram_bins=args.hist_bins,
        tree_type=args.tree_type,
        single_component_only=single_component_only,
        mode_name=mode_name,
    )


def read_surface(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    surface = reader.GetOutput()
    if surface is None or surface.GetNumberOfPoints() == 0:
        fail(f"Unable to read surface `{path}`.")
    out = vtk.vtkPolyData()
    out.DeepCopy(surface)
    return out


def get_scalar_array(surface: vtk.vtkPolyData, requested_name: str) -> vtk.vtkDataArray:
    pd = surface.GetPointData()
    if pd is None:
        fail("Surface point data is missing.")
    if requested_name:
        arr = pd.GetArray(requested_name)
        if arr is None:
            fail(f"Scalar array `{requested_name}` not found.")
        return arr
    arr = pd.GetScalars()
    if arr is not None:
        return arr
    if pd.GetNumberOfArrays() <= 0:
        fail("Surface has no point arrays.")
    arr = pd.GetArray(0)
    if arr is None:
        fail("Failed to access first point array.")
    return arr


def read_tree_txt(path: Path) -> List[NodeRecord]:
    nodes: List[NodeRecord] = []
    seen: Dict[int, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        node_id, parent_id, scalar = int(parts[0]), int(parts[1]), float(parts[2])
        if node_id in seen:
            fail(f"Duplicate node id `{node_id}`.")
        seen[node_id] = len(nodes)
        nodes.append(NodeRecord(node_id=node_id, parent_id=parent_id, scalar=scalar))
    if not nodes:
        fail(f"Tree txt is empty: `{path}`.")
    return nodes


def build_node_index(nodes: List[NodeRecord]) -> Dict[int, int]:
    return {node.node_id: i for i, node in enumerate(nodes)}


def attach_geometry(nodes: List[NodeRecord], surface: vtk.vtkPolyData) -> None:
    point_count = surface.GetNumberOfPoints()
    for node in nodes:
        if node.node_id < 0 or node.node_id >= point_count:
            fail(f"Node id {node.node_id} is outside point range [0, {point_count - 1}].")
        point = surface.GetPoint(node.node_id)
        node.point = (point[0], point[1], point[2])


def classify_critical_type(node: NodeRecord) -> int:
    if node.parent_id == node.node_id or node.parent_id < 0:
        return 3
    if not node.children:
        return 0
    if len(node.children) > 1:
        return 2
    return 1


def build_tree_relations(nodes: List[NodeRecord], node_id_to_index: Dict[int, int]) -> None:
    root_count = 0
    for node in nodes:
        node.children.clear()
        node.incident_arc_ids.clear()
    for node in nodes:
        if node.parent_id == node.node_id or node.parent_id < 0:
            root_count += 1
            continue
        if node.parent_id not in node_id_to_index:
            fail(f"Parent id {node.parent_id} not found for node {node.node_id}.")
        nodes[node_id_to_index[node.parent_id]].children.append(node.node_id)
    if root_count <= 0:
        fail("Expected at least one root in the tree forest.")
    for node in nodes:
        node.degree = len(node.children) + (0 if node.parent_id == node.node_id or node.parent_id < 0 else 1)
        node.critical_type = classify_critical_type(node)


def build_arcs(nodes: List[NodeRecord], node_id_to_index: Dict[int, int]) -> List[ArcRecord]:
    arcs: List[ArcRecord] = []
    for node in nodes:
        if node.parent_id == node.node_id or node.parent_id < 0:
            continue
        parent = nodes[node_id_to_index[node.parent_id]]
        arc = ArcRecord(
            arc_id=len(arcs),
            down_node_id=node.node_id,
            up_node_id=node.parent_id,
            region_size=1.0,
            region_span=abs(parent.scalar - node.scalar),
        )
        arcs.append(arc)
        node.incident_arc_ids.append(arc.arc_id)
        parent.incident_arc_ids.append(arc.arc_id)
        node.region_span = arc.region_span
    return arcs


def other_node_id(arc: ArcRecord, node_id: int) -> int:
    if arc.down_node_id == node_id:
        return arc.up_node_id
    if arc.up_node_id == node_id:
        return arc.down_node_id
    fail("Arc/node mismatch.")


def orient_branch(node_ids: List[int], arc_ids: List[int], node_id_to_index: Dict[int, int], nodes: List[NodeRecord], tree_type: int) -> None:
    if len(node_ids) < 2:
        return
    first_scalar = nodes[node_id_to_index[node_ids[0]]].scalar
    last_scalar = nodes[node_id_to_index[node_ids[-1]]].scalar
    should_reverse = (tree_type == 0 and first_scalar > last_scalar) or (tree_type != 0 and first_scalar < last_scalar)
    if should_reverse:
        node_ids.reverse()
        arc_ids.reverse()


def extract_branches(nodes: List[NodeRecord], arcs: List[ArcRecord], node_id_to_index: Dict[int, int], tree_type: int) -> List[BranchRecord]:
    branches: List[BranchRecord] = []
    arc_visited = [False] * len(arcs)

    def is_endpoint(node_id: int) -> bool:
        return len(nodes[node_id_to_index[node_id]].incident_arc_ids) != 2

    branch_id = 0
    for node in nodes:
        if not is_endpoint(node.node_id):
            continue
        for arc_id in node.incident_arc_ids:
            if arc_visited[arc_id]:
                continue
            node_ids = [node.node_id]
            arc_ids: List[int] = []
            current_node_id = node.node_id
            current_arc_id = arc_id
            while True:
                arc_visited[current_arc_id] = True
                arc_ids.append(current_arc_id)
                next_node_id = other_node_id(arcs[current_arc_id], current_node_id)
                node_ids.append(next_node_id)
                if is_endpoint(next_node_id):
                    break
                next_node = nodes[node_id_to_index[next_node_id]]
                next_arc_id = next((candidate for candidate in next_node.incident_arc_ids if candidate != current_arc_id), -1)
                if next_arc_id < 0:
                    fail("Failed to continue branch traversal.")
                current_node_id = next_node_id
                current_arc_id = next_arc_id
            orient_branch(node_ids, arc_ids, node_id_to_index, nodes, tree_type)
            branches.append(BranchRecord(branch_id=branch_id, node_ids=node_ids, arc_ids=arc_ids))
            branch_id += 1
    if not all(arc_visited):
        fail("Some arcs were not assigned to branches.")
    return branches


def build_cell_graph(surface: vtk.vtkPolyData) -> Tuple[List[List[int]], List[Tuple[float, float, float]], List[List[int]]]:
    num_cells = surface.GetNumberOfCells()
    point_to_cells: Dict[int, List[int]] = defaultdict(list)
    cell_points: List[List[int]] = []
    cell_centers: List[Tuple[float, float, float]] = []

    for cell_id in range(num_cells):
        cell = surface.GetCell(cell_id)
        pts = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        cell_points.append(pts)
        center = [0.0, 0.0, 0.0]
        for pt_id in pts:
            pt = surface.GetPoint(pt_id)
            center[0] += pt[0]
            center[1] += pt[1]
            center[2] += pt[2]
            point_to_cells[pt_id].append(cell_id)
        inv = 1.0 / max(len(pts), 1)
        cell_centers.append((center[0] * inv, center[1] * inv, center[2] * inv))

    adjacency = [set() for _ in range(num_cells)]
    for shared_cells in point_to_cells.values():
        for a in shared_cells:
            for b in shared_cells:
                if a != b:
                    adjacency[a].add(b)
    return [sorted(list(n)) for n in adjacency], cell_centers, cell_points


def find_crossing_cells(surface: vtk.vtkPolyData, scalar_array: vtk.vtkDataArray, iso_value: float) -> List[int]:
    crossing: List[int] = []
    for cell_id in range(surface.GetNumberOfCells()):
        cell = surface.GetCell(cell_id)
        values = [scalar_array.GetComponent(cell.GetPointId(i), 0) for i in range(cell.GetNumberOfPoints())]
        if iso_value >= min(values) and iso_value <= max(values):
            crossing.append(cell_id)
    if crossing:
        return crossing
    best_diff = float("inf")
    best_cells: List[int] = []
    for cell_id in range(surface.GetNumberOfCells()):
        cell = surface.GetCell(cell_id)
        values = [scalar_array.GetComponent(cell.GetPointId(i), 0) for i in range(cell.GetNumberOfPoints())]
        diff = min(abs(v - iso_value) for v in values)
        if diff + 1e-8 < best_diff:
            best_diff = diff
            best_cells = [cell_id]
        elif abs(diff - best_diff) <= 1e-8:
            best_cells.append(cell_id)
    return best_cells


def keep_single_component(active_cells: List[int], adjacency: List[List[int]], cell_centers: List[Tuple[float, float, float]], node_point: Tuple[float, float, float]) -> List[int]:
    active_set = set(active_cells)
    visited = set()
    best_component: List[int] = []
    best_score = float("inf")

    for start in active_cells:
        if start in visited:
            continue
        comp: List[int] = []
        q = deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt in active_set and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        cx = sum(cell_centers[c][0] for c in comp) / len(comp)
        cy = sum(cell_centers[c][1] for c in comp) / len(comp)
        cz = sum(cell_centers[c][2] for c in comp) / len(comp)
        score = (cx - node_point[0]) ** 2 + (cy - node_point[1]) ** 2 + (cz - node_point[2]) ** 2
        if score < best_score:
            best_score = score
            best_component = comp
    return sorted(best_component)


def build_distance_field(num_cells: int, active_cells: List[int], adjacency: List[List[int]]) -> List[float]:
    dist = [float("inf")] * num_cells
    q = deque()
    for cell_id in active_cells:
        dist[cell_id] = 0.0
        q.append(cell_id)
    while q:
        cur = q.popleft()
        cur_dist = dist[cur]
        for nxt in adjacency[cur]:
            new_dist = cur_dist + 1.0
            if new_dist < dist[nxt]:
                dist[nxt] = new_dist
                q.append(nxt)
    return [d if math.isfinite(d) else float(num_cells) for d in dist]


def compute_centroid(active_cells: List[int], cell_centers: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if not active_cells:
        return (0.0, 0.0, 0.0)
    sx = sum(cell_centers[c][0] for c in active_cells)
    sy = sum(cell_centers[c][1] for c in active_cells)
    sz = sum(cell_centers[c][2] for c in active_cells)
    n = float(len(active_cells))
    return (sx / n, sy / n, sz / n)


def compute_bbox_diagonal(active_cells: List[int], cell_centers: List[Tuple[float, float, float]]) -> float:
    if not active_cells:
        return 0.0
    xs = [cell_centers[c][0] for c in active_cells]
    ys = [cell_centers[c][1] for c in active_cells]
    zs = [cell_centers[c][2] for c in active_cells]
    return math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2)


def build_descriptor(node: NodeRecord, surface: vtk.vtkPolyData, scalar_array: vtk.vtkDataArray, adjacency: List[List[int]], cell_centers: List[Tuple[float, float, float]], single_component_only: bool) -> SurfaceDescriptor:
    active_cells = find_crossing_cells(surface, scalar_array, node.scalar)
    if single_component_only:
        active_cells = keep_single_component(active_cells, adjacency, cell_centers, node.point)
    distance_field = build_distance_field(surface.GetNumberOfCells(), active_cells, adjacency)
    return SurfaceDescriptor(
        node_id=node.node_id,
        iso_value=node.scalar,
        active_cell_ids=active_cells,
        distance_field=distance_field,
        centroid=compute_centroid(active_cells, cell_centers),
        bbox_diagonal=compute_bbox_diagonal(active_cells, cell_centers),
    )


def compute_histogram(field: List[float], num_bins: int, min_val: float, max_val: float) -> List[float]:
    hist = [0.0] * num_bins
    value_range = max_val - min_val
    if value_range < 1e-8:
        return hist
    for value in field:
        bin_id = int((value - min_val) / value_range * (num_bins - 1))
        bin_id = max(0, min(num_bins - 1, bin_id))
        hist[bin_id] += 1.0
    total = sum(hist)
    return [v / total for v in hist] if total > 0 else hist


def compute_entropy(hist: List[float]) -> float:
    entropy = 0.0
    for p in hist:
        if p > 1e-10:
            entropy -= p * math.log2(p)
    return entropy


def compute_mutual_information(field_a: List[float], field_b: List[float], num_bins: int, min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    joint = [0.0] * (num_bins * num_bins)
    range_a = max_a - min_a
    range_b = max_b - min_b
    if range_a < 1e-8 or range_b < 1e-8:
        return 0.0
    for va, vb in zip(field_a, field_b):
        ba = max(0, min(num_bins - 1, int((va - min_a) / range_a * (num_bins - 1))))
        bb = max(0, min(num_bins - 1, int((vb - min_b) / range_b * (num_bins - 1))))
        joint[ba * num_bins + bb] += 1.0
    total = sum(joint)
    if total <= 0:
        return 0.0
    joint = [v / total for v in joint]
    marginal_a = [0.0] * num_bins
    marginal_b = [0.0] * num_bins
    for a in range(num_bins):
        for b in range(num_bins):
            marginal_a[a] += joint[a * num_bins + b]
            marginal_b[b] += joint[a * num_bins + b]
    mi = 0.0
    for a in range(num_bins):
        for b in range(num_bins):
            pab = joint[a * num_bins + b]
            pa = marginal_a[a]
            pb = marginal_b[b]
            if pab > 1e-10 and pa > 1e-10 and pb > 1e-10:
                mi += pab * math.log2(pab / (pa * pb))
    return mi


def compute_dist_ism(desc_a: SurfaceDescriptor, desc_b: SurfaceDescriptor, num_bins: int) -> float:
    min_a, max_a = min(desc_a.distance_field), max(desc_a.distance_field)
    min_b, max_b = min(desc_b.distance_field), max(desc_b.distance_field)
    if max_a - min_a < 1e-8 or max_b - min_b < 1e-8:
        return 0.0
    mi = compute_mutual_information(desc_a.distance_field, desc_b.distance_field, num_bins, min_a, max_a, min_b, max_b)
    hist_a = compute_histogram(desc_a.distance_field, num_bins, min_a, max_a)
    hist_b = compute_histogram(desc_b.distance_field, num_bins, min_b, max_b)
    h_a = compute_entropy(hist_a)
    h_b = compute_entropy(hist_b)
    if h_a + h_b <= 1e-10:
        return 0.0
    nmi = 2.0 * mi / (h_a + h_b)
    return 1.0 - nmi


def compute_centroid_distance(a: SurfaceDescriptor, b: SurfaceDescriptor) -> float:
    return math.sqrt(sum((a.centroid[i] - b.centroid[i]) ** 2 for i in range(3)))


def compute_bbox_relative_difference(a: SurfaceDescriptor, b: SurfaceDescriptor) -> float:
    denom = max(a.bbox_diagonal, 1e-8)
    return abs(a.bbox_diagonal - b.bbox_diagonal) / denom


def simplify_branches(nodes: List[NodeRecord], arcs: List[ArcRecord], branches: List[BranchRecord], node_id_to_index: Dict[int, int], surface: vtk.vtkPolyData, scalar_array: vtk.vtkDataArray, adjacency: List[List[int]], cell_centers: List[Tuple[float, float, float]], options: ProgramOptions) -> SimplificationResult:
    descriptor_cache: Dict[int, SurfaceDescriptor] = {}

    def get_descriptor(node_id: int) -> SurfaceDescriptor:
        if node_id not in descriptor_cache:
            node = nodes[node_id_to_index[node_id]]
            descriptor_cache[node_id] = build_descriptor(node, surface, scalar_array, adjacency, cell_centers, options.single_component_only)
        return descriptor_cache[node_id]

    simplified_branches = [BranchRecord(branch_id=b.branch_id, node_ids=list(b.node_ids), arc_ids=list(b.arc_ids), keep_indices=[]) for b in branches]
    distance_records: List[DistanceRecord] = []

    for branch in simplified_branches:
        if not branch.node_ids:
            continue
        branch.keep_indices = [0]
        ref_index = 0
        for i in range(1, len(branch.node_ids) - 1):
            ref_node_id = branch.node_ids[ref_index]
            candidate_node_id = branch.node_ids[i]
            ref_desc = get_descriptor(ref_node_id)
            cand_desc = get_descriptor(candidate_node_id)
            dist_ism = compute_dist_ism(ref_desc, cand_desc, options.histogram_bins)
            dist_pos = compute_centroid_distance(ref_desc, cand_desc)
            dist_bbox = compute_bbox_relative_difference(ref_desc, cand_desc)
            keep = dist_ism >= options.epsilon_ism or dist_pos >= options.epsilon_pos or dist_bbox >= options.epsilon_bbox
            distance_records.append(
                DistanceRecord(
                    branch_id=branch.branch_id,
                    ref_node_id=ref_node_id,
                    candidate_node_id=candidate_node_id,
                    dist_ism=dist_ism,
                    dist_pos=dist_pos,
                    dist_bbox=dist_bbox,
                    kept=1 if keep else 0,
                )
            )
            if keep:
                branch.keep_indices.append(i)
                ref_index = i
        last_index = len(branch.node_ids) - 1
        if branch.keep_indices[-1] != last_index:
            branch.keep_indices.append(last_index)

    kept = set()
    for branch in simplified_branches:
        for keep_index in branch.keep_indices:
            kept.add(branch.node_ids[keep_index])
    kept_node_ids = sorted(kept, key=lambda nid: nodes[node_id_to_index[nid]].scalar)
    node_keep_mask = {node.node_id: (1 if node.node_id in kept else 0) for node in nodes}

    new_arcs: List[NewArcRecord] = []
    new_arc_id = 0
    for branch in simplified_branches:
        for i in range(1, len(branch.keep_indices)):
            start_index = branch.keep_indices[i - 1]
            end_index = branch.keep_indices[i]
            down_node_id = branch.node_ids[start_index]
            up_node_id = branch.node_ids[end_index]
            region_span = abs(nodes[node_id_to_index[up_node_id]].scalar - nodes[node_id_to_index[down_node_id]].scalar)
            merged_arc_ids = [branch.arc_ids[k] for k in range(start_index, end_index)]
            region_size = float(len(merged_arc_ids))
            new_arcs.append(
                NewArcRecord(
                    new_arc_id=new_arc_id,
                    down_node_id=down_node_id,
                    up_node_id=up_node_id,
                    region_size=region_size,
                    region_span=region_span,
                    merged_arc_ids=merged_arc_ids,
                )
            )
            new_arc_id += 1

    return SimplificationResult(
        tree=SimplifiedTree(kept_node_ids=kept_node_ids, new_arcs=new_arcs, node_keep_mask=node_keep_mask),
        distances=distance_records,
    )


def build_simplified_node_polydata(tree: SimplifiedTree, nodes: List[NodeRecord], node_id_to_index: Dict[int, int]) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    node_id_arr = vtk.vtkIntArray(); node_id_arr.SetName("NodeId")
    scalar_arr = vtk.vtkDoubleArray(); scalar_arr.SetName("Scalar")
    critical_arr = vtk.vtkIntArray(); critical_arr.SetName("CriticalType")
    kept_arr = vtk.vtkIntArray(); kept_arr.SetName("IsKept")
    for node_id in tree.kept_node_ids:
        node = nodes[node_id_to_index[node_id]]
        pid = points.InsertNextPoint(node.point)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
        node_id_arr.InsertNextValue(node.node_id)
        scalar_arr.InsertNextValue(node.scalar)
        critical_arr.InsertNextValue(node.critical_type)
        kept_arr.InsertNextValue(1)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)
    poly.GetPointData().AddArray(node_id_arr)
    poly.GetPointData().AddArray(scalar_arr)
    poly.GetPointData().AddArray(critical_arr)
    poly.GetPointData().AddArray(kept_arr)
    poly.GetPointData().SetScalars(scalar_arr)
    return poly


def build_simplified_arc_polydata(tree: SimplifiedTree, nodes: List[NodeRecord], node_id_to_index: Dict[int, int]) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    node_point_ids: Dict[int, int] = {}
    point_node_id_arr = vtk.vtkIntArray(); point_node_id_arr.SetName("NodeId")
    point_scalar_arr = vtk.vtkDoubleArray(); point_scalar_arr.SetName("Scalar")
    for node_id in tree.kept_node_ids:
        node = nodes[node_id_to_index[node_id]]
        node_point_ids[node_id] = points.InsertNextPoint(node.point)
        point_node_id_arr.InsertNextValue(node.node_id)
        point_scalar_arr.InsertNextValue(node.scalar)
    new_arc_id_arr = vtk.vtkIntArray(); new_arc_id_arr.SetName("NewArcId")
    down_arr = vtk.vtkIntArray(); down_arr.SetName("downNodeId")
    up_arr = vtk.vtkIntArray(); up_arr.SetName("upNodeId")
    region_size_arr = vtk.vtkDoubleArray(); region_size_arr.SetName("RegionSize")
    region_span_arr = vtk.vtkDoubleArray(); region_span_arr.SetName("RegionSpan")
    merged_count_arr = vtk.vtkIntArray(); merged_count_arr.SetName("MergedArcCount")
    for arc in tree.new_arcs:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, node_point_ids[arc.down_node_id])
        line.GetPointIds().SetId(1, node_point_ids[arc.up_node_id])
        lines.InsertNextCell(line)
        new_arc_id_arr.InsertNextValue(arc.new_arc_id)
        down_arr.InsertNextValue(arc.down_node_id)
        up_arr.InsertNextValue(arc.up_node_id)
        region_size_arr.InsertNextValue(arc.region_size)
        region_span_arr.InsertNextValue(arc.region_span)
        merged_count_arr.InsertNextValue(len(arc.merged_arc_ids))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    poly.GetPointData().AddArray(point_node_id_arr)
    poly.GetPointData().AddArray(point_scalar_arr)
    poly.GetCellData().AddArray(new_arc_id_arr)
    poly.GetCellData().AddArray(down_arr)
    poly.GetCellData().AddArray(up_arr)
    poly.GetCellData().AddArray(region_size_arr)
    poly.GetCellData().AddArray(region_span_arr)
    poly.GetCellData().AddArray(merged_count_arr)
    return poly


def build_node_keep_mask_polydata(nodes: List[NodeRecord], tree: SimplifiedTree) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    node_id_arr = vtk.vtkIntArray(); node_id_arr.SetName("NodeId")
    scalar_arr = vtk.vtkDoubleArray(); scalar_arr.SetName("Scalar")
    kept_arr = vtk.vtkIntArray(); kept_arr.SetName("IsKept")
    for node in nodes:
        pid = points.InsertNextPoint(node.point)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
        node_id_arr.InsertNextValue(node.node_id)
        scalar_arr.InsertNextValue(node.scalar)
        kept_arr.InsertNextValue(tree.node_keep_mask.get(node.node_id, 0))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)
    poly.GetPointData().AddArray(node_id_arr)
    poly.GetPointData().AddArray(scalar_arr)
    poly.GetPointData().AddArray(kept_arr)
    poly.GetPointData().SetScalars(kept_arr)
    return poly


def write_polydata(poly: vtk.vtkPolyData, path: Path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    if writer.Write() == 0:
        fail(f"Failed to write `{path}`.")


def write_stats(path: Path, options: ProgramOptions, scalar_array_name: str, nodes: List[NodeRecord], arcs: List[ArcRecord], branches: List[BranchRecord], result: SimplificationResult) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write(f"mode={options.mode_name}\n")
        out.write(f"tree={options.tree_path}\n")
        out.write(f"surface={options.surface_path}\n")
        out.write(f"scalar_array={scalar_array_name}\n")
        out.write(f"epsilon_ism={options.epsilon_ism}\n")
        out.write(f"epsilon_pos={options.epsilon_pos}\n")
        out.write(f"epsilon_bbox={options.epsilon_bbox}\n")
        out.write(f"hist_bins={options.histogram_bins}\n")
        out.write(f"original_nodes={len(nodes)}\n")
        out.write(f"kept_nodes={len(result.tree.kept_node_ids)}\n")
        out.write(f"original_arcs={len(arcs)}\n")
        out.write(f"simplified_arcs={len(result.tree.new_arcs)}\n")
        out.write(f"branches={len(branches)}\n\n")
        out.write("[branch_summary]\n")
        for branch in branches:
            kept_count = sum(result.tree.node_keep_mask.get(node_id, 0) for node_id in branch.node_ids)
            out.write(f"branch={branch.branch_id}, nodes={len(branch.node_ids)}, kept={kept_count}\n")
        out.write("\n[distances]\n")
        out.write("branchId,refNodeId,candidateNodeId,distIsm,distPos,distBbox,kept\n")
        for record in result.distances:
            out.write(f"{record.branch_id},{record.ref_node_id},{record.candidate_node_id},{record.dist_ism},{record.dist_pos},{record.dist_bbox},{record.kept}\n")


def run_surface_pipeline(options: ProgramOptions) -> None:
    options.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    surface = read_surface(options.surface_path)
    scalar_array = get_scalar_array(surface, options.scalar_array_name)
    scalar_array_name = scalar_array.GetName() or "<unnamed>"
    nodes = read_tree_txt(options.tree_path)
    node_id_to_index = build_node_index(nodes)
    attach_geometry(nodes, surface)
    build_tree_relations(nodes, node_id_to_index)
    arcs = build_arcs(nodes, node_id_to_index)
    branches = extract_branches(nodes, arcs, node_id_to_index, options.tree_type)
    adjacency, cell_centers, _ = build_cell_graph(surface)
    result = simplify_branches(nodes, arcs, branches, node_id_to_index, surface, scalar_array, adjacency, cell_centers, options)
    write_polydata(build_simplified_node_polydata(result.tree, nodes, node_id_to_index), Path(str(options.output_prefix) + "_nodes.vtp"))
    write_polydata(build_simplified_arc_polydata(result.tree, nodes, node_id_to_index), Path(str(options.output_prefix) + "_arcs.vtp"))
    write_polydata(build_node_keep_mask_polydata(nodes, result.tree), Path(str(options.output_prefix) + "_node_mask.vtp"))
    write_stats(Path(str(options.output_prefix) + "_stats.txt"), options, scalar_array_name, nodes, arcs, branches, result)
    print(f"Surface ISM finished. Kept {len(result.tree.kept_node_ids)} / {len(nodes)} nodes using scalar array `{scalar_array_name}`.")
