#include "MergeTreeIO.h"
#include "IsmMergeTreeSimplifier.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkVertex.h>
#include <vtkXMLUnstructuredGridWriter.h>

namespace {

vtkDataArray *requireArray(vtkFieldData *fieldData,
                           const std::string &name,
                           const char *owner) {
  if (fieldData == nullptr) {
    throw std::runtime_error(std::string("Missing ") + owner + ".");
  }
  auto *array = fieldData->GetArray(name.c_str());
  if (array == nullptr) {
    throw std::runtime_error("Missing required array `" + name + "` in " +
                             owner + ".");
  }
  return array;
}

double tuple1(vtkDataArray *array, vtkIdType id) {
  return array->GetComponent(id, 0);
}

} // namespace

std::vector<TreeNode> MergeTreeIO::readNodesFromVTU(vtkUnstructuredGrid *grid) {
  auto *pd = grid->GetPointData();
  auto *nodeIdArr = requireArray(pd, "NodeId", "node point data");
  auto *scalarArr = requireArray(pd, "Scalar", "node point data");
  auto *criticalTypeArr = requireArray(pd, "CriticalType", "node point data");
  auto *vertexIdArr = requireArray(pd, "VertexId", "node point data");
  auto *regionSizeArr = requireArray(pd, "RegionSize", "node point data");
  auto *regionSpanArr = requireArray(pd, "RegionSpan", "node point data");

  const vtkIdType nPts = grid->GetNumberOfPoints();
  std::vector<TreeNode> nodes(nPts);

  for (vtkIdType i = 0; i < nPts; i++) {
    nodes[i].nodeId = static_cast<int>(tuple1(nodeIdArr, i));
    nodes[i].scalar = tuple1(scalarArr, i);
    nodes[i].criticalType = static_cast<int>(tuple1(criticalTypeArr, i));
    nodes[i].vertexId = static_cast<int>(tuple1(vertexIdArr, i));
    nodes[i].regionSize = tuple1(regionSizeArr, i);
    nodes[i].regionSpan = tuple1(regionSpanArr, i);
    double pt[3];
    grid->GetPoint(i, pt);
    nodes[i].position = {pt[0], pt[1], pt[2]};
  }

  std::cout << "Read " << nodes.size() << " nodes from grid." << std::endl;
  return nodes;
}

std::vector<TreeArc> MergeTreeIO::readArcsFromVTU(vtkUnstructuredGrid *grid) {
  auto *cd = grid->GetCellData();
  auto *downNodeIdArr = requireArray(cd, "downNodeId", "arc cell data");
  auto *upNodeIdArr = requireArray(cd, "upNodeId", "arc cell data");
  auto *segmentationIdArr = requireArray(cd, "SegmentationId", "arc cell data");
  auto *regionSizeArr = requireArray(cd, "RegionSize", "arc cell data");
  auto *regionSpanArr = requireArray(cd, "RegionSpan", "arc cell data");

  const vtkIdType nCells = grid->GetNumberOfCells();
  std::vector<TreeArc> arcs(nCells);

  for (vtkIdType i = 0; i < nCells; i++) {
    arcs[i].arcId = static_cast<int>(i);
    arcs[i].downNodeId = static_cast<int>(tuple1(downNodeIdArr, i));
    arcs[i].upNodeId = static_cast<int>(tuple1(upNodeIdArr, i));
    arcs[i].segmentationId = static_cast<int>(tuple1(segmentationIdArr, i));
    arcs[i].regionSize = tuple1(regionSizeArr, i);
    arcs[i].regionSpan = tuple1(regionSpanArr, i);
  }

  std::cout << "Read " << arcs.size() << " arcs from grid." << std::endl;
  return arcs;
}

void MergeTreeIO::writeSimplifiedNodes(vtkUnstructuredGrid *originalGrid,
                                        const std::vector<int> &keptNodeIds,
                                        const std::string &path) {
  auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  grid->DeepCopy(originalGrid);

  // Add KeepNode array
  auto keepNodeArr = vtkSmartPointer<vtkIntArray>::New();
  keepNodeArr->SetName("KeepNode");
  keepNodeArr->SetNumberOfTuples(grid->GetNumberOfPoints());

  auto *nodeIdArr = grid->GetPointData()->GetArray("NodeId");
  if (!nodeIdArr) {
    std::cerr << "Warning: NodeId array not found in original grid" << std::endl;
    return;
  }

  // Create a set for quick lookup
  std::unordered_set<int> keptSet(keptNodeIds.begin(), keptNodeIds.end());

  for (vtkIdType i = 0; i < grid->GetNumberOfPoints(); i++) {
    int nodeId = static_cast<int>(tuple1(nodeIdArr, i));
    int keep = keptSet.count(nodeId) ? 1 : 0;
    keepNodeArr->SetValue(i, keep);
  }

  grid->GetPointData()->AddArray(keepNodeArr);
  writeGrid(grid, path);
}

void MergeTreeIO::writeGrid(vtkSmartPointer<vtkUnstructuredGrid> grid,
                             const std::string &path) {
  auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  writer->SetFileName(path.c_str());
  writer->SetInputData(grid);
  writer->Write();
}

void MergeTreeIO::writeStatsCSV(const std::string &path,
                                const std::vector<TopologicalChain> &chains,
                                const std::vector<int> &keptNodeIds,
                                const std::vector<double> &distToRef) {
  std::ofstream ofs(path);
  if (!ofs) {
    throw std::runtime_error("Unable to write stats to `" + path + "`.");
  }

  ofs << "=== ISM Merge Tree Simplification Statistics ===" << std::endl;
  ofs << std::endl;
  ofs << "Total chains: " << chains.size() << std::endl;
  ofs << "Total nodes kept: " << keptNodeIds.size() << std::endl;
  ofs << std::endl;

  ofs << "ChainId,NodeCount" << std::endl;
  for (size_t i = 0; i < chains.size(); i++) {
    ofs << i << "," << chains[i].nodeIds.size() << std::endl;
  }

  ofs.close();
}