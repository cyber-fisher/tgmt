#include "MergeTreeIO.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkIdList.h>
#include <vtkImageData.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <iostream>
#include <stdexcept>

namespace active_graph {

int MergeTreeIO::readTreeNodes(const std::string &filePath) {
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(filePath.c_str());
  reader->Update();

  auto *ugrid = reader->GetOutput();
  if (!ugrid) {
    std::cerr << "ERROR: Failed to read tree nodes from " << filePath << std::endl;
    return -1;
  }

  vtkIdType numPoints = ugrid->GetNumberOfPoints();
  std::cout << "Read " << numPoints << " tree nodes from " << filePath << std::endl;

  auto *nodeIdArray = ugrid->GetPointData()->GetArray("NodeId");
  auto *scalarArray = ugrid->GetPointData()->GetArray("Scalar");
  auto *criticalTypeArray = ugrid->GetPointData()->GetArray("CriticalType");
  auto *vertexIdArray = ugrid->GetPointData()->GetArray("VertexId");
  auto *regionSizeArray = ugrid->GetPointData()->GetArray("RegionSize");
  auto *regionSpanArray = ugrid->GetPointData()->GetArray("RegionSpan");

  if (!nodeIdArray || !scalarArray) {
    std::cerr << "ERROR: Missing required point data arrays" << std::endl;
    return -1;
  }

  nodes_.clear();
  for (vtkIdType i = 0; i < numPoints; ++i) {
    TreeNode node;
    node.nodeId = static_cast<int>(nodeIdArray->GetComponent(i, 0));
    node.scalar = scalarArray->GetComponent(i, 0);
    node.criticalType = criticalTypeArray ? static_cast<int>(criticalTypeArray->GetComponent(i, 0)) : 0;
    node.vertexId = vertexIdArray ? static_cast<int>(vertexIdArray->GetComponent(i, 0)) : -1;
    node.regionSize = regionSizeArray ? regionSizeArray->GetComponent(i, 0) : 0.0;
    node.regionSpan = regionSpanArray ? regionSpanArray->GetComponent(i, 0) : 0.0;
    ugrid->GetPoint(i, node.position.data());
    nodes_[node.nodeId] = node;
  }

  std::cout << "Parsed " << nodes_.size() << " tree nodes" << std::endl;
  return 0;
}

int MergeTreeIO::readTreeArcs(const std::string &filePath) {
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(filePath.c_str());
  reader->Update();

  auto *ugrid = reader->GetOutput();
  if (!ugrid) {
    std::cerr << "ERROR: Failed to read tree arcs from " << filePath << std::endl;
    return -1;
  }

  vtkIdType numCells = ugrid->GetNumberOfCells();
  std::cout << "Read " << numCells << " tree arcs from " << filePath << std::endl;

  auto *segIdArray = ugrid->GetCellData()->GetArray("SegmentationId");
  auto *downNodeIdArray = ugrid->GetCellData()->GetArray("downNodeId");
  auto *upNodeIdArray = ugrid->GetCellData()->GetArray("upNodeId");
  auto *regionSizeArray = ugrid->GetCellData()->GetArray("RegionSize");
  auto *regionSpanArray = ugrid->GetCellData()->GetArray("RegionSpan");

  if (!downNodeIdArray || !upNodeIdArray) {
    std::cerr << "ERROR: Missing required cell data arrays" << std::endl;
    return -1;
  }

  arcs_.clear();
  for (vtkIdType i = 0; i < numCells; ++i) {
    TreeArc arc;
    arc.arcId = static_cast<int>(i);
    arc.segmentationId = segIdArray ? static_cast<int>(segIdArray->GetComponent(i, 0)) : -1;
    arc.downNodeId = static_cast<int>(downNodeIdArray->GetComponent(i, 0));
    arc.upNodeId = static_cast<int>(upNodeIdArray->GetComponent(i, 0));
    arc.regionSize = regionSizeArray ? regionSizeArray->GetComponent(i, 0) : 0.0;
    arc.regionSpan = regionSpanArray ? regionSpanArray->GetComponent(i, 0) : 0.0;
    arcs_[arc.arcId] = arc;
  }

  std::cout << "Parsed " << arcs_.size() << " tree arcs" << std::endl;
  return 0;
}

int MergeTreeIO::readSegmentation(const std::string &filePath) {
  auto reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
  reader->SetFileName(filePath.c_str());
  reader->Update();

  imageData_ = reader->GetOutput();
  if (!imageData_) {
    std::cerr << "ERROR: Failed to read segmentation from " << filePath << std::endl;
    return -1;
  }

  int dims[3];
  imageData_->GetDimensions(dims);
  std::cout << "Read image data with dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;

  auto *scalarArray = imageData_->GetPointData()->GetArray("v");
  if (!scalarArray) {
    std::cerr << "WARNING: Scalar field 'v' not found in image data" << std::endl;
  }

  auto *segIdArray = imageData_->GetPointData()->GetArray("SegmentationId");
  if (!segIdArray) {
    std::cerr << "WARNING: SegmentationId not found in image data" << std::endl;
  }

  return 0;
}

int MergeTreeIO::buildAdjacency() {
  adjacency_.clear();

  for (const auto &[nodeId, node] : nodes_) {
    adjacency_[nodeId] = std::vector<int>();
  }

  for (const auto &[arcId, arc] : arcs_) {
    adjacency_[arc.downNodeId].push_back(arc.upNodeId);
  }

  std::cout << "Built adjacency list for " << adjacency_.size() << " nodes" << std::endl;
  return 0;
}

const std::vector<int> &MergeTreeIO::getChildren(int nodeId) const {
  static const std::vector<int> empty;
  auto it = adjacency_.find(nodeId);
  if (it != adjacency_.end()) {
    return it->second;
  }
  return empty;
}

int MergeTreeIO::getArcId(int downNodeId, int upNodeId) const {
  for (const auto &[arcId, arc] : arcs_) {
    if (arc.downNodeId == downNodeId && arc.upNodeId == upNodeId) {
      return arcId;
    }
  }
  return -1;
}

} // namespace active_graph