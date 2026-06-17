#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkImageData.h>

namespace active_graph {

struct TreeNode {
  int nodeId;
  double scalar;
  int criticalType;
  int vertexId;
  double regionSize;
  double regionSpan;
  std::array<double, 3> position;
};

struct TreeArc {
  int arcId;
  int segmentationId;
  int downNodeId;
  int upNodeId;
  double regionSize;
  double regionSpan;
};

class MergeTreeIO {
public:
  int readTreeNodes(const std::string &filePath);
  int readTreeArcs(const std::string &filePath);
  int readSegmentation(const std::string &filePath);

  const std::map<int, TreeNode> &getNodes() const { return nodes_; }
  const std::map<int, TreeArc> &getArcs() const { return arcs_; }
  const vtkSmartPointer<vtkImageData> &getImageData() const { return imageData_; }

  int buildAdjacency();
  const std::vector<int> &getChildren(int nodeId) const;
  int getArcId(int downNodeId, int upNodeId) const;

private:
  std::map<int, TreeNode> nodes_;
  std::map<int, TreeArc> arcs_;
  vtkSmartPointer<vtkImageData> imageData_;
  std::map<int, std::vector<int>> adjacency_;
};

} // namespace active_graph