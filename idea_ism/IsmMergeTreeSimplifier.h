#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

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

struct TopologicalChain {
  std::vector<int> nodeIds;
  std::vector<int> arcIds;
};

struct LocalVolumeBlock {
  std::array<int, 6> extent; // xmin xmax ymin ymax zmin zmax
  std::vector<float> scalarValues;
  std::vector<int> segmentationIds;
  int dimX;
  int dimY;
  int dimZ;
};

struct IsmDescriptor {
  int nodeId;
  std::array<int, 6> extent;
  std::vector<float> distanceField;
  std::vector<unsigned char> boundaryMask;
  std::array<double, 3> centroid;
  double bboxDiagonal;
};

struct IsmSimplificationParams {
  std::string treeNodesPath;
  std::string treeArcsPath;
  std::string segmentationPath;
  std::string scalarFieldName{"v"};
  std::string outputPrefix{"ism_simplified"};
  double epsilonIsm{0.15};
  double epsilonPos{0.10};
  double epsilonBbox{0.10};
  int histogramBins{32};
  int paddingVoxels{3};
  int maxBlockEdge{64};
  bool writeDebugVolumes{false};
};

class IsmMergeTreeSimplifier {
public:
  IsmMergeTreeSimplifier();
  ~IsmMergeTreeSimplifier();

  int run(const IsmSimplificationParams &params);

private:
  int loadData(const IsmSimplificationParams &params);
  int buildTopologicalChains();
  int buildDescriptor(int nodeId);
  int simplify(const IsmSimplificationParams &params);
  int writeOutputs(const IsmSimplificationParams &params) const;

  // Helper methods
  bool isStructuralNode(const TreeNode &node) const;
  std::vector<int> getLocalSegmentationIds(int nodeId) const;
  LocalVolumeBlock extractLocalBlock(int nodeId) const;
  void buildBoundaryMask(const LocalVolumeBlock &block, int nodeId,
                         IsmDescriptor &desc) const;
  void buildDistanceField(const LocalVolumeBlock &block,
                          IsmDescriptor &desc) const;

  // Data members
  vtkSmartPointer<vtkUnstructuredGrid> nodeGrid_;
  vtkSmartPointer<vtkUnstructuredGrid> arcGrid_;
  vtkSmartPointer<vtkImageData> segmentationImage_;

  std::vector<TreeNode> nodes_;
  std::vector<TreeArc> arcs_;
  std::vector<TopologicalChain> chains_;

  // Simplification results
  std::vector<int> keptNodeIds_;
  std::vector<std::pair<int, int>> simplifiedArcs_;
  std::vector<double> distToRef_;
};