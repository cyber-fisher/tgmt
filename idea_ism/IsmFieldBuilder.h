#pragma once

#include <memory>
#include <vector>

#include <vtkImageData.h>
#include <vtkSmartPointer.h>

// Forward declarations
struct TreeNode;
struct TreeArc;
struct IsmDescriptor;

class IsmFieldBuilder {
public:
  static std::unique_ptr<IsmDescriptor> buildDescriptor(
      int nodeId,
      const std::vector<TreeNode> &nodes,
      const std::vector<TreeArc> &arcs,
      vtkSmartPointer<vtkImageData> imageData);

private:
  static std::vector<int> getLocalSegmentationIds(
      int nodeId,
      const std::vector<TreeNode> &nodes,
      const std::vector<TreeArc> &arcs);

  static void extractLocalBlock(
      vtkSmartPointer<vtkImageData> imageData,
      const std::vector<int> &segmentationIds,
      int paddingVoxels,
      std::array<int, 6> &extent,
      std::vector<float> &scalarValues,
      std::vector<int> &blockSegIds);

  static void buildBoundaryMask(
      const std::vector<float> &scalarValues,
      const std::array<int, 6> &extent,
      double scalarThreshold,
      std::vector<unsigned char> &boundaryMask);

  static void buildDistanceField(
      const std::vector<unsigned char> &boundaryMask,
      const std::array<int, 6> &extent,
      std::vector<float> &distanceField);

  static std::array<double, 3> computeCentroid(
      const std::vector<unsigned char> &boundaryMask,
      const std::array<int, 6> &extent);

  static double computeBboxDiagonal(const std::array<int, 6> &extent);
};