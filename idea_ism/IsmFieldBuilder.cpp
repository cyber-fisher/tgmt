#include "IsmFieldBuilder.h"
#include "IsmMergeTreeSimplifier.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <vtkDataArray.h>
#include <vtkPointData.h>

std::unique_ptr<IsmDescriptor> IsmFieldBuilder::buildDescriptor(
    int nodeId,
    const std::vector<TreeNode> &nodes,
    const std::vector<TreeArc> &arcs,
    vtkSmartPointer<vtkImageData> imageData) {

  auto desc = std::make_unique<IsmDescriptor>();
  desc->nodeId = nodeId;

  // Find node
  const TreeNode *node = nullptr;
  for (const auto &n : nodes) {
    if (n.nodeId == nodeId) {
      node = &n;
      break;
    }
  }
  if (!node) {
    std::cerr << "Node " << nodeId << " not found" << std::endl;
    return nullptr;
  }

  // Get local segmentation IDs
  auto segIds = getLocalSegmentationIds(nodeId, nodes, arcs);
  if (segIds.empty()) {
    std::cerr << "No segmentation IDs found for node " << nodeId << std::endl;
    return nullptr;
  }

  // Extract local block
  std::vector<float> scalarValues;
  std::vector<int> blockSegIds;
  extractLocalBlock(imageData, segIds, 3, desc->extent, scalarValues, blockSegIds);

  if (scalarValues.empty()) {
    std::cerr << "Empty local block for node " << nodeId << std::endl;
    return nullptr;
  }

  // Build boundary mask
  buildBoundaryMask(scalarValues, desc->extent, node->scalar, desc->boundaryMask);

  // Build distance field
  buildDistanceField(desc->boundaryMask, desc->extent, desc->distanceField);

  // Compute centroid
  desc->centroid = computeCentroid(desc->boundaryMask, desc->extent);

  // Compute bbox diagonal
  desc->bboxDiagonal = computeBboxDiagonal(desc->extent);

  return desc;
}

std::vector<int> IsmFieldBuilder::getLocalSegmentationIds(
    int nodeId,
    const std::vector<TreeNode> &nodes,
    const std::vector<TreeArc> &arcs) {

  std::unordered_set<int> segIdSet;

  // Find arcs incident to this node
  for (const auto &arc : arcs) {
    if (arc.downNodeId == nodeId || arc.upNodeId == nodeId) {
      if (arc.segmentationId >= 0) {
        segIdSet.insert(arc.segmentationId);
      }
    }
  }

  return std::vector<int>(segIdSet.begin(), segIdSet.end());
}

void IsmFieldBuilder::extractLocalBlock(
    vtkSmartPointer<vtkImageData> imageData,
    const std::vector<int> &segmentationIds,
    int paddingVoxels,
    std::array<int, 6> &extent,
    std::vector<float> &scalarValues,
    std::vector<int> &blockSegIds) {

  // Get segmentation array
  auto *segArray = imageData->GetPointData()->GetArray("SegmentationId");
  auto *scalarArray = imageData->GetPointData()->GetArray("v");

  if (!segArray || !scalarArray) {
    std::cerr << "Missing required arrays in image data" << std::endl;
    return;
  }

  // Create set for quick lookup
  std::unordered_set<int> segIdSet(segmentationIds.begin(), segmentationIds.end());

  // Find bounding box of matching segmentation IDs
  int dims[3];
  imageData->GetDimensions(dims);

  int minCoord[3] = {dims[0], dims[1], dims[2]};
  int maxCoord[3] = {0, 0, 0};
  bool found = false;

  for (int z = 0; z < dims[2]; z++) {
    for (int y = 0; y < dims[1]; y++) {
      for (int x = 0; x < dims[0]; x++) {
        vtkIdType id = z * dims[1] * dims[0] + y * dims[0] + x;
        int segId = static_cast<int>(segArray->GetComponent(id, 0));

        if (segIdSet.count(segId)) {
          minCoord[0] = std::min(minCoord[0], x);
          minCoord[1] = std::min(minCoord[1], y);
          minCoord[2] = std::min(minCoord[2], z);
          maxCoord[0] = std::max(maxCoord[0], x);
          maxCoord[1] = std::max(maxCoord[1], y);
          maxCoord[2] = std::max(maxCoord[2], z);
          found = true;
        }
      }
    }
  }

  if (!found) {
    std::cerr << "No matching segmentation IDs found in image" << std::endl;
    return;
  }

  // Apply padding
  for (int d = 0; d < 3; d++) {
    minCoord[d] = std::max(0, minCoord[d] - paddingVoxels);
    maxCoord[d] = std::min(dims[d] - 1, maxCoord[d] + paddingVoxels);
  }

  extent = {minCoord[0], maxCoord[0], minCoord[1], maxCoord[1], minCoord[2], maxCoord[2]};

  // Extract block
  int blockDims[3] = {maxCoord[0] - minCoord[0] + 1,
                      maxCoord[1] - minCoord[1] + 1,
                      maxCoord[2] - minCoord[2] + 1};

  scalarValues.resize(blockDims[0] * blockDims[1] * blockDims[2]);
  blockSegIds.resize(blockDims[0] * blockDims[1] * blockDims[2]);

  for (int z = 0; z < blockDims[2]; z++) {
    for (int y = 0; y < blockDims[1]; y++) {
      for (int x = 0; x < blockDims[0]; x++) {
        int srcX = minCoord[0] + x;
        int srcY = minCoord[1] + y;
        int srcZ = minCoord[2] + z;

        vtkIdType srcId = srcZ * dims[1] * dims[0] + srcY * dims[0] + srcX;
        vtkIdType dstId = z * blockDims[1] * blockDims[0] + y * blockDims[0] + x;

        scalarValues[dstId] = static_cast<float>(scalarArray->GetComponent(srcId, 0));
        blockSegIds[dstId] = static_cast<int>(segArray->GetComponent(srcId, 0));
      }
    }
  }
}

void IsmFieldBuilder::buildBoundaryMask(
    const std::vector<float> &scalarValues,
    const std::array<int, 6> &extent,
    double scalarThreshold,
    std::vector<unsigned char> &boundaryMask) {

  int dimX = extent[1] - extent[0] + 1;
  int dimY = extent[3] - extent[2] + 1;
  int dimZ = extent[5] - extent[4] + 1;

  boundaryMask.resize(scalarValues.size(), 0);

  // Create binary inside/outside field
  std::vector<unsigned char> inside(scalarValues.size());
  for (size_t i = 0; i < scalarValues.size(); i++) {
    inside[i] = (scalarValues[i] <= scalarThreshold) ? 1 : 0;
  }

  // Find boundary voxels (6-connected)
  for (int z = 0; z < dimZ; z++) {
    for (int y = 0; y < dimY; y++) {
      for (int x = 0; x < dimX; x++) {
        int idx = z * dimY * dimX + y * dimX + x;
        unsigned char centerVal = inside[idx];

        // Check 6 neighbors
        bool isBoundary = false;
        if (x > 0 && inside[idx - 1] != centerVal) isBoundary = true;
        if (x < dimX - 1 && inside[idx + 1] != centerVal) isBoundary = true;
        if (y > 0 && inside[idx - dimX] != centerVal) isBoundary = true;
        if (y < dimY - 1 && inside[idx + dimX] != centerVal) isBoundary = true;
        if (z > 0 && inside[idx - dimY * dimX] != centerVal) isBoundary = true;
        if (z < dimZ - 1 && inside[idx + dimY * dimX] != centerVal) isBoundary = true;

        boundaryMask[idx] = isBoundary ? 1 : 0;
      }
    }
  }
}

void IsmFieldBuilder::buildDistanceField(
    const std::vector<unsigned char> &boundaryMask,
    const std::array<int, 6> &extent,
    std::vector<float> &distanceField) {

  int dimX = extent[1] - extent[0] + 1;
  int dimY = extent[3] - extent[2] + 1;
  int dimZ = extent[5] - extent[4] + 1;

  distanceField.resize(boundaryMask.size(), std::numeric_limits<float>::max());

  // Initialize distance field: boundary voxels have distance 0
  for (size_t i = 0; i < boundaryMask.size(); i++) {
    if (boundaryMask[i]) {
      distanceField[i] = 0.0f;
    }
  }

  // Simple BFS distance transform
  std::queue<std::pair<int, std::array<int, 3>>> bfsQueue;

  // Initialize queue with boundary voxels
  for (int z = 0; z < dimZ; z++) {
    for (int y = 0; y < dimY; y++) {
      for (int x = 0; x < dimX; x++) {
        int idx = z * dimY * dimX + y * dimX + x;
        if (boundaryMask[idx]) {
          bfsQueue.push({idx, {x, y, z}});
        }
      }
    }
  }

  // BFS to compute distances
  const int dx[] = {1, -1, 0, 0, 0, 0};
  const int dy[] = {0, 0, 1, -1, 0, 0};
  const int dz[] = {0, 0, 0, 0, 1, -1};

  while (!bfsQueue.empty()) {
    auto [idx, coord] = bfsQueue.front();
    bfsQueue.pop();

    float currentDist = distanceField[idx];

    for (int d = 0; d < 6; d++) {
      int nx = coord[0] + dx[d];
      int ny = coord[1] + dy[d];
      int nz = coord[2] + dz[d];

      if (nx >= 0 && nx < dimX && ny >= 0 && ny < dimY && nz >= 0 && nz < dimZ) {
        int nidx = nz * dimY * dimX + ny * dimX + nx;
        float newDist = currentDist + 1.0f;

        if (newDist < distanceField[nidx]) {
          distanceField[nidx] = newDist;
          bfsQueue.push({nidx, {nx, ny, nz}});
        }
      }
    }
  }
}

std::array<double, 3> IsmFieldBuilder::computeCentroid(
    const std::vector<unsigned char> &boundaryMask,
    const std::array<int, 6> &extent) {

  int dimX = extent[1] - extent[0] + 1;
  int dimY = extent[3] - extent[2] + 1;
  int dimZ = extent[5] - extent[4] + 1;

  double sumX = 0, sumY = 0, sumZ = 0;
  int count = 0;

  for (int z = 0; z < dimZ; z++) {
    for (int y = 0; y < dimY; y++) {
      for (int x = 0; x < dimX; x++) {
        int idx = z * dimY * dimX + y * dimX + x;
        if (boundaryMask[idx]) {
          sumX += (extent[0] + x);
          sumY += (extent[2] + y);
          sumZ += (extent[4] + z);
          count++;
        }
      }
    }
  }

  if (count == 0) {
    return {0.0, 0.0, 0.0};
  }

  return {sumX / count, sumY / count, sumZ / count};
}

double IsmFieldBuilder::computeBboxDiagonal(const std::array<int, 6> &extent) {
  double dx = extent[1] - extent[0];
  double dy = extent[3] - extent[2];
  double dz = extent[5] - extent[4];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}