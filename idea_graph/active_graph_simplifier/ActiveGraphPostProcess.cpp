#include "ActiveGraphPostProcess.h"
#include "MergeTreeIO.h"

#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace active_graph {

ActiveGraphPostProcess::ActiveGraphPostProcess(MergeTreeIO &io) : io_(io) {}

void ActiveGraphPostProcess::setScalarFieldName(const std::string &scalarFieldName) {
  scalarFieldName_ = scalarFieldName.empty() ? "v" : scalarFieldName;
  scalarRangeReady_ = false;
}

void ActiveGraphPostProcess::setSimilarityMode(SimilarityMode mode) {
  similarityMode_ = mode;
}

void ActiveGraphPostProcess::setGnnOptions(int hiddenDim, unsigned int seed) {
  gnnHiddenDim_ = std::max(1, hiddenDim);
  gnnSeed_ = seed;
}

std::string ActiveGraphPostProcess::getGraphDistanceName() const {
  return similarityMode_ == SimilarityMode::GNN ? "distGNN" : "distJ";
}

bool ActiveGraphPostProcess::ensureScalarRange() const {
  if (scalarRangeReady_) {
    return true;
  }

  auto imageData = io_.getImageData();
  if (!imageData) {
    return false;
  }

  auto *scalarArray = imageData->GetPointData()->GetArray(scalarFieldName_.c_str());
  if (!scalarArray) {
    return false;
  }

  scalarMin_ = std::numeric_limits<double>::max();
  scalarMax_ = -std::numeric_limits<double>::max();
  const vtkIdType pointCount = imageData->GetNumberOfPoints();
  for (vtkIdType pid = 0; pid < pointCount; ++pid) {
    const double value = scalarArray->GetComponent(pid, 0);
    scalarMin_ = std::min(scalarMin_, value);
    scalarMax_ = std::max(scalarMax_, value);
  }
  scalarRangeReady_ = true;
  return true;
}

bool ActiveGraphPostProcess::ensureSegmentationBboxes() const {
  if (segmentationBboxesReady_) {
    return true;
  }

  auto imageData = io_.getImageData();
  if (!imageData) {
    return false;
  }

  auto *segIdArray = imageData->GetPointData()->GetArray("SegmentationId");
  if (!segIdArray) {
    return false;
  }

  int dims[3];
  imageData->GetDimensions(dims);
  auto getPointId = [dims](int i, int j, int k) -> vtkIdType {
    return static_cast<vtkIdType>(k * dims[1] * dims[0] + j * dims[0] + i);
  };

  segmentationBboxes_.clear();
  for (int k = 0; k < dims[2]; ++k) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int i = 0; i < dims[0]; ++i) {
        const vtkIdType pid = getPointId(i, j, k);
        const int seg = static_cast<int>(segIdArray->GetComponent(pid, 0));
        auto it = segmentationBboxes_.find(seg);
        if (it == segmentationBboxes_.end()) {
          segmentationBboxes_[seg] = {i, i, j, j, k, k};
        } else {
          it->second[0] = std::min(it->second[0], i);
          it->second[1] = std::max(it->second[1], i);
          it->second[2] = std::min(it->second[2], j);
          it->second[3] = std::max(it->second[3], j);
          it->second[4] = std::min(it->second[4], k);
          it->second[5] = std::max(it->second[5], k);
        }
      }
    }
  }

  segmentationBboxesReady_ = true;
  return true;
}

bool ActiveGraphPostProcess::isCriticalNode(int nodeId) {
  const auto &nodes = io_.getNodes();
  auto it = nodes.find(nodeId);
  if (it == nodes.end()) {
    return false;
  }
  return it->second.criticalType != 0;
}

void ActiveGraphPostProcess::sortGridEdge(GridEdge &edge) {
  if (edge.first > edge.second) {
    std::swap(edge.first, edge.second);
  }
}

std::set<int> ActiveGraphPostProcess::getLocalSegmentationIds(const TopologicalChain &chain,
                                                              size_t nodeIndex) const {
  std::set<int> segIds;
  const auto &arcs = io_.getArcs();

  auto appendArcSegmentation = [&](size_t arcIndex) {
    if (arcIndex >= chain.arcIds.size()) {
      return;
    }
    auto arcIt = arcs.find(chain.arcIds[arcIndex]);
    if (arcIt != arcs.end() && arcIt->second.segmentationId >= 0) {
      segIds.insert(arcIt->second.segmentationId);
    }
  };

  if (chain.nodeIds.size() < 2) {
    return segIds;
  }

  if (nodeIndex == 0) {
    appendArcSegmentation(0);
    return segIds;
  }

  if (nodeIndex + 1 == chain.nodeIds.size()) {
    appendArcSegmentation(chain.arcIds.size() - 1);
    return segIds;
  }

  appendArcSegmentation(nodeIndex - 1);
  appendArcSegmentation(nodeIndex);
  return segIds;
}

int ActiveGraphPostProcess::extractTopologicalChains() {
  chains_.clear();

  const auto &nodes = io_.getNodes();
  const auto &arcs = io_.getArcs();

  std::map<int, int> inDegree;
  std::map<int, int> outDegree;
  std::map<int, std::vector<int>> upArcs;

  for (const auto &[arcId, arc] : arcs) {
    inDegree[arc.upNodeId]++;
    outDegree[arc.downNodeId]++;
    upArcs[arc.upNodeId].push_back(arc.downNodeId);
  }

  std::vector<int> criticalNodes;
  for (const auto &[nodeId, node] : nodes) {
    if (isCriticalNode(nodeId) || inDegree[nodeId] == 0 || outDegree[nodeId] == 0 ||
        outDegree[nodeId] > 1) {
      criticalNodes.push_back(nodeId);
    }
  }

  std::cout << "Found " << criticalNodes.size() << " critical/structural nodes" << std::endl;

  int chainId = 0;
  for (int startNodeId : criticalNodes) {
    const auto &children = io_.getChildren(startNodeId);

    for (int nextNodeId : children) {
      TopologicalChain chain;
      chain.chainId = chainId++;
      chain.nodeIds.push_back(startNodeId);

      int currentId = nextNodeId;
      int prevId = startNodeId;
      while (true) {
        chain.nodeIds.push_back(currentId);
        
        int arcId = io_.getArcId(prevId, currentId);
        if (arcId >= 0) {
          chain.arcIds.push_back(arcId);
        }

        if (isCriticalNode(currentId) || io_.getChildren(currentId).size() != 1 ||
            inDegree[currentId] != 1) {
          break;
        }

        prevId = currentId;
        currentId = io_.getChildren(currentId)[0];
      }

      if (chain.nodeIds.size() >= 2) {
        std::sort(chain.nodeIds.begin(), chain.nodeIds.end(),
                  [&nodes](int a, int b) { return nodes.at(a).scalar < nodes.at(b).scalar; });

        chains_.push_back(chain);
      }
    }
  }

  std::cout << "Extracted " << chains_.size() << " topological chains" << std::endl;
  return 0;
}

int ActiveGraphPostProcess::buildActiveSubgraph(int nodeId, const std::set<int> &segIds,
                                                 ActiveSubgraph &subgraph) {
  subgraph.activeEdges.clear();
  subgraph.activeVertices.clear();

  const auto &nodes = io_.getNodes();
  auto it = nodes.find(nodeId);
  if (it == nodes.end()) {
    return -1;
  }

  double threshold = it->second.scalar;
  subgraph.threshold = threshold;

  auto imageData = io_.getImageData();
  if (!imageData) {
    std::cerr << "ERROR: No image data available" << std::endl;
    return -1;
  }

  int dims[3];
  imageData->GetDimensions(dims);

  auto *scalarArray = imageData->GetPointData()->GetArray(scalarFieldName_.c_str());
  auto *segIdArray = imageData->GetPointData()->GetArray("SegmentationId");

  if (!scalarArray || !segIdArray) {
    std::cerr << "ERROR: Missing required arrays in image data: scalar field '"
              << scalarFieldName_ << "' and SegmentationId are required" << std::endl;
    return -1;
  }

  std::array<double, 3> centroid = {0, 0, 0};
  std::array<double, 3> bboxMin = {1e30, 1e30, 1e30};
  std::array<double, 3> bboxMax = {-1e30, -1e30, -1e30};
  int vertexCount = 0;

  auto getPointId = [dims](int i, int j, int k) -> vtkIdType {
    return static_cast<vtkIdType>(k * dims[1] * dims[0] + j * dims[0] + i);
  };

  auto processEdge = [&](vtkIdType pid1, vtkIdType pid2) {
    // 如果segIds为空，跳过segmentation id检查
    if (!segIds.empty()) {
      int seg1 = static_cast<int>(segIdArray->GetComponent(pid1, 0));
      int seg2 = static_cast<int>(segIdArray->GetComponent(pid2, 0));

      if (segIds.find(seg1) == segIds.end() && segIds.find(seg2) == segIds.end()) {
        return;
      }
    }

    double val1 = scalarArray->GetComponent(pid1, 0);
    double val2 = scalarArray->GetComponent(pid2, 0);

    if ((val1 <= threshold && val2 > threshold) || (val2 <= threshold && val1 > threshold)) {
      GridEdge edge(pid1, pid2);
      sortGridEdge(edge);
      subgraph.activeEdges.push_back(edge);

      subgraph.activeVertices.insert(pid1);
      subgraph.activeVertices.insert(pid2);
    }
  };

  // 合并所有相关segmentation的bounding box
  int imin = 0;
  int imax = dims[0] - 1;
  int jmin = 0;
  int jmax = dims[1] - 1;
  int kmin = 0;
  int kmax = dims[2] - 1;
  if (!segIds.empty()) {
    if (!ensureSegmentationBboxes()) {
      std::cerr << "ERROR: Failed to build segmentation bounding boxes" << std::endl;
      return -1;
    }
    imin = dims[0];
    imax = -1;
    jmin = dims[1];
    jmax = -1;
    kmin = dims[2];
    kmax = -1;
    for (int segId : segIds) {
      auto bboxIt = segmentationBboxes_.find(segId);
      if (bboxIt == segmentationBboxes_.end()) {
        continue;
      }
      const auto &bbox = bboxIt->second;
      imin = std::min(imin, bbox[0]);
      imax = std::max(imax, bbox[1]);
      jmin = std::min(jmin, bbox[2]);
      jmax = std::max(jmax, bbox[3]);
      kmin = std::min(kmin, bbox[4]);
      kmax = std::max(kmax, bbox[5]);
    }
    if (imax < imin || jmax < jmin || kmax < kmin) {
      bboxMin = {0.0, 0.0, 0.0};
      bboxMax = {0.0, 0.0, 0.0};
      subgraph.centroid = centroid;
      subgraph.bboxMin = bboxMin;
      subgraph.bboxMax = bboxMax;
      subgraph.diagLength = 0.0;
      return 0;
    }
  }

  // 只遍历相关的bounding box区域
  for (int k = kmin; k <= kmax && k + 1 < dims[2]; ++k) {
    for (int j = jmin; j <= jmax && j + 1 < dims[1]; ++j) {
      for (int i = imin; i <= imax && i + 1 < dims[0]; ++i) {
        vtkIdType pid = getPointId(i, j, k);

        if (i + 1 < dims[0]) {
          processEdge(pid, getPointId(i + 1, j, k));
        }
        if (j + 1 < dims[1]) {
          processEdge(pid, getPointId(i, j + 1, k));
        }
        if (k + 1 < dims[2]) {
          processEdge(pid, getPointId(i, j, k + 1));
        }
      }
    }
  }

  std::sort(subgraph.activeEdges.begin(), subgraph.activeEdges.end());
  subgraph.activeEdges.erase(
      std::unique(subgraph.activeEdges.begin(), subgraph.activeEdges.end()),
      subgraph.activeEdges.end());

  for (vtkIdType pid : subgraph.activeVertices) {
    double p[3];
    imageData->GetPoint(pid, p);
    centroid[0] += p[0];
    centroid[1] += p[1];
    centroid[2] += p[2];

    bboxMin[0] = std::min(bboxMin[0], p[0]);
    bboxMin[1] = std::min(bboxMin[1], p[1]);
    bboxMin[2] = std::min(bboxMin[2], p[2]);
    bboxMax[0] = std::max(bboxMax[0], p[0]);
    bboxMax[1] = std::max(bboxMax[1], p[1]);
    bboxMax[2] = std::max(bboxMax[2], p[2]);
    vertexCount++;
  }

  if (vertexCount > 0) {
    centroid[0] /= vertexCount;
    centroid[1] /= vertexCount;
    centroid[2] /= vertexCount;
  } else {
    bboxMin = {0.0, 0.0, 0.0};
    bboxMax = {0.0, 0.0, 0.0};
  }

  subgraph.centroid = centroid;
  subgraph.bboxMin = bboxMin;
  subgraph.bboxMax = bboxMax;
  subgraph.diagLength = std::sqrt(
      (bboxMax[0] - bboxMin[0]) * (bboxMax[0] - bboxMin[0]) +
      (bboxMax[1] - bboxMin[1]) * (bboxMax[1] - bboxMin[1]) +
      (bboxMax[2] - bboxMin[2]) * (bboxMax[2] - bboxMin[2]));

  return 0;
}

double ActiveGraphPostProcess::computeJaccardDistance(const ActiveSubgraph &a,
                                                       const ActiveSubgraph &b) {
  const auto &edgesA = a.activeEdges;
  const auto &edgesB = b.activeEdges;

  size_t i = 0, j = 0;
  size_t interSize = 0;
  size_t unionSize = 0;

  while (i < edgesA.size() && j < edgesB.size()) {
    if (edgesA[i] == edgesB[j]) {
      interSize++;
      i++;
      j++;
    } else if (edgesA[i] < edgesB[j]) {
      i++;
    } else {
      j++;
    }
  }

  unionSize = edgesA.size() + edgesB.size() - interSize;

  if (unionSize == 0) {
    return 0.0;
  }

  return 1.0 - static_cast<double>(interSize) / static_cast<double>(unionSize);
}

double ActiveGraphPostProcess::deterministicWeight(int row, int col, int layerIndex) const {
  std::uint64_t x = static_cast<std::uint64_t>(gnnSeed_) + 0x9e3779b97f4a7c15ULL;
  x ^= static_cast<std::uint64_t>(row + 1) * 0xbf58476d1ce4e5b9ULL;
  x ^= static_cast<std::uint64_t>(col + 1) * 0x94d049bb133111ebULL;
  x ^= static_cast<std::uint64_t>(layerIndex + 1) * 0x2545f4914f6cdd1dULL;
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  const double unit = static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
  return unit * 2.0 - 1.0;
}

std::vector<double> ActiveGraphPostProcess::applyGcnLayer(
    const std::vector<double> &features,
    const std::vector<std::vector<int>> &neighbors,
    int inDim,
    int outDim,
    int layerIndex) const {
  const int vertexCount = static_cast<int>(neighbors.size());
  std::vector<double> output(static_cast<size_t>(vertexCount) * outDim, 0.0);
  if (vertexCount == 0) {
    return output;
  }

  const double scale = std::sqrt(2.0 / static_cast<double>(std::max(1, inDim + outDim)));
  std::vector<double> weights(static_cast<size_t>(inDim) * outDim, 0.0);
  for (int in = 0; in < inDim; ++in) {
    for (int out = 0; out < outDim; ++out) {
      weights[static_cast<size_t>(in) * outDim + out] =
          deterministicWeight(in, out, layerIndex) * scale;
    }
  }

  std::vector<double> aggregate(inDim, 0.0);
  for (int v = 0; v < vertexCount; ++v) {
    std::fill(aggregate.begin(), aggregate.end(), 0.0);
    const double degreeV = static_cast<double>(neighbors[v].size() + 1);

    for (int f = 0; f < inDim; ++f) {
      aggregate[f] += features[static_cast<size_t>(v) * inDim + f] / degreeV;
    }

    for (int nbr : neighbors[v]) {
      const double degreeNbr = static_cast<double>(neighbors[nbr].size() + 1);
      const double norm = 1.0 / std::sqrt(degreeV * degreeNbr);
      for (int f = 0; f < inDim; ++f) {
        aggregate[f] += features[static_cast<size_t>(nbr) * inDim + f] * norm;
      }
    }

    for (int out = 0; out < outDim; ++out) {
      double value = 0.0;
      for (int in = 0; in < inDim; ++in) {
        value += aggregate[in] * weights[static_cast<size_t>(in) * outDim + out];
      }
      output[static_cast<size_t>(v) * outDim + out] = std::max(0.0, value);
    }
  }

  return output;
}

std::vector<double> ActiveGraphPostProcess::encodeGnn(const ActiveSubgraph &subgraph) const {
  const int featureDim = 11;
  const int hiddenDim = std::max(1, gnnHiddenDim_);
  const int vertexCount = static_cast<int>(subgraph.activeVertices.size());

  std::vector<double> emptyEmbedding(static_cast<size_t>(hiddenDim) * 2 + 2, 0.0);
  if (vertexCount == 0) {
    return emptyEmbedding;
  }

  auto imageData = io_.getImageData();
  auto *scalarArray = imageData->GetPointData()->GetArray(scalarFieldName_.c_str());
  auto *segIdArray = imageData->GetPointData()->GetArray("SegmentationId");
  if (!scalarArray || !segIdArray) {
    return emptyEmbedding;
  }

  if (!ensureScalarRange()) {
    return emptyEmbedding;
  }
  const double scalarRange = std::max(1e-12, scalarMax_ - scalarMin_);
  const double spatialScale = std::max(1e-12, subgraph.diagLength);

  std::vector<vtkIdType> vertexIds(subgraph.activeVertices.begin(), subgraph.activeVertices.end());
  std::unordered_map<vtkIdType, int> localIndex;
  localIndex.reserve(vertexIds.size());
  for (int i = 0; i < vertexCount; ++i) {
    localIndex[vertexIds[static_cast<size_t>(i)]] = i;
  }

  std::vector<std::vector<int>> neighbors(static_cast<size_t>(vertexCount));
  for (const auto &edge : subgraph.activeEdges) {
    auto itA = localIndex.find(edge.first);
    auto itB = localIndex.find(edge.second);
    if (itA == localIndex.end() || itB == localIndex.end()) {
      continue;
    }
    neighbors[static_cast<size_t>(itA->second)].push_back(itB->second);
    neighbors[static_cast<size_t>(itB->second)].push_back(itA->second);
  }

  std::vector<double> features(static_cast<size_t>(vertexCount) * featureDim, 0.0);
  for (int i = 0; i < vertexCount; ++i) {
    const vtkIdType pid = vertexIds[static_cast<size_t>(i)];
    const double scalar = scalarArray->GetComponent(pid, 0);
    const int segId = static_cast<int>(segIdArray->GetComponent(pid, 0));
    double p[3];
    imageData->GetPoint(pid, p);

    const size_t base = static_cast<size_t>(i) * featureDim;
    features[base + 0] = (scalar - scalarMin_) / scalarRange;
    features[base + 1] = (scalar - subgraph.threshold) / scalarRange;
    features[base + 2] = std::abs(scalar - subgraph.threshold) / scalarRange;
    features[base + 3] = (p[0] - subgraph.centroid[0]) / spatialScale;
    features[base + 4] = (p[1] - subgraph.centroid[1]) / spatialScale;
    features[base + 5] = (p[2] - subgraph.centroid[2]) / spatialScale;
    features[base + 6] = static_cast<double>(neighbors[static_cast<size_t>(i)].size()) / 6.0;
    features[base + 7] = std::sin(static_cast<double>(segId) * 0.017453292519943295);
    features[base + 8] = std::cos(static_cast<double>(segId) * 0.017453292519943295);
    features[base + 9] = std::sin(static_cast<double>(segId) * 0.0009765625);
    features[base + 10] = std::cos(static_cast<double>(segId) * 0.0009765625);
  }

  std::vector<double> hidden1 = applyGcnLayer(features, neighbors, featureDim, hiddenDim, 0);
  std::vector<double> hidden2 = applyGcnLayer(hidden1, neighbors, hiddenDim, hiddenDim, 1);

  std::vector<double> embedding(static_cast<size_t>(hiddenDim) * 2 + 2, 0.0);
  for (int h = 0; h < hiddenDim; ++h) {
    embedding[static_cast<size_t>(hiddenDim) + h] = -std::numeric_limits<double>::max();
  }

  for (int i = 0; i < vertexCount; ++i) {
    for (int h = 0; h < hiddenDim; ++h) {
      const double value = hidden2[static_cast<size_t>(i) * hiddenDim + h];
      embedding[static_cast<size_t>(h)] += value;
      embedding[static_cast<size_t>(hiddenDim) + h] =
          std::max(embedding[static_cast<size_t>(hiddenDim) + h], value);
    }
  }

  for (int h = 0; h < hiddenDim; ++h) {
    embedding[static_cast<size_t>(h)] /= static_cast<double>(vertexCount);
  }
  embedding[static_cast<size_t>(hiddenDim) * 2] =
      std::log1p(static_cast<double>(subgraph.activeVertices.size()));
  embedding[static_cast<size_t>(hiddenDim) * 2 + 1] =
      std::log1p(static_cast<double>(subgraph.activeEdges.size()));

  double norm = 0.0;
  for (double value : embedding) {
    norm += value * value;
  }
  norm = std::sqrt(norm);
  if (norm > 1e-12) {
    for (double &value : embedding) {
      value /= norm;
    }
  }

  return embedding;
}

double ActiveGraphPostProcess::computeGnnDistance(const ActiveSubgraph &a,
                                                  const ActiveSubgraph &b) {
  const std::vector<double> embA = encodeGnn(a);
  const std::vector<double> embB = encodeGnn(b);
  const size_t dim = std::min(embA.size(), embB.size());
  if (dim == 0) {
    return 0.0;
  }

  double dot = 0.0;
  double normA = 0.0;
  double normB = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    dot += embA[i] * embB[i];
    normA += embA[i] * embA[i];
    normB += embB[i] * embB[i];
  }

  if (normA <= 1e-12 && normB <= 1e-12) {
    return 0.0;
  }
  if (normA <= 1e-12 || normB <= 1e-12) {
    return 1.0;
  }

  const double cosine = dot / std::sqrt(normA * normB);
  return 1.0 - std::max(-1.0, std::min(1.0, cosine));
}

double ActiveGraphPostProcess::computePositionDistance(const ActiveSubgraph &a,
                                                         const ActiveSubgraph &b) {
  double dx = a.centroid[0] - b.centroid[0];
  double dy = a.centroid[1] - b.centroid[1];
  double dz = a.centroid[2] - b.centroid[2];
  double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

  double maxDiag = std::max(a.diagLength, 1e-10);
  return dist / maxDiag;
}

double ActiveGraphPostProcess::computeBboxDistance(const ActiveSubgraph &a,
                                                     const ActiveSubgraph &b) {
  double diff = std::abs(a.diagLength - b.diagLength);
  double maxDiag = std::max(a.diagLength, 1e-10);
  return diff / maxDiag;
}

int ActiveGraphPostProcess::greedySimplify(double epsilonJ, double epsilonPos, double epsilonBbox) {
  keptNodes_.clear();
  decisions_.clear();

  for (const auto &chain : chains_) {
    if (chain.nodeIds.empty()) {
      continue;
    }

    int startNodeId = chain.nodeIds.front();
    int endNodeId = chain.nodeIds.back();
    keptNodes_.insert(startNodeId);
    keptNodes_.insert(endNodeId);

    if (chain.nodeIds.size() <= 2) {
      continue;
    }

    std::set<int> startSegIds = getLocalSegmentationIds(chain, 0);
    ActiveSubgraph refSubgraph;
    if (buildActiveSubgraph(startNodeId, startSegIds, refSubgraph) != 0) {
      std::cerr << "WARNING: Failed to build active subgraph for reference node " << startNodeId
                << std::endl;
      continue;
    }

    int refNodeId = startNodeId;

    for (size_t i = 1; i < chain.nodeIds.size() - 1; ++i) {
      int nodeId = chain.nodeIds[i];
      std::set<int> segIds = getLocalSegmentationIds(chain, i);

      ActiveSubgraph candSubgraph;
      if (buildActiveSubgraph(nodeId, segIds, candSubgraph) != 0) {
        std::cerr << "WARNING: Failed to build active subgraph for node " << nodeId << std::endl;
        continue;
      }

      double distJ = similarityMode_ == SimilarityMode::GNN
                         ? computeGnnDistance(refSubgraph, candSubgraph)
                         : computeJaccardDistance(refSubgraph, candSubgraph);
      double distPos = computePositionDistance(refSubgraph, candSubgraph);
      double distBbox = computeBboxDistance(refSubgraph, candSubgraph);
      bool keep = distJ >= epsilonJ || distPos >= epsilonPos || distBbox >= epsilonBbox;

      decisions_.push_back(
          {chain.chainId, refNodeId, nodeId, distJ, distPos, distBbox, keep});

      if (keep) {
        keptNodes_.insert(nodeId);
        refNodeId = nodeId;
        refSubgraph = candSubgraph;
      }
    }
  }

  std::cout << "Simplified: kept " << keptNodes_.size() << " nodes" << std::endl;
  return 0;
}

} // namespace active_graph
