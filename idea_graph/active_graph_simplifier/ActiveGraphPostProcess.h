#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <vtkType.h>

namespace active_graph {

struct TreeNode;
struct TreeArc;

using GridEdge = std::pair<vtkIdType, vtkIdType>;

struct ActiveSubgraph {
  std::vector<GridEdge> activeEdges;
  std::set<vtkIdType> activeVertices;
  std::array<double, 3> centroid;
  std::array<double, 3> bboxMin;
  std::array<double, 3> bboxMax;
  double diagLength;
  double threshold;
};

struct TopologicalChain {
  int chainId;
  std::vector<int> nodeIds;
  std::vector<int> arcIds;
};

struct SimplificationDecision {
  int chainId;
  int refNodeId;
  int candidateNodeId;
  double distJ;
  double distPos;
  double distBbox;
  bool keep;
};

class MergeTreeIO;

class ActiveGraphPostProcess {
public:
  enum class SimilarityMode {
    Jaccard,
    GNN,
  };

  ActiveGraphPostProcess(MergeTreeIO &io);

  void setScalarFieldName(const std::string &scalarFieldName);
  void setSimilarityMode(SimilarityMode mode);
  void setGnnOptions(int hiddenDim, unsigned int seed);
  std::string getGraphDistanceName() const;

  int extractTopologicalChains();
  const std::vector<TopologicalChain> &getChains() const { return chains_; }

  int buildActiveSubgraph(int nodeId, const std::set<int> &segIds, ActiveSubgraph &subgraph);

  double computeJaccardDistance(const ActiveSubgraph &a, const ActiveSubgraph &b);
  double computeGnnDistance(const ActiveSubgraph &a, const ActiveSubgraph &b);
  double computePositionDistance(const ActiveSubgraph &a, const ActiveSubgraph &b);
  double computeBboxDistance(const ActiveSubgraph &a, const ActiveSubgraph &b);

  int greedySimplify(double epsilonJ, double epsilonPos, double epsilonBbox);

  const std::set<int> &getKeptNodes() const { return keptNodes_; }
  const std::vector<SimplificationDecision> &getDecisions() const { return decisions_; }

private:
  MergeTreeIO &io_;
  std::vector<TopologicalChain> chains_;
  std::set<int> keptNodes_;
  std::vector<SimplificationDecision> decisions_;
  std::string scalarFieldName_{"v"};
  SimilarityMode similarityMode_{SimilarityMode::Jaccard};
  int gnnHiddenDim_{64};
  unsigned int gnnSeed_{17};
  mutable bool scalarRangeReady_{false};
  mutable double scalarMin_{0.0};
  mutable double scalarMax_{1.0};
  mutable bool segmentationBboxesReady_{false};
  mutable std::map<int, std::array<int, 6>> segmentationBboxes_;

  bool isCriticalNode(int nodeId);
  void sortGridEdge(GridEdge &edge);
  std::set<int> getLocalSegmentationIds(const TopologicalChain &chain, std::size_t nodeIndex) const;
  bool ensureScalarRange() const;
  bool ensureSegmentationBboxes() const;
  std::vector<double> encodeGnn(const ActiveSubgraph &subgraph) const;
  std::vector<double> applyGcnLayer(const std::vector<double> &features,
                                    const std::vector<std::vector<int>> &neighbors,
                                    int inDim,
                                    int outDim,
                                    int layerIndex) const;
  double deterministicWeight(int row, int col, int layerIndex) const;
};

} // namespace active_graph
