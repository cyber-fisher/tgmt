#pragma once

#include <string>

#include "ActiveGraphPostProcess.h"
#include "MergeTreeIO.h"

namespace active_graph {

struct ActiveGraphSimplificationParams {
  enum class GraphSimilarityMode {
    Jaccard,
    GNN,
  };

  std::string treeNodesPath;
  std::string treeArcsPath;
  std::string segmentationPath;
  std::string scalarFieldName{"v"};
  std::string outputPrefix{"active_graph"};
  GraphSimilarityMode graphSimilarityMode{GraphSimilarityMode::Jaccard};
  double epsilonJ{0.15};
  double epsilonPos{0.10};
  double epsilonBbox{0.10};
  int gnnHiddenDim{64};
  unsigned int gnnSeed{17};
  bool useWeightedDistance{false};
  double wJ{0.7};
  double wP{0.2};
  double wB{0.1};
};

class ActiveGraphMergeTreeSimplifier {
public:
  int run(const ActiveGraphSimplificationParams &params);

private:
  int loadData(const ActiveGraphSimplificationParams &params);
  int buildTopologicalChains();
  int simplify();
  int writeOutputs(const ActiveGraphSimplificationParams &params) const;

  MergeTreeIO io_;
  ActiveGraphPostProcess postProcess_{io_};
  ActiveGraphSimplificationParams params_;
};

} // namespace active_graph
