#include "ActiveGraphMergeTreeSimplifier.h"

#include <cstring>
#include <iostream>
#include <string>

void printUsage(const char *progName) {
  std::cout << "Usage: " << progName << " [options]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -n <path>     Tree nodes file path (.vtu)" << std::endl;
  std::cout << "  -a <path>     Tree arcs file path (.vtu)" << std::endl;
  std::cout << "  -s <path>     Segmentation file path (.vti)" << std::endl;
  std::cout << "  -o <prefix>   Output file prefix (default: active_graph)" << std::endl;
  std::cout << "  --scalar <name>       Scalar field name in .vti (default: v)" << std::endl;
  std::cout << "  --similarity <mode>   Graph distance: jaccard or gnn (default: jaccard)" << std::endl;
  std::cout << "  -j <value>    Epsilon for graph distance / Jaccard / GNN (default: 0.15)" << std::endl;
  std::cout << "  -p <value>    Epsilon for position distance (default: 0.10)" << std::endl;
  std::cout << "  -b <value>    Epsilon for bbox distance (default: 0.10)" << std::endl;
  std::cout << "  --gnn-hidden <value>  GNN hidden dimension (default: 64)" << std::endl;
  std::cout << "  --gnn-seed <value>    GNN deterministic weight seed (default: 17)" << std::endl;
  std::cout << "  -h            Print this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Example:" << std::endl;
  std::cout << "  " << progName << " -n nodes.vtu -a arcs.vtu -s data.vti -o result -j 0.2 -p 0.15 -b 0.15" << std::endl;
  std::cout << "  " << progName << " --similarity gnn -o result_gnn -j 0.15" << std::endl;
}

int main(int argc, char *argv[]) {
  active_graph::ActiveGraphSimplificationParams params;

  params.treeNodesPath = "/home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_nodes.vtu";
  params.treeArcsPath = "/home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_arcs.vtu";
  params.segmentationPath = "/home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_2.vti";

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      return 0;
    }
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      params.treeNodesPath = argv[++i];
    } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
      params.treeArcsPath = argv[++i];
    } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
      params.segmentationPath = argv[++i];
    } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      params.outputPrefix = argv[++i];
    } else if (strcmp(argv[i], "--scalar") == 0 && i + 1 < argc) {
      params.scalarFieldName = argv[++i];
    } else if (strcmp(argv[i], "--similarity") == 0 && i + 1 < argc) {
      const std::string mode = argv[++i];
      if (mode == "jaccard") {
        params.graphSimilarityMode =
            active_graph::ActiveGraphSimplificationParams::GraphSimilarityMode::Jaccard;
      } else if (mode == "gnn" || mode == "bgnn") {
        params.graphSimilarityMode =
            active_graph::ActiveGraphSimplificationParams::GraphSimilarityMode::GNN;
      } else {
        std::cerr << "ERROR: Unknown similarity mode '" << mode
                  << "'. Expected 'jaccard' or 'gnn'." << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
      params.epsilonJ = std::stod(argv[++i]);
    } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
      params.epsilonPos = std::stod(argv[++i]);
    } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
      params.epsilonBbox = std::stod(argv[++i]);
    } else if ((strcmp(argv[i], "--gnn-hidden") == 0 ||
                strcmp(argv[i], "--bgnn-hidden") == 0) &&
               i + 1 < argc) {
      params.gnnHiddenDim = std::stoi(argv[++i]);
    } else if ((strcmp(argv[i], "--gnn-seed") == 0 ||
                strcmp(argv[i], "--bgnn-seed") == 0) &&
               i + 1 < argc) {
      params.gnnSeed = static_cast<unsigned int>(std::stoul(argv[++i]));
    } else {
      std::cerr << "ERROR: Unknown or incomplete option '" << argv[i] << "'" << std::endl;
      printUsage(argv[0]);
      return 1;
    }
  }

  std::cout << "=== Configuration ===" << std::endl;
  std::cout << "Nodes file: " << params.treeNodesPath << std::endl;
  std::cout << "Arcs file: " << params.treeArcsPath << std::endl;
  std::cout << "Segmentation file: " << params.segmentationPath << std::endl;
  std::cout << "Scalar field: " << params.scalarFieldName << std::endl;
  std::cout << "Output prefix: " << params.outputPrefix << std::endl;
  const bool useGnn =
      params.graphSimilarityMode ==
      active_graph::ActiveGraphSimplificationParams::GraphSimilarityMode::GNN;
  std::cout << "Graph similarity: " << (useGnn ? "gnn" : "jaccard") << std::endl;
  std::cout << "Epsilon Graph: " << params.epsilonJ << std::endl;
  std::cout << "Epsilon Position: " << params.epsilonPos << std::endl;
  std::cout << "Epsilon BBox: " << params.epsilonBbox << std::endl;
  if (useGnn) {
    std::cout << "GNN hidden dim: " << params.gnnHiddenDim << std::endl;
    std::cout << "GNN seed: " << params.gnnSeed << std::endl;
  }
  std::cout << std::endl;

  active_graph::ActiveGraphMergeTreeSimplifier simplifier;
  int result = simplifier.run(params);

  return result;
}
