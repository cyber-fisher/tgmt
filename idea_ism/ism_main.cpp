#include "IsmMergeTreeSimplifier.h"

#include <iostream>
#include <string>
#include <cstring>

void printUsage(const char* programName) {
  std::cerr << "Usage: " << programName << " <nodes.vtu> <arcs.vtu> <segmentation.vti> [options]" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  -o, --output <prefix>      Output file prefix (default: ism_simplified)" << std::endl;
  std::cerr << "  -e, --epsilon-ism <value>  ISM distance threshold (default: 0.15)" << std::endl;
  std::cerr << "  -p, --epsilon-pos <value>  Position distance threshold (default: 0.10)" << std::endl;
  std::cerr << "  -b, --epsilon-bbox <value> Bounding box difference threshold (default: 0.10)" << std::endl;
  std::cerr << "  -h, --help                 Show this help message" << std::endl;
  std::cerr << "Example: " << programName << " nodes.vtu arcs.vtu segmentation.vti -o my_output -e 0.2" << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printUsage(argv[0]);
    return 1;
  }

  IsmSimplificationParams params;
  params.treeNodesPath = argv[1];
  params.treeArcsPath = argv[2];
  params.segmentationPath = argv[3];

  // Parse optional arguments
  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
      if (i + 1 < argc) {
        params.outputPrefix = argv[++i];
      } else {
        std::cerr << "Error: -o/--output requires a value" << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--epsilon-ism") == 0) {
      if (i + 1 < argc) {
        try {
          params.epsilonIsm = std::stod(argv[++i]);
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid epsilon-ism value: " << argv[i] << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: -e/--epsilon-ism requires a value" << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--epsilon-pos") == 0) {
      if (i + 1 < argc) {
        try {
          params.epsilonPos = std::stod(argv[++i]);
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid epsilon-pos value: " << argv[i] << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: -p/--epsilon-pos requires a value" << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--epsilon-bbox") == 0) {
      if (i + 1 < argc) {
        try {
          params.epsilonBbox = std::stod(argv[++i]);
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid epsilon-bbox value: " << argv[i] << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: -b/--epsilon-bbox requires a value" << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Error: Unknown option: " << argv[i] << std::endl;
      printUsage(argv[0]);
      return 1;
    }
  }

  std::cout << "Running ISM Merge Tree Simplifier with parameters:" << std::endl;
  std::cout << "  Nodes: " << params.treeNodesPath << std::endl;
  std::cout << "  Arcs: " << params.treeArcsPath << std::endl;
  std::cout << "  Segmentation: " << params.segmentationPath << std::endl;
  std::cout << "  Output prefix: " << params.outputPrefix << std::endl;
  std::cout << "  Epsilon ISM: " << params.epsilonIsm << std::endl;
  std::cout << "  Epsilon Pos: " << params.epsilonPos << std::endl;
  std::cout << "  Epsilon BBox: " << params.epsilonBbox << std::endl;

  IsmMergeTreeSimplifier simplifier;
  int result = simplifier.run(params);

  if (result != 0) {
    std::cerr << "Simplification failed with error code: " << result << std::endl;
    return result;
  }

  std::cout << "Simplification completed successfully!" << std::endl;
  return 0;
}
