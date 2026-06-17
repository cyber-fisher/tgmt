#include "IsmMergeTreeSimplifier.h"
#include "MergeTreeIO.h"
#include "IsmFieldBuilder.h"
#include "SimilarityMetrics.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkXMLGenericDataObjectReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

IsmMergeTreeSimplifier::IsmMergeTreeSimplifier() = default;
IsmMergeTreeSimplifier::~IsmMergeTreeSimplifier() = default;

int IsmMergeTreeSimplifier::run(const IsmSimplificationParams &params) {
  std::cout << "Starting ISM merge tree simplification..." << std::endl;

  // Step 1: Load data
  if (loadData(params) != 0) {
    std::cerr << "Failed to load data" << std::endl;
    return -1;
  }

  // Step 2: Build topological chains
  if (buildTopologicalChains() != 0) {
    std::cerr << "Failed to build topological chains" << std::endl;
    return -2;
  }

  std::cout << "Built " << chains_.size() << " topological chains" << std::endl;

  // Step 3: Simplify
  if (simplify(params) != 0) {
    std::cerr << "Failed to simplify" << std::endl;
    return -3;
  }

  std::cout << "Simplification complete. Kept " << keptNodeIds_.size()
            << " nodes out of " << nodes_.size() << std::endl;

  // Step 4: Write outputs
  if (writeOutputs(params) != 0) {
    std::cerr << "Failed to write outputs" << std::endl;
    return -4;
  }

  std::cout << "Outputs written to: " << params.outputPrefix << std::endl;
  return 0;
}

int IsmMergeTreeSimplifier::loadData(const IsmSimplificationParams &params) {
  std::cout << "Loading data..." << std::endl;

  // Load nodes
  auto reader = vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
  reader->SetFileName(params.treeNodesPath.c_str());
  reader->Update();
  nodeGrid_ = vtkUnstructuredGrid::SafeDownCast(reader->GetOutput());
  if (!nodeGrid_) {
    std::cerr << "Failed to read node grid from: " << params.treeNodesPath << std::endl;
    return -1;
  }

  // Load arcs
  reader->SetFileName(params.treeArcsPath.c_str());
  reader->Update();
  arcGrid_ = vtkUnstructuredGrid::SafeDownCast(reader->GetOutput());
  if (!arcGrid_) {
    std::cerr << "Failed to read arc grid from: " << params.treeArcsPath << std::endl;
    return -2;
  }

  // Load segmentation image
  reader->SetFileName(params.segmentationPath.c_str());
  reader->Update();
  segmentationImage_ = vtkImageData::SafeDownCast(reader->GetOutput());
  if (!segmentationImage_) {
    std::cerr << "Failed to read segmentation image from: " << params.segmentationPath << std::endl;
    return -3;
  }

  // Parse nodes
  nodes_ = MergeTreeIO::readNodesFromVTU(nodeGrid_);
  arcs_ = MergeTreeIO::readArcsFromVTU(arcGrid_);

  std::cout << "Loaded " << nodes_.size() << " nodes and " << arcs_.size()
            << " arcs" << std::endl;

  return 0;
}

int IsmMergeTreeSimplifier::buildTopologicalChains() {
  std::cout << "Building topological chains..." << std::endl;

  // Build adjacency: node -> incident arcs
  std::unordered_map<int, std::vector<int>> nodeToArcs;
  for (size_t i = 0; i < arcs_.size(); i++) {
    nodeToArcs[arcs_[i].downNodeId].push_back(i);
    nodeToArcs[arcs_[i].upNodeId].push_back(i);
  }

  // Identify structural nodes (degree != 1 or critical)
  std::unordered_set<int> structuralNodeIds;
  for (const auto &node : nodes_) {
    if (isStructuralNode(node)) {
      structuralNodeIds.insert(node.nodeId);
    }
  }

  std::cout << "Found " << structuralNodeIds.size() << " structural nodes" << std::endl;

  // Extract chains between structural nodes
  std::unordered_set<int> visitedArcs;
  std::unordered_map<int, int> nodeIdToIndex;
  for (size_t i = 0; i < nodes_.size(); i++) {
    nodeIdToIndex[nodes_[i].nodeId] = i;
  }

  for (const auto &node : nodes_) {
    if (structuralNodeIds.find(node.nodeId) == structuralNodeIds.end())
      continue;

    for (int arcIdx : nodeToArcs[node.nodeId]) {
      if (visitedArcs.count(arcIdx))
        continue;

      TopologicalChain chain;
      int currentNodeId = node.nodeId;
      int currentArcIdx = arcIdx;

      while (true) {
        visitedArcs.insert(currentArcIdx);
        chain.arcIds.push_back(currentArcIdx);

        const auto &arc = arcs_[currentArcIdx];
        int nextNodeId = (arc.downNodeId == currentNodeId) ? arc.upNodeId : arc.downNodeId;
        chain.nodeIds.push_back(currentNodeId);

        // Check if next node is structural
        if (structuralNodeIds.count(nextNodeId)) {
          chain.nodeIds.push_back(nextNodeId);
          break;
        }

        // Find next arc
        int nextArcIdx = -1;
        for (int candArcIdx : nodeToArcs[nextNodeId]) {
          if (candArcIdx != currentArcIdx) {
            nextArcIdx = candArcIdx;
            break;
          }
        }

        if (nextArcIdx == -1)
          break;

        currentNodeId = nextNodeId;
        currentArcIdx = nextArcIdx;
      }

      if (chain.nodeIds.size() >= 2) {
        chains_.push_back(chain);
      }
    }
  }

  return 0;
}

bool IsmMergeTreeSimplifier::isStructuralNode(const TreeNode &node) const {
  // Structural nodes are: root, leaf, branching point, or critical type != 0
  // For now, we'll consider nodes with CriticalType != 0 as structural
  // In a full implementation, we'd also check degree
  return node.criticalType != 0;
}

int IsmMergeTreeSimplifier::buildDescriptor(int nodeId) {
  // This is a placeholder - actual implementation will use IsmFieldBuilder
  return 0;
}

int IsmMergeTreeSimplifier::simplify(const IsmSimplificationParams &params) {
  std::cout << "Simplifying chains..." << std::endl;

  // For each chain, perform greedy simplification
  for (size_t chainIdx = 0; chainIdx < chains_.size(); chainIdx++) {
    const auto &chain = chains_[chainIdx];
    std::cout << "Processing chain " << chainIdx + 1 << "/" << chains_.size()
              << " with " << chain.nodeIds.size() << " nodes" << std::endl;

    if (chain.nodeIds.size() <= 2) {
      // Keep both endpoints
      keptNodeIds_.push_back(chain.nodeIds.front());
      keptNodeIds_.push_back(chain.nodeIds.back());
      continue;
    }

    // Greedy simplification: keep first node as reference
    keptNodeIds_.push_back(chain.nodeIds.front());
    int refNodeId = chain.nodeIds.front();

    // Cache reference descriptor
    std::unique_ptr<IsmDescriptor> refDescCache = nullptr;

    for (size_t i = 1; i < chain.nodeIds.size() - 1; i++) {
      int candidateNodeId = chain.nodeIds[i];
      std::cout << "  Processing node " << i << "/" << chain.nodeIds.size() - 2
                << " (id=" << candidateNodeId << ")" << std::endl;

      // Build descriptor for candidate
      auto candDesc = IsmFieldBuilder::buildDescriptor(
          candidateNodeId, nodes_, arcs_, segmentationImage_);

      if (!candDesc) {
        // If we can't build descriptors, keep the node to be safe
        keptNodeIds_.push_back(candidateNodeId);
        refNodeId = candidateNodeId;
        refDescCache = nullptr; // Invalidate cache
        continue;
      }

      // Build reference descriptor (use cache if available)
      if (!refDescCache) {
        refDescCache = IsmFieldBuilder::buildDescriptor(
            refNodeId, nodes_, arcs_, segmentationImage_);
      }

      if (!refDescCache) {
        // If we can't build reference descriptor, keep the node
        keptNodeIds_.push_back(candidateNodeId);
        refNodeId = candidateNodeId;
        refDescCache = nullptr;
        continue;
      }

      // Compute similarity
      double distIsm = SimilarityMetrics::computeDistISM(
          *refDescCache, *candDesc, 32); // 32 histogram bins

      // Compute position and bbox differences
      double distPos = 0.0;
      for (int d = 0; d < 3; d++) {
        double diff = refDescCache->centroid[d] - candDesc->centroid[d];
        distPos += diff * diff;
      }
      distPos = std::sqrt(distPos);

      double distBbox = std::abs(refDescCache->bboxDiagonal - candDesc->bboxDiagonal) /
                        refDescCache->bboxDiagonal;

      // Decision: keep if any metric exceeds threshold
      bool keep = (distIsm >= params.epsilonIsm) || (distPos >= params.epsilonPos) || (distBbox >= params.epsilonBbox);

      if (keep) {
        keptNodeIds_.push_back(candidateNodeId);
        refNodeId = candidateNodeId;
        distToRef_.push_back(distIsm);
      }
    }

    // Always keep last node
    keptNodeIds_.push_back(chain.nodeIds.back());
  }

  // Remove duplicates from keptNodeIds_
  std::sort(keptNodeIds_.begin(), keptNodeIds_.end());
  keptNodeIds_.erase(std::unique(keptNodeIds_.begin(), keptNodeIds_.end()),
                     keptNodeIds_.end());

  return 0;
}

int IsmMergeTreeSimplifier::writeOutputs(const IsmSimplificationParams &params) const {
  std::cout << "Writing outputs..." << std::endl;

  // Write simplified nodes
  MergeTreeIO::writeSimplifiedNodes(nodeGrid_, keptNodeIds_,
                                    params.outputPrefix + "_simplified_nodes.vtu");

  // Write simplified arcs (simplified version)
  // For now, just copy original arcs - a full implementation would rebuild arcs
  MergeTreeIO::writeGrid(arcGrid_, params.outputPrefix + "_simplified_arcs.vtu");

  // Write stats CSV
  MergeTreeIO::writeStatsCSV(params.outputPrefix + "_stats.csv",
                              chains_, keptNodeIds_, distToRef_);

  return 0;
}

// Helper method implementations (stubs for now)
std::vector<int> IsmMergeTreeSimplifier::getLocalSegmentationIds(int nodeId) const {
  // TODO: Implement based on adjacent arcs
  return std::vector<int>();
}

LocalVolumeBlock IsmMergeTreeSimplifier::extractLocalBlock(int nodeId) const {
  // TODO: Implement local block extraction
  return LocalVolumeBlock();
}