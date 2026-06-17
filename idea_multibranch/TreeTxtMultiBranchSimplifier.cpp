#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVertex.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct NodeRecord {
  int nodeId{};
  int parentId{-1};
  double scalar{};
  std::vector<int> children{};
  int degree{};
  int criticalType{};
  double regionSize{};
  double regionSpan{};
  std::vector<int> subtreeVertices{};
  double point[3]{0.0, 0.0, 0.0};
  std::vector<int> incidentArcIds{};
};

struct ArcRecord {
  int arcId{};
  int downNodeId{};
  int upNodeId{};
  int segmentationId{-1};
  double regionSize{};
  double regionSpan{};
};

struct BranchRecord {
  int branchId{};
  std::vector<int> nodeIds{};
  std::vector<int> arcIds{};
  std::vector<int> keepIndices{};
};

struct BranchInfo {
  int branchId{};
  std::vector<int> nodeIds{};
  double scalarMin{};
  double scalarMax{};
  double persistence{};
};

struct NewArcRecord {
  int newArcId{};
  int downNodeId{};
  int upNodeId{};
  double regionSize{};
  double regionSpan{};
  std::vector<int> mergedArcIds{};
};

struct SimplifiedTree {
  std::vector<int> keptNodeIds{};
  std::vector<NewArcRecord> newArcs{};
  std::unordered_map<int, int> nodeKeepMask{};
};

struct ProgramOptions {
  std::string treePath{};
  std::string imagePath{};
  std::string outputPrefix{"idea_multibranch/result/output"};
  double eps{0.25};
  double alpha{0.5};
  int treeType{0};
};

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error(msg);
}

void printUsage() {
  std::cerr
    << "Usage: treeTxtMultiBranchSimplifier -t <tree.txt> -s <image.vti> "
    << "[-o <output_prefix>] [-e <eps>] [-a <alpha>] [-T <0|1>]\n"
    << "  -a  alpha weight mixing (0=geo only, 1=topo only, default 0.5)\n";
}

ProgramOptions parseArgs(int argc, char **argv) {
  ProgramOptions opt{};
  for(int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto requireValue = [&](const std::string &name) -> std::string {
      if(i + 1 >= argc) fail("Missing value for " + name + ".");
      return argv[++i];
    };
    if(arg == "-t") opt.treePath = requireValue(arg);
    else if(arg == "-s") opt.imagePath = requireValue(arg);
    else if(arg == "-o") opt.outputPrefix = requireValue(arg);
    else if(arg == "-e") opt.eps = std::stod(requireValue(arg));
    else if(arg == "-a") opt.alpha = std::stod(requireValue(arg));
    else if(arg == "-T") opt.treeType = std::stoi(requireValue(arg));
    else if(arg == "-h" || arg == "--help") { printUsage(); std::exit(0); }
    else fail("Unknown argument: " + arg);
  }
  if(opt.treePath.empty() || opt.imagePath.empty()) {
    printUsage();
    fail("Both -t and -s are required.");
  }
  if(opt.eps < 0.0 || opt.eps > 1.0) fail("eps must be in [0, 1].");
  if(opt.alpha < 0.0 || opt.alpha > 1.0) fail("alpha must be in [0, 1].");
  if(opt.treeType != 0 && opt.treeType != 1) fail("treeType must be 0 or 1.");
  return opt;
}

vtkSmartPointer<vtkImageData> readImage(const std::string &path) {
  vtkNew<vtkXMLImageDataReader> reader{};
  reader->SetFileName(path.c_str());
  reader->Update();
  auto *image = reader->GetOutput();
  if(image == nullptr) fail("Unable to read image `" + path + "`.");
  auto out = vtkSmartPointer<vtkImageData>::New();
  out->DeepCopy(image);
  return out;
}

std::vector<NodeRecord> readTreeTxt(const std::string &path) {
  std::ifstream in(path);
  if(!in) fail("Unable to open tree file `" + path + "`.");
  std::vector<NodeRecord> nodes{};
  std::unordered_map<int, int> idToIndex{};
  std::string line{};
  while(std::getline(in, line)) {
    if(line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    NodeRecord node{};
    if(!(iss >> node.nodeId >> node.parentId >> node.scalar)) continue;
    if(idToIndex.count(node.nodeId) != 0)
      fail("Duplicate node id: " + std::to_string(node.nodeId));
    idToIndex[node.nodeId] = static_cast<int>(nodes.size());
    nodes.emplace_back(std::move(node));
  }
  if(nodes.empty()) fail("Tree txt is empty: `" + path + "`.");
  return nodes;
}

std::unordered_map<int, int> buildNodeIndex(const std::vector<NodeRecord> &nodes) {
  std::unordered_map<int, int> m{};
  m.reserve(nodes.size());
  for(size_t i = 0; i < nodes.size(); ++i)
    m[nodes[i].nodeId] = static_cast<int>(i);
  return m;
}

void attachGeometryFromImage(std::vector<NodeRecord> &nodes, vtkImageData *image) {
  if(image == nullptr) fail("Image is null.");
  const auto pointCount = image->GetNumberOfPoints();
  for(auto &node : nodes) {
    if(node.nodeId < 0 || node.nodeId >= pointCount)
      fail("Node id " + std::to_string(node.nodeId) + " outside VTI range.");
    image->GetPoint(node.nodeId, node.point);
  }
}

int classifyCriticalType(const NodeRecord &node) {
  if(node.parentId == node.nodeId || node.parentId < 0) return 3;
  if(node.children.empty()) return 0;
  if(node.children.size() > 1) return 2;
  return 1;
}

void buildTreeRelations(std::vector<NodeRecord> &nodes,
                        const std::unordered_map<int, int> &nodeIdToIndex) {
  for(auto &node : nodes) { node.children.clear(); node.incidentArcIds.clear(); }
  int rootCount = 0;
  for(auto &node : nodes) {
    if(node.parentId == node.nodeId || node.parentId < 0) { ++rootCount; continue; }
    const auto parentIt = nodeIdToIndex.find(node.parentId);
    if(parentIt == nodeIdToIndex.end())
      fail("Parent id " + std::to_string(node.parentId) + " not found.");
    nodes[parentIt->second].children.emplace_back(node.nodeId);
  }
  if(rootCount != 1)
    fail("Expected one root, got " + std::to_string(rootCount) + ".");
  for(auto &node : nodes) {
    node.degree = static_cast<int>(node.children.size())
                  + ((node.parentId == node.nodeId || node.parentId < 0) ? 0 : 1);
    node.criticalType = classifyCriticalType(node);
  }
}

void computeSubtreeData(std::vector<NodeRecord> &nodes,
                        const std::unordered_map<int, int> &nodeIdToIndex) {
  int rootIndex = -1;
  for(size_t i = 0; i < nodes.size(); ++i) {
    if(nodes[i].parentId == nodes[i].nodeId || nodes[i].parentId < 0) {
      rootIndex = static_cast<int>(i); break;
    }
  }
  if(rootIndex < 0) fail("Root not found.");
  std::vector<int> order{};
  order.reserve(nodes.size());
  std::vector<int> stack{rootIndex};
  while(!stack.empty()) {
    const int idx = stack.back(); stack.pop_back();
    order.emplace_back(idx);
    for(const auto childId : nodes[idx].children)
      stack.emplace_back(nodeIdToIndex.at(childId));
  }
  for(auto it = order.rbegin(); it != order.rend(); ++it) {
    auto &node = nodes[*it];
    node.subtreeVertices.clear();
    node.subtreeVertices.emplace_back(node.nodeId);
    for(const auto childId : node.children) {
      const auto &child = nodes[nodeIdToIndex.at(childId)];
      node.subtreeVertices.insert(node.subtreeVertices.end(),
        child.subtreeVertices.begin(), child.subtreeVertices.end());
    }
    std::sort(node.subtreeVertices.begin(), node.subtreeVertices.end());
    node.subtreeVertices.erase(
      std::unique(node.subtreeVertices.begin(), node.subtreeVertices.end()),
      node.subtreeVertices.end());
    node.regionSize = static_cast<double>(node.subtreeVertices.size());
  }
}

std::vector<ArcRecord> buildArcs(std::vector<NodeRecord> &nodes,
                                 const std::unordered_map<int, int> &nodeIdToIndex) {
  std::vector<ArcRecord> arcs{};
  arcs.reserve(nodes.size() > 0 ? nodes.size() - 1 : 0);
  for(auto &node : nodes) {
    if(node.parentId == node.nodeId || node.parentId < 0) continue;
    const auto parentIndex = nodeIdToIndex.at(node.parentId);
    ArcRecord arc{};
    arc.arcId = static_cast<int>(arcs.size());
    arc.downNodeId = node.nodeId;
    arc.upNodeId = node.parentId;
    arc.regionSize = node.regionSize;
    arc.regionSpan = std::abs(nodes[parentIndex].scalar - node.scalar);
    arcs.emplace_back(arc);
    node.incidentArcIds.emplace_back(arc.arcId);
    nodes[parentIndex].incidentArcIds.emplace_back(arc.arcId);
    node.regionSpan = arc.regionSpan;
  }
  return arcs;
}

int otherNodeId(const ArcRecord &arc, int nodeId) {
  if(arc.downNodeId == nodeId) return arc.upNodeId;
  if(arc.upNodeId == nodeId) return arc.downNodeId;
  fail("Arc/node mismatch.");
}

void orientBranch(std::vector<int> &nodeIds, std::vector<int> &arcIds,
                  const std::unordered_map<int, int> &nodeIdToIndex,
                  const std::vector<NodeRecord> &nodes, int treeType) {
  if(nodeIds.size() < 2) return;
  const auto firstScalar = nodes[nodeIdToIndex.at(nodeIds.front())].scalar;
  const auto lastScalar = nodes[nodeIdToIndex.at(nodeIds.back())].scalar;
  const bool shouldReverse = (treeType == 0 && firstScalar > lastScalar)
                             || (treeType != 0 && firstScalar < lastScalar);
  if(shouldReverse) {
    std::reverse(nodeIds.begin(), nodeIds.end());
    std::reverse(arcIds.begin(), arcIds.end());
  }
}

std::vector<BranchRecord> extractBranches(const std::vector<NodeRecord> &nodes,
                                          const std::vector<ArcRecord> &arcs,
                                          const std::unordered_map<int, int> &nodeIdToIndex,
                                          int treeType) {
  std::vector<BranchRecord> branches{};
  std::vector<char> arcVisited(arcs.size(), 0);
  auto isEndpoint = [&](int nodeId) {
    return nodes[nodeIdToIndex.at(nodeId)].incidentArcIds.size() != 2;
  };
  int branchId = 0;
  for(const auto &node : nodes) {
    if(!isEndpoint(node.nodeId)) continue;
    for(const auto arcId : node.incidentArcIds) {
      if(arcVisited[arcId] != 0) continue;
      BranchRecord branch{};
      branch.branchId = branchId++;
      branch.nodeIds.emplace_back(node.nodeId);
      int currentNodeId = node.nodeId;
      int currentArcId = arcId;
      while(true) {
        arcVisited[currentArcId] = 1;
        branch.arcIds.emplace_back(currentArcId);
        const auto nextNodeId = otherNodeId(arcs[currentArcId], currentNodeId);
        branch.nodeIds.emplace_back(nextNodeId);
        if(isEndpoint(nextNodeId)) break;
        const auto &nextNode = nodes[nodeIdToIndex.at(nextNodeId)];
        int nextArcId = -1;
        for(const auto cand : nextNode.incidentArcIds) {
          if(cand != currentArcId) { nextArcId = cand; break; }
        }
        if(nextArcId < 0) fail("Failed to continue branch traversal.");
        currentNodeId = nextNodeId;
        currentArcId = nextArcId;
      }
      orientBranch(branch.nodeIds, branch.arcIds, nodeIdToIndex, nodes, treeType);
      branches.emplace_back(std::move(branch));
    }
  }
  return branches;
}

double computeIoU(const std::vector<int> &a, const std::vector<int> &b) {
  size_t i = 0, j = 0, intersection = 0;
  while(i < a.size() && j < b.size()) {
    if(a[i] == b[j]) { ++intersection; ++i; ++j; }
    else if(a[i] < b[j]) ++i;
    else ++j;
  }
  const size_t unionSize = a.size() + b.size() - intersection;
  if(unionSize == 0) return 1.0;
  return static_cast<double>(intersection) / static_cast<double>(unionSize);
}

std::vector<int> buildBranchLocalVolume(const BranchRecord &branch,
                                        size_t nodeOffset,
                                        const std::vector<NodeRecord> &nodes,
                                        const std::unordered_map<int, int> &nodeIdToIndex,
                                        int treeType) {
  const auto threshold = nodes[nodeIdToIndex.at(branch.nodeIds.at(nodeOffset))].scalar;
  std::vector<int> volume{};
  for(const auto nodeId : branch.nodeIds) {
    const auto scalar = nodes[nodeIdToIndex.at(nodeId)].scalar;
    const bool inside = treeType == 0 ? scalar <= threshold : scalar >= threshold;
    if(inside) volume.emplace_back(nodeId);
  }
  std::sort(volume.begin(), volume.end());
  volume.erase(std::unique(volume.begin(), volume.end()), volume.end());
  return volume;
}

std::vector<int> buildBranchLocalVolumeAtScalar(
    const std::vector<int> &branchNodeIds, double threshold,
    const std::vector<NodeRecord> &nodes,
    const std::unordered_map<int, int> &nodeIdToIndex, int treeType) {
  std::vector<int> volume{};
  for(const auto nodeId : branchNodeIds) {
    const auto scalar = nodes[nodeIdToIndex.at(nodeId)].scalar;
    const bool inside = treeType == 0 ? scalar <= threshold : scalar >= threshold;
    if(inside) volume.emplace_back(nodeId);
  }
  std::sort(volume.begin(), volume.end());
  volume.erase(std::unique(volume.begin(), volume.end()), volume.end());
  return volume;
}

void simplifyBranchByVolumeIoU(BranchRecord &branch,
                               const std::vector<NodeRecord> &nodes,
                               const std::unordered_map<int, int> &nodeIdToIndex,
                               double eps, int treeType) {
  branch.keepIndices.clear();
  if(branch.nodeIds.empty()) return;
  branch.keepIndices.emplace_back(0);
  int refIndex = 0;
  auto refVolume = buildBranchLocalVolume(
    branch, static_cast<size_t>(refIndex), nodes, nodeIdToIndex, treeType);
  for(size_t i = 1; i + 1 < branch.nodeIds.size(); ++i) {
    const auto curVolume = buildBranchLocalVolume(branch, i, nodes, nodeIdToIndex, treeType);
    const auto iou = computeIoU(refVolume, curVolume);
    if(1.0 - iou >= eps) {
      branch.keepIndices.emplace_back(static_cast<int>(i));
      refIndex = static_cast<int>(i);
      refVolume = curVolume;
    }
  }
  const int lastIndex = static_cast<int>(branch.nodeIds.size() - 1);
  if(branch.keepIndices.back() != lastIndex)
    branch.keepIndices.emplace_back(lastIndex);
}

std::vector<int> findMainChain(const std::vector<NodeRecord> &nodes,
                               const std::unordered_map<int, int> &nodeIdToIndex) {
  int rootIndex = -1;
  for(size_t i = 0; i < nodes.size(); ++i) {
    if(nodes[i].parentId == nodes[i].nodeId || nodes[i].parentId < 0) {
      rootIndex = static_cast<int>(i); break;
    }
  }
  if(rootIndex < 0) fail("Root not found for main chain.");
  double rootScalar = nodes[rootIndex].scalar;
  int bestLeafIndex = rootIndex;
  double bestSpan = 0.0;
  for(size_t i = 0; i < nodes.size(); ++i) {
    if(nodes[i].children.empty()) {
      double span = std::abs(nodes[i].scalar - rootScalar);
      if(span > bestSpan) { bestSpan = span; bestLeafIndex = static_cast<int>(i); }
    }
  }
  std::vector<int> chain{};
  int cur = bestLeafIndex;
  while(true) {
    chain.emplace_back(nodes[cur].nodeId);
    if(cur == rootIndex) break;
    cur = nodeIdToIndex.at(nodes[cur].parentId);
  }
  return chain;
}

std::vector<BranchInfo> computeBranchInfo(
    const std::vector<BranchRecord> &branches,
    const std::vector<NodeRecord> &nodes,
    const std::unordered_map<int, int> &nodeIdToIndex) {
  std::vector<BranchInfo> infos{};
  infos.reserve(branches.size());
  for(const auto &branch : branches) {
    BranchInfo info{};
    info.branchId = branch.branchId;
    info.nodeIds = branch.nodeIds;
    info.scalarMin = std::numeric_limits<double>::max();
    info.scalarMax = std::numeric_limits<double>::lowest();
    for(const auto nodeId : branch.nodeIds) {
      double s = nodes[nodeIdToIndex.at(nodeId)].scalar;
      if(s < info.scalarMin) info.scalarMin = s;
      if(s > info.scalarMax) info.scalarMax = s;
    }
    info.persistence = info.scalarMax - info.scalarMin;
    infos.emplace_back(std::move(info));
  }
  return infos;
}

std::vector<int> findActiveBranchIndices(double scalarLevel,
                                         const std::vector<BranchInfo> &branchInfos) {
  std::vector<int> active{};
  for(size_t i = 0; i < branchInfos.size(); ++i) {
    if(scalarLevel >= branchInfos[i].scalarMin &&
       scalarLevel <= branchInfos[i].scalarMax) {
      active.emplace_back(static_cast<int>(i));
    }
  }
  return active;
}

double computeWeightedDistance(
    double refScalar, double candidateScalar,
    const std::vector<int> &activeBranchIndices,
    const std::vector<BranchInfo> &branchInfos,
    const std::vector<NodeRecord> &nodes,
    const std::unordered_map<int, int> &nodeIdToIndex,
    int treeType, double alpha) {

  const int n = static_cast<int>(activeBranchIndices.size());
  if(n == 0) return 0.0;

  std::vector<double> distances(n, 0.0);
  std::vector<double> geoWeights(n, 0.0);
  std::vector<double> topoWeights(n, 0.0);

  double totalPersistence = 0.0;
  for(int k = 0; k < n; ++k)
    totalPersistence += branchInfos[activeBranchIndices[k]].persistence;

  for(int k = 0; k < n; ++k) {
    const auto &bi = branchInfos[activeBranchIndices[k]];
    topoWeights[k] = (totalPersistence > 0.0)
                       ? bi.persistence / totalPersistence
                       : 1.0 / n;

    auto refVol = buildBranchLocalVolumeAtScalar(
      bi.nodeIds, refScalar, nodes, nodeIdToIndex, treeType);
    auto candVol = buildBranchLocalVolumeAtScalar(
      bi.nodeIds, candidateScalar, nodes, nodeIdToIndex, treeType);

    double iou = computeIoU(refVol, candVol);
    distances[k] = 1.0 - iou;

    double refSize = static_cast<double>(refVol.size());
    double candSize = static_cast<double>(candVol.size());
    double maxSize = std::max(refSize, candSize);
    geoWeights[k] = (maxSize > 0.0)
                      ? std::abs(refSize - candSize) / maxSize
                      : 0.0;
  }

  double totalGeo = 0.0;
  for(int k = 0; k < n; ++k) totalGeo += geoWeights[k];
  if(totalGeo > 0.0) {
    for(int k = 0; k < n; ++k) geoWeights[k] /= totalGeo;
  } else {
    for(int k = 0; k < n; ++k) geoWeights[k] = 1.0 / n;
  }

  double totalWeight = 0.0;
  double weightedDist = 0.0;
  for(int k = 0; k < n; ++k) {
    double w = alpha * topoWeights[k] + (1.0 - alpha) * geoWeights[k];
    totalWeight += w;
    weightedDist += w * distances[k];
  }
  if(totalWeight > 0.0) weightedDist /= totalWeight;
  return weightedDist;
}

std::unordered_set<int> simplifyWithMultiBranch(
    const std::vector<int> &mainChain,
    const std::vector<BranchInfo> &branchInfos,
    std::vector<BranchRecord> &branches,
    const std::vector<NodeRecord> &nodes,
    const std::unordered_map<int, int> &nodeIdToIndex,
    double eps, double alpha, int treeType) {

  std::unordered_set<int> mainChainSet(mainChain.begin(), mainChain.end());
  std::unordered_set<int> keptNodes{};

  // Always keep structural nodes (leaves, bifurcations, root)
  for(const auto &node : nodes) {
    if(node.criticalType != 1) keptNodes.insert(node.nodeId);
  }

  // Simplify main chain with multi-branch weighting
  if(mainChain.size() > 2) {
    double refScalar = nodes[nodeIdToIndex.at(mainChain[0])].scalar;
    keptNodes.insert(mainChain[0]);
    for(size_t i = 1; i + 1 < mainChain.size(); ++i) {
      double candidateScalar = nodes[nodeIdToIndex.at(mainChain[i])].scalar;
      auto active = findActiveBranchIndices(candidateScalar, branchInfos);
      if(active.empty()) continue;
      double D = computeWeightedDistance(refScalar, candidateScalar, active,
                                         branchInfos, nodes, nodeIdToIndex,
                                         treeType, alpha);
      if(D >= eps) {
        keptNodes.insert(mainChain[i]);
        refScalar = candidateScalar;
      }
    }
    keptNodes.insert(mainChain.back());
  } else {
    for(const auto nid : mainChain) keptNodes.insert(nid);
  }

  // Simplify non-main-chain branches independently with IoU
  for(auto &branch : branches) {
    bool onMainChain = true;
    for(const auto nid : branch.nodeIds) {
      if(mainChainSet.find(nid) == mainChainSet.end()) {
        onMainChain = false; break;
      }
    }
    if(!onMainChain) {
      simplifyBranchByVolumeIoU(branch, nodes, nodeIdToIndex, eps, treeType);
      for(const auto idx : branch.keepIndices)
        keptNodes.insert(branch.nodeIds[idx]);
    }
  }

  return keptNodes;
}

SimplifiedTree rebuildFromKeptNodes(
    const std::unordered_set<int> &keptNodes,
    const std::vector<NodeRecord> &nodes,
    const std::vector<ArcRecord> &arcs,
    const std::vector<BranchRecord> &branches,
    const std::unordered_map<int, int> &nodeIdToIndex) {

  SimplifiedTree tree{};
  tree.keptNodeIds.assign(keptNodes.begin(), keptNodes.end());
  std::sort(tree.keptNodeIds.begin(), tree.keptNodeIds.end(),
    [&](int a, int b) {
      return nodes.at(nodeIdToIndex.at(a)).scalar
             < nodes.at(nodeIdToIndex.at(b)).scalar;
    });

  for(const auto &node : nodes)
    tree.nodeKeepMask[node.nodeId] = keptNodes.count(node.nodeId) > 0 ? 1 : 0;

  int newArcId = 0;
  for(const auto &branch : branches) {
    std::vector<int> branchKeptIndices{};
    for(size_t i = 0; i < branch.nodeIds.size(); ++i) {
      if(keptNodes.count(branch.nodeIds[i]) > 0)
        branchKeptIndices.emplace_back(static_cast<int>(i));
    }
    for(size_t i = 1; i < branchKeptIndices.size(); ++i) {
      const int startIdx = branchKeptIndices[i - 1];
      const int endIdx = branchKeptIndices[i];
      NewArcRecord newArc{};
      newArc.newArcId = newArcId++;
      newArc.downNodeId = branch.nodeIds[startIdx];
      newArc.upNodeId = branch.nodeIds[endIdx];
      newArc.regionSpan = std::abs(
        nodes.at(nodeIdToIndex.at(newArc.upNodeId)).scalar
        - nodes.at(nodeIdToIndex.at(newArc.downNodeId)).scalar);
      for(int off = startIdx; off < endIdx; ++off) {
        const int oldArcId = branch.arcIds[off];
        newArc.regionSize += arcs[oldArcId].regionSize;
        newArc.mergedArcIds.emplace_back(oldArcId);
      }
      tree.newArcs.emplace_back(std::move(newArc));
    }
  }
  return tree;
}

vtkSmartPointer<vtkUnstructuredGrid>
buildSimplifiedNodeGrid(const SimplifiedTree &tree,
                        const std::vector<NodeRecord> &nodes,
                        const std::unordered_map<int, int> &nodeIdToIndex) {
  vtkNew<vtkUnstructuredGrid> grid{};
  vtkNew<vtkPoints> points{};
  vtkNew<vtkCellArray> cells{};
  vtkNew<vtkIntArray> nodeIdArray{};
  vtkNew<vtkDoubleArray> scalarArray{};
  vtkNew<vtkIntArray> criticalTypeArray{};
  vtkNew<vtkDoubleArray> regionSizeArray{};
  vtkNew<vtkDoubleArray> regionSpanArray{};
  vtkNew<vtkIntArray> subtreeSizeArray{};

  nodeIdArray->SetName("NodeId");
  scalarArray->SetName("Scalar");
  criticalTypeArray->SetName("CriticalType");
  regionSizeArray->SetName("RegionSize");
  regionSpanArray->SetName("RegionSpan");
  subtreeSizeArray->SetName("SubtreeSize");

  for(const auto nodeId : tree.keptNodeIds) {
    const auto &node = nodes.at(nodeIdToIndex.at(nodeId));
    const auto pointId = points->InsertNextPoint(node.point);
    vtkNew<vtkVertex> vertex{};
    vertex->GetPointIds()->SetId(0, pointId);
    cells->InsertNextCell(vertex);
    nodeIdArray->InsertNextValue(node.nodeId);
    scalarArray->InsertNextValue(node.scalar);
    criticalTypeArray->InsertNextValue(node.criticalType);
    regionSizeArray->InsertNextValue(node.regionSize);
    regionSpanArray->InsertNextValue(node.regionSpan);
    subtreeSizeArray->InsertNextValue(
      static_cast<int>(node.subtreeVertices.size()));
  }

  grid->SetPoints(points);
  grid->SetCells(VTK_VERTEX, cells);
  grid->GetPointData()->AddArray(nodeIdArray);
  grid->GetPointData()->AddArray(scalarArray);
  grid->GetPointData()->AddArray(criticalTypeArray);
  grid->GetPointData()->AddArray(regionSizeArray);
  grid->GetPointData()->AddArray(regionSpanArray);
  grid->GetPointData()->AddArray(subtreeSizeArray);
  grid->GetPointData()->SetScalars(scalarArray);
  return grid;
}

vtkSmartPointer<vtkUnstructuredGrid>
buildSimplifiedArcGrid(const SimplifiedTree &tree,
                       const std::vector<NodeRecord> &nodes,
                       const std::unordered_map<int, int> &nodeIdToIndex) {
  vtkNew<vtkUnstructuredGrid> grid{};
  vtkNew<vtkPoints> points{};
  vtkNew<vtkCellArray> cells{};
  vtkNew<vtkIntArray> pointNodeIdArray{};
  vtkNew<vtkDoubleArray> pointScalarArray{};
  vtkNew<vtkIntArray> newArcIdArray{};
  vtkNew<vtkIntArray> downNodeIdArray{};
  vtkNew<vtkIntArray> upNodeIdArray{};
  vtkNew<vtkDoubleArray> regionSizeArray{};
  vtkNew<vtkDoubleArray> regionSpanArray{};
  vtkNew<vtkIntArray> mergedArcCountArray{};

  pointNodeIdArray->SetName("NodeId");
  pointScalarArray->SetName("Scalar");
  newArcIdArray->SetName("NewArcId");
  downNodeIdArray->SetName("downNodeId");
  upNodeIdArray->SetName("upNodeId");
  regionSizeArray->SetName("RegionSize");
  regionSpanArray->SetName("RegionSpan");
  mergedArcCountArray->SetName("MergedArcCount");

  std::unordered_map<int, vtkIdType> nodePointIds{};
  for(const auto nodeId : tree.keptNodeIds) {
    const auto &node = nodes.at(nodeIdToIndex.at(nodeId));
    nodePointIds[nodeId] = points->InsertNextPoint(node.point);
    pointNodeIdArray->InsertNextValue(node.nodeId);
    pointScalarArray->InsertNextValue(node.scalar);
  }

  for(const auto &newArc : tree.newArcs) {
    vtkIdType ids[2]{nodePointIds.at(newArc.downNodeId),
                     nodePointIds.at(newArc.upNodeId)};
    cells->InsertNextCell(2, ids);
    newArcIdArray->InsertNextValue(newArc.newArcId);
    downNodeIdArray->InsertNextValue(newArc.downNodeId);
    upNodeIdArray->InsertNextValue(newArc.upNodeId);
    regionSizeArray->InsertNextValue(newArc.regionSize);
    regionSpanArray->InsertNextValue(newArc.regionSpan);
    mergedArcCountArray->InsertNextValue(
      static_cast<int>(newArc.mergedArcIds.size()));
  }

  grid->SetPoints(points);
  grid->SetCells(VTK_LINE, cells);
  grid->GetPointData()->AddArray(pointNodeIdArray);
  grid->GetPointData()->AddArray(pointScalarArray);
  grid->GetCellData()->AddArray(newArcIdArray);
  grid->GetCellData()->AddArray(downNodeIdArray);
  grid->GetCellData()->AddArray(upNodeIdArray);
  grid->GetCellData()->AddArray(regionSizeArray);
  grid->GetCellData()->AddArray(regionSpanArray);
  grid->GetCellData()->AddArray(mergedArcCountArray);
  grid->GetCellData()->SetScalars(regionSpanArray);
  return grid;
}

vtkSmartPointer<vtkUnstructuredGrid>
buildNodeKeepMaskGrid(const std::vector<NodeRecord> &nodes,
                      const SimplifiedTree &tree) {
  vtkNew<vtkUnstructuredGrid> grid{};
  vtkNew<vtkPoints> points{};
  vtkNew<vtkCellArray> cells{};
  vtkNew<vtkIntArray> nodeIdArray{};
  vtkNew<vtkDoubleArray> scalarArray{};
  vtkNew<vtkIntArray> isKeptArray{};

  nodeIdArray->SetName("NodeId");
  scalarArray->SetName("Scalar");
  isKeptArray->SetName("IsKept");

  for(const auto &node : nodes) {
    const auto pointId = points->InsertNextPoint(node.point);
    vtkNew<vtkVertex> vertex{};
    vertex->GetPointIds()->SetId(0, pointId);
    cells->InsertNextCell(vertex);
    nodeIdArray->InsertNextValue(node.nodeId);
    scalarArray->InsertNextValue(node.scalar);
    const auto it = tree.nodeKeepMask.find(node.nodeId);
    isKeptArray->InsertNextValue(it == tree.nodeKeepMask.end() ? 0 : it->second);
  }

  grid->SetPoints(points);
  grid->SetCells(VTK_VERTEX, cells);
  grid->GetPointData()->AddArray(nodeIdArray);
  grid->GetPointData()->AddArray(scalarArray);
  grid->GetPointData()->AddArray(isKeptArray);
  grid->GetPointData()->SetScalars(isKeptArray);
  return grid;
}

void writeGrid(vtkUnstructuredGrid *grid, const std::string &path) {
  vtkNew<vtkXMLUnstructuredGridWriter> writer{};
  writer->SetFileName(path.c_str());
  writer->SetInputData(grid);
  if(writer->Write() == 0) fail("Failed to write `" + path + "`.");
}

void writeStats(const std::string &path,
                const std::vector<NodeRecord> &nodes,
                const std::vector<ArcRecord> &arcs,
                const std::vector<BranchRecord> &branches,
                const SimplifiedTree &tree,
                const std::vector<int> &mainChain,
                double eps, double alpha) {
  std::ofstream out(path);
  if(!out) fail("Failed to open stats file `" + path + "`.");
  out << "mode=txt_tree_multi_branch_weighted_iou\n";
  out << "eps=" << std::setprecision(6) << eps << '\n';
  out << "alpha=" << std::setprecision(6) << alpha << '\n';
  out << "original_nodes=" << nodes.size() << '\n';
  out << "kept_nodes=" << tree.keptNodeIds.size() << '\n';
  out << "original_arcs=" << arcs.size() << '\n';
  out << "simplified_arcs=" << tree.newArcs.size() << '\n';
  out << "branches=" << branches.size() << '\n';
  out << "main_chain_length=" << mainChain.size() << '\n';
  out << '\n';
  for(const auto &branch : branches) {
    int kept = 0;
    for(const auto nid : branch.nodeIds) {
      if(tree.nodeKeepMask.count(nid) && tree.nodeKeepMask.at(nid) == 1) ++kept;
    }
    out << "branch=" << branch.branchId
        << ", nodes=" << branch.nodeIds.size()
        << ", kept=" << kept << '\n';
  }
}

std::string joinPath(const std::string &prefix, const std::string &suffix) {
  return prefix + suffix;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opt = parseArgs(argc, argv);
    std::filesystem::create_directories(
      std::filesystem::path(opt.outputPrefix).parent_path());

    auto image = readImage(opt.imagePath);
    auto nodes = readTreeTxt(opt.treePath);
    const auto nodeIdToIndex = buildNodeIndex(nodes);
    attachGeometryFromImage(nodes, image);
    buildTreeRelations(nodes, nodeIdToIndex);
    computeSubtreeData(nodes, nodeIdToIndex);
    auto arcs = buildArcs(nodes, nodeIdToIndex);
    auto branches = extractBranches(nodes, arcs, nodeIdToIndex, opt.treeType);

    auto mainChain = findMainChain(nodes, nodeIdToIndex);
    auto branchInfos = computeBranchInfo(branches, nodes, nodeIdToIndex);

    std::cout << "Main chain length: " << mainChain.size() << " nodes\n";
    std::cout << "Total branches: " << branches.size() << '\n';
    std::cout << "Alpha (topo/geo mix): " << opt.alpha << '\n';

    auto keptNodes = simplifyWithMultiBranch(
      mainChain, branchInfos, branches, nodes, nodeIdToIndex,
      opt.eps, opt.alpha, opt.treeType);

    const auto simplifiedTree = rebuildFromKeptNodes(
      keptNodes, nodes, arcs, branches, nodeIdToIndex);

    auto nodeGrid = buildSimplifiedNodeGrid(simplifiedTree, nodes, nodeIdToIndex);
    auto arcGrid = buildSimplifiedArcGrid(simplifiedTree, nodes, nodeIdToIndex);
    auto maskGrid = buildNodeKeepMaskGrid(nodes, simplifiedTree);

    writeGrid(nodeGrid, joinPath(opt.outputPrefix, "_nodes.vtu"));
    writeGrid(arcGrid, joinPath(opt.outputPrefix, "_arcs.vtu"));
    writeGrid(maskGrid, joinPath(opt.outputPrefix, "_node_mask.vtu"));
    writeStats(joinPath(opt.outputPrefix, "_stats.txt"),
               nodes, arcs, branches, simplifiedTree, mainChain,
               opt.eps, opt.alpha);

    std::cout << "Simplification finished. Kept "
              << simplifiedTree.keptNodeIds.size() << " / " << nodes.size()
              << " nodes.\n";
  } catch(const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
