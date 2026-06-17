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
  std::string outputPrefix{"idea_iou/result/output"};
  double eps{0.25};
  int treeType{0};
};

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error(msg);
}

void printUsage() {
  std::cerr
    << "Usage: treeTxtIoUSimplifier -t <tree.txt> -s <segmentation.vti> "
    << "[-o <output_prefix>] [-e <eps>] [-T <0|1>]\n";
}

ProgramOptions parseArgs(int argc, char **argv) {
  ProgramOptions opt{};
  for(int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto requireValue = [&](const std::string &name) -> std::string {
      if(i + 1 >= argc) {
        fail("Missing value for " + name + ".");
      }
      return argv[++i];
    };

    if(arg == "-t") {
      opt.treePath = requireValue(arg);
    } else if(arg == "-s") {
      opt.imagePath = requireValue(arg);
    } else if(arg == "-o") {
      opt.outputPrefix = requireValue(arg);
    } else if(arg == "-e") {
      opt.eps = std::stod(requireValue(arg));
    } else if(arg == "-T") {
      opt.treeType = std::stoi(requireValue(arg));
    } else if(arg == "-h" || arg == "--help") {
      printUsage();
      std::exit(0);
    } else {
      fail("Unknown argument: " + arg);
    }
  }

  if(opt.treePath.empty() || opt.imagePath.empty()) {
    printUsage();
    fail("Both -t and -s are required.");
  }
  if(opt.eps < 0.0 || opt.eps > 1.0) {
    fail("eps must be in [0, 1].");
  }
  if(opt.treeType != 0 && opt.treeType != 1) {
    fail("treeType must be 0 (JT) or 1 (ST).");
  }
  return opt;
}

vtkSmartPointer<vtkImageData> readImage(const std::string &path) {
  vtkNew<vtkXMLImageDataReader> reader{};
  reader->SetFileName(path.c_str());
  reader->Update();
  auto *image = reader->GetOutput();
  if(image == nullptr) {
    fail("Unable to read segmentation image `" + path + "`.");
  }
  auto out = vtkSmartPointer<vtkImageData>::New();
  out->DeepCopy(image);
  return out;
}

std::vector<NodeRecord> readTreeTxt(const std::string &path) {
  std::ifstream in(path);
  if(!in) {
    fail("Unable to open tree file `" + path + "`.");
  }

  std::vector<NodeRecord> nodes{};
  std::unordered_map<int, int> idToIndex{};
  std::string line{};
  while(std::getline(in, line)) {
    if(line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    NodeRecord node{};
    if(!(iss >> node.nodeId >> node.parentId >> node.scalar)) {
      continue;
    }
    if(idToIndex.count(node.nodeId) != 0) {
      fail("Duplicate node id in txt tree: " + std::to_string(node.nodeId));
    }
    idToIndex[node.nodeId] = static_cast<int>(nodes.size());
    nodes.emplace_back(std::move(node));
  }

  if(nodes.empty()) {
    fail("Tree txt is empty: `" + path + "`.");
  }
  return nodes;
}

std::unordered_map<int, int> buildNodeIndex(const std::vector<NodeRecord> &nodes) {
  std::unordered_map<int, int> nodeIdToIndex{};
  nodeIdToIndex.reserve(nodes.size());
  for(size_t i = 0; i < nodes.size(); ++i) {
    nodeIdToIndex[nodes[i].nodeId] = static_cast<int>(i);
  }
  return nodeIdToIndex;
}

void attachGeometryFromImage(std::vector<NodeRecord> &nodes, vtkImageData *image) {
  if(image == nullptr) {
    fail("Image is null.");
  }
  const auto pointCount = image->GetNumberOfPoints();
  for(auto &node : nodes) {
    if(node.nodeId < 0 || node.nodeId >= pointCount) {
      fail("Node id " + std::to_string(node.nodeId)
           + " is outside VTI point range [0, "
           + std::to_string(pointCount - 1) + "].");
    }
    image->GetPoint(node.nodeId, node.point);
  }
}

int classifyCriticalType(const NodeRecord &node) {
  if(node.parentId == node.nodeId || node.parentId < 0) {
    return 3;
  }
  if(node.children.empty()) {
    return 0;
  }
  if(node.children.size() > 1) {
    return 2;
  }
  return 1;
}

void buildTreeRelations(std::vector<NodeRecord> &nodes,
                        const std::unordered_map<int, int> &nodeIdToIndex) {
  for(auto &node : nodes) {
    node.children.clear();
    node.incidentArcIds.clear();
  }

  int rootCount = 0;
  for(auto &node : nodes) {
    if(node.parentId == node.nodeId || node.parentId < 0) {
      ++rootCount;
      continue;
    }
    const auto parentIt = nodeIdToIndex.find(node.parentId);
    if(parentIt == nodeIdToIndex.end()) {
      fail("Parent id " + std::to_string(node.parentId)
           + " not found for node " + std::to_string(node.nodeId) + ".");
    }
    nodes[parentIt->second].children.emplace_back(node.nodeId);
  }

  if(rootCount != 1) {
    fail("Expected exactly one root in txt tree, got " + std::to_string(rootCount)
         + ".");
  }

  for(auto &node : nodes) {
    node.degree = static_cast<int>(node.children.size())
                  + ((node.parentId == node.nodeId || node.parentId < 0) ? 0 : 1);
    node.criticalType = classifyCriticalType(node);
  }
}

void computeSubtreeData(std::vector<NodeRecord> &nodes,
                        const std::unordered_map<int, int> &nodeIdToIndex) {
  std::vector<int> order{};
  order.reserve(nodes.size());
  int rootIndex = -1;
  for(size_t i = 0; i < nodes.size(); ++i) {
    if(nodes[i].parentId == nodes[i].nodeId || nodes[i].parentId < 0) {
      rootIndex = static_cast<int>(i);
      break;
    }
  }
  if(rootIndex < 0) {
    fail("Root not found.");
  }

  std::vector<int> stack{rootIndex};
  while(!stack.empty()) {
    const int nodeIndex = stack.back();
    stack.pop_back();
    order.emplace_back(nodeIndex);
    for(const auto childId : nodes[nodeIndex].children) {
      stack.emplace_back(nodeIdToIndex.at(childId));
    }
  }

  if(order.size() != nodes.size()) {
    fail("Tree traversal did not visit all nodes.");
  }

  for(auto it = order.rbegin(); it != order.rend(); ++it) {
    auto &node = nodes[*it];
    node.subtreeVertices.clear();
    node.subtreeVertices.emplace_back(node.nodeId);
    for(const auto childId : node.children) {
      const auto &child = nodes[nodeIdToIndex.at(childId)];
      node.subtreeVertices.insert(node.subtreeVertices.end(),
                                  child.subtreeVertices.begin(),
                                  child.subtreeVertices.end());
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
    if(node.parentId == node.nodeId || node.parentId < 0) {
      continue;
    }
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
  if(arc.downNodeId == nodeId) {
    return arc.upNodeId;
  }
  if(arc.upNodeId == nodeId) {
    return arc.downNodeId;
  }
  fail("Arc/node mismatch.");
}

void orientBranch(std::vector<int> &nodeIds,
                  std::vector<int> &arcIds,
                  const std::unordered_map<int, int> &nodeIdToIndex,
                  const std::vector<NodeRecord> &nodes,
                  int treeType) {
  if(nodeIds.size() < 2) {
    return;
  }
  const auto firstScalar = nodes[nodeIdToIndex.at(nodeIds.front())].scalar;
  const auto lastScalar = nodes[nodeIdToIndex.at(nodeIds.back())].scalar;

  const bool shouldReverse
    = (treeType == 0 && firstScalar > lastScalar)
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
    if(!isEndpoint(node.nodeId)) {
      continue;
    }

    for(const auto arcId : node.incidentArcIds) {
      if(arcVisited[arcId] != 0) {
        continue;
      }

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

        if(isEndpoint(nextNodeId)) {
          break;
        }

        const auto &nextNode = nodes[nodeIdToIndex.at(nextNodeId)];
        int nextArcId = -1;
        for(const auto candidateArcId : nextNode.incidentArcIds) {
          if(candidateArcId != currentArcId) {
            nextArcId = candidateArcId;
            break;
          }
        }
        if(nextArcId < 0) {
          fail("Failed to continue branch traversal.");
        }

        currentNodeId = nextNodeId;
        currentArcId = nextArcId;
      }

      orientBranch(branch.nodeIds, branch.arcIds, nodeIdToIndex, nodes, treeType);
      branches.emplace_back(std::move(branch));
    }
  }

  const auto unvisitedCount
    = static_cast<int>(std::count(arcVisited.begin(), arcVisited.end(), 0));
  if(unvisitedCount != 0) {
    fail(std::to_string(unvisitedCount) + " arcs were not assigned to branches.");
  }

  return branches;
}

double computeIoU(const std::vector<int> &a, const std::vector<int> &b) {
  size_t i = 0;
  size_t j = 0;
  size_t intersection = 0;
  while(i < a.size() && j < b.size()) {
    if(a[i] == b[j]) {
      ++intersection;
      ++i;
      ++j;
    } else if(a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  const size_t unionSize = a.size() + b.size() - intersection;
  if(unionSize == 0) {
    return 1.0;
  }
  return static_cast<double>(intersection) / static_cast<double>(unionSize);
}

std::vector<int> buildBranchLocalVolume(const BranchRecord &branch,
                                        size_t nodeOffset,
                                        const std::vector<NodeRecord> &nodes,
                                        const std::unordered_map<int, int> &nodeIdToIndex,
                                        int treeType) {
  const auto threshold
    = nodes[nodeIdToIndex.at(branch.nodeIds.at(nodeOffset))].scalar;

  std::vector<int> volume{};
  volume.reserve(branch.nodeIds.size());
  for(const auto nodeId : branch.nodeIds) {
    const auto scalar = nodes[nodeIdToIndex.at(nodeId)].scalar;
    const bool inside = treeType == 0 ? scalar <= threshold : scalar >= threshold;
    if(inside) {
      volume.emplace_back(nodeId);
    }
  }

  std::sort(volume.begin(), volume.end());
  volume.erase(std::unique(volume.begin(), volume.end()), volume.end());
  return volume;
}

void simplifyBranchByVolumeIoU(BranchRecord &branch,
                               const std::vector<NodeRecord> &nodes,
                               const std::unordered_map<int, int> &nodeIdToIndex,
                               double eps,
                               int treeType) {
  branch.keepIndices.clear();
  if(branch.nodeIds.empty()) {
    return;
  }
  branch.keepIndices.emplace_back(0);
  int refIndex = 0;
  auto refVolume = buildBranchLocalVolume(
    branch, static_cast<size_t>(refIndex), nodes, nodeIdToIndex, treeType);

  for(size_t i = 1; i + 1 < branch.nodeIds.size(); ++i) {
    const auto curVolume
      = buildBranchLocalVolume(branch, i, nodes, nodeIdToIndex, treeType);
    const auto iou = computeIoU(refVolume, curVolume);
    const auto dist = 1.0 - iou;
    if(dist >= eps) {
      branch.keepIndices.emplace_back(static_cast<int>(i));
      refIndex = static_cast<int>(i);
      refVolume = curVolume;
    }
  }

  const int lastIndex = static_cast<int>(branch.nodeIds.size() - 1);
  if(branch.keepIndices.back() != lastIndex) {
    branch.keepIndices.emplace_back(lastIndex);
  }
}

SimplifiedTree rebuildSimplifiedTree(const std::vector<BranchRecord> &branches,
                                     const std::vector<NodeRecord> &nodes,
                                     const std::vector<ArcRecord> &arcs,
                                     const std::unordered_map<int, int> &nodeIdToIndex) {
  SimplifiedTree tree{};
  std::unordered_set<int> kept{};

  for(const auto &branch : branches) {
    for(const auto keepIndex : branch.keepIndices) {
      kept.insert(branch.nodeIds.at(keepIndex));
    }
  }

  tree.keptNodeIds.assign(kept.begin(), kept.end());
  std::sort(tree.keptNodeIds.begin(),
            tree.keptNodeIds.end(),
            [&](int a, int b) {
              return nodes.at(nodeIdToIndex.at(a)).scalar
                     < nodes.at(nodeIdToIndex.at(b)).scalar;
            });

  for(const auto &node : nodes) {
    tree.nodeKeepMask[node.nodeId] = kept.count(node.nodeId) > 0 ? 1 : 0;
  }

  int newArcId = 0;
  for(const auto &branch : branches) {
    for(size_t i = 1; i < branch.keepIndices.size(); ++i) {
      const int startIndex = branch.keepIndices[i - 1];
      const int endIndex = branch.keepIndices[i];
      NewArcRecord newArc{};
      newArc.newArcId = newArcId++;
      newArc.downNodeId = branch.nodeIds[startIndex];
      newArc.upNodeId = branch.nodeIds[endIndex];
      newArc.regionSpan
        = std::abs(nodes.at(nodeIdToIndex.at(newArc.upNodeId)).scalar
                   - nodes.at(nodeIdToIndex.at(newArc.downNodeId)).scalar);
      for(int arcOffset = startIndex; arcOffset < endIndex; ++arcOffset) {
        const int oldArcId = branch.arcIds[arcOffset];
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
  vtkNew<vtkIntArray> vertexIdArray{};
  vtkNew<vtkDoubleArray> regionSizeArray{};
  vtkNew<vtkDoubleArray> regionSpanArray{};
  vtkNew<vtkIntArray> isKeptArray{};
  vtkNew<vtkIntArray> subtreeSizeArray{};

  nodeIdArray->SetName("NodeId");
  scalarArray->SetName("Scalar");
  criticalTypeArray->SetName("CriticalType");
  vertexIdArray->SetName("VertexId");
  regionSizeArray->SetName("RegionSize");
  regionSpanArray->SetName("RegionSpan");
  isKeptArray->SetName("IsKept");
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
    vertexIdArray->InsertNextValue(node.nodeId);
    regionSizeArray->InsertNextValue(node.regionSize);
    regionSpanArray->InsertNextValue(node.regionSpan);
    isKeptArray->InsertNextValue(1);
    subtreeSizeArray->InsertNextValue(
      static_cast<int>(node.subtreeVertices.size()));
  }

  grid->SetPoints(points);
  grid->SetCells(VTK_VERTEX, cells);
  grid->GetPointData()->AddArray(nodeIdArray);
  grid->GetPointData()->AddArray(scalarArray);
  grid->GetPointData()->AddArray(criticalTypeArray);
  grid->GetPointData()->AddArray(vertexIdArray);
  grid->GetPointData()->AddArray(regionSizeArray);
  grid->GetPointData()->AddArray(regionSpanArray);
  grid->GetPointData()->AddArray(isKeptArray);
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
  vtkNew<vtkIntArray> segmentationIdArray{};
  vtkNew<vtkIntArray> downNodeIdArray{};
  vtkNew<vtkIntArray> upNodeIdArray{};
  vtkNew<vtkDoubleArray> regionSizeArray{};
  vtkNew<vtkDoubleArray> regionSpanArray{};
  vtkNew<vtkIntArray> mergedArcCountArray{};

  pointNodeIdArray->SetName("NodeId");
  pointScalarArray->SetName("Scalar");
  newArcIdArray->SetName("NewArcId");
  segmentationIdArray->SetName("SegmentationId");
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
    segmentationIdArray->InsertNextValue(-1);
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
  grid->GetCellData()->AddArray(segmentationIdArray);
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
  if(writer->Write() == 0) {
    fail("Failed to write `" + path + "`.");
  }
}

void writeStats(const std::string &path,
                const std::vector<NodeRecord> &nodes,
                const std::vector<ArcRecord> &arcs,
                const std::vector<BranchRecord> &branches,
                const SimplifiedTree &tree,
                double eps) {
  std::ofstream out(path);
  if(!out) {
    fail("Failed to open stats file `" + path + "`.");
  }

  out << "mode=txt_tree_branch_volume_iou\n";
  out << "eps=" << std::setprecision(6) << eps << '\n';
  out << "original_nodes=" << nodes.size() << '\n';
  out << "kept_nodes=" << tree.keptNodeIds.size() << '\n';
  out << "original_arcs=" << arcs.size() << '\n';
  out << "simplified_arcs=" << tree.newArcs.size() << '\n';
  out << "branches=" << branches.size() << '\n';
  out << '\n';

  for(const auto &branch : branches) {
    out << "branch=" << branch.branchId
        << ", nodes=" << branch.nodeIds.size()
        << ", kept=" << branch.keepIndices.size() << '\n';
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

    for(auto &branch : branches) {
      simplifyBranchByVolumeIoU(
        branch, nodes, nodeIdToIndex, opt.eps, opt.treeType);
    }

    const auto simplifiedTree
      = rebuildSimplifiedTree(branches, nodes, arcs, nodeIdToIndex);
    auto simplifiedNodeGrid
      = buildSimplifiedNodeGrid(simplifiedTree, nodes, nodeIdToIndex);
    auto simplifiedArcGrid
      = buildSimplifiedArcGrid(simplifiedTree, nodes, nodeIdToIndex);
    auto nodeMaskGrid = buildNodeKeepMaskGrid(nodes, simplifiedTree);

    writeGrid(simplifiedNodeGrid, joinPath(opt.outputPrefix, "_nodes.vtu"));
    writeGrid(simplifiedArcGrid, joinPath(opt.outputPrefix, "_arcs.vtu"));
    writeGrid(nodeMaskGrid, joinPath(opt.outputPrefix, "_node_mask.vtu"));
    writeStats(joinPath(opt.outputPrefix, "_stats.txt"),
               nodes,
               arcs,
               branches,
               simplifiedTree,
               opt.eps);

    std::cout << "Simplification finished. Kept "
              << simplifiedTree.keptNodeIds.size() << " / " << nodes.size()
              << " nodes.\n";
  } catch(const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
