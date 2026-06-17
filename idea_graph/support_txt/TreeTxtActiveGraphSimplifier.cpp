#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
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
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
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
  std::vector<int> incidentArcIds{};
  int criticalType{};
  double regionSize{};
  double regionSpan{};
  std::vector<int> subtreeVertices{};
  std::array<double, 3> point{0.0, 0.0, 0.0};
};

struct ArcRecord {
  int arcId{};
  int downNodeId{};
  int upNodeId{};
  double regionSize{};
  double regionSpan{};
};

struct BranchRecord {
  int branchId{};
  std::vector<int> nodeIds{};
  std::vector<int> arcIds{};
};

struct DistanceRecord {
  int branchId{};
  int refNodeId{};
  int candidateNodeId{};
  double distJ{};
  double distPos{};
  double distBbox{};
  int keep{};
};

struct ProgramOptions {
  std::string treePath{};
  std::string imagePath{};
  std::string outputPrefix{"support_txt/out/output"};
  std::string scalarArrayName{};
  double epsilonJ{0.15};
  double epsilonPos{0.10};
  double epsilonBbox{0.10};
  int treeType{0};
};

struct ActiveSubgraph {
  std::vector<std::pair<vtkIdType, vtkIdType>> activeEdges{};
  std::unordered_set<vtkIdType> activeVertices{};
  std::array<double, 3> centroid{0.0, 0.0, 0.0};
  std::array<double, 3> bboxMin{0.0, 0.0, 0.0};
  std::array<double, 3> bboxMax{0.0, 0.0, 0.0};
  double diagLength{};
};

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error(msg);
}

void printUsage() {
  std::cerr
    << "Usage: treeTxtActiveGraphSimplifier -t <tree.txt> -s <field.vti> "
    << "[-o <output_prefix>] [-j <eps_j>] [-p <eps_pos>] "
    << "[-b <eps_bbox>] [-T <0|1>] [-a <scalar_array>]\n";
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

    if(arg == "-t" || arg == "--tree") {
      opt.treePath = requireValue(arg);
    } else if(arg == "-s" || arg == "--scalar-image") {
      opt.imagePath = requireValue(arg);
    } else if(arg == "-o" || arg == "--output") {
      opt.outputPrefix = requireValue(arg);
    } else if(arg == "-j" || arg == "--epsilon-j") {
      opt.epsilonJ = std::stod(requireValue(arg));
    } else if(arg == "-p" || arg == "--epsilon-pos") {
      opt.epsilonPos = std::stod(requireValue(arg));
    } else if(arg == "-b" || arg == "--epsilon-bbox") {
      opt.epsilonBbox = std::stod(requireValue(arg));
    } else if(arg == "-T" || arg == "--tree-type") {
      opt.treeType = std::stoi(requireValue(arg));
    } else if(arg == "-a" || arg == "--scalar-array") {
      opt.scalarArrayName = requireValue(arg);
    } else if(arg == "-h" || arg == "--help") {
      printUsage();
      std::exit(0);
    } else {
      fail("Unknown argument: " + arg);
    }
  }

  if(opt.treePath.empty() || opt.imagePath.empty()) {
    printUsage();
    fail("Both tree and scalar image are required.");
  }
  if(opt.epsilonJ < 0.0 || opt.epsilonPos < 0.0 || opt.epsilonBbox < 0.0) {
    fail("Thresholds must be non-negative.");
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
    fail("Unable to read image `" + path + "`.");
  }
  auto out = vtkSmartPointer<vtkImageData>::New();
  out->DeepCopy(image);
  return out;
}

vtkDataArray *getScalarArray(vtkImageData *image, const std::string &requestedName) {
  if(image == nullptr || image->GetPointData() == nullptr) {
    fail("Image point data is missing.");
  }

  auto *pointData = image->GetPointData();
  if(!requestedName.empty()) {
    auto *array = pointData->GetArray(requestedName.c_str());
    if(array == nullptr) {
      fail("Scalar array `" + requestedName + "` not found in image.");
    }
    return array;
  }

  if(auto *scalars = pointData->GetScalars(); scalars != nullptr) {
    return scalars;
  }
  if(pointData->GetNumberOfArrays() <= 0) {
    fail("Image has no point arrays.");
  }
  auto *array = pointData->GetArray(0);
  if(array == nullptr) {
    fail("Failed to access first point array.");
  }
  return array;
}

std::vector<NodeRecord> readTreeTxt(const std::string &path) {
  std::ifstream in(path);
  if(!in) {
    fail("Unable to open tree file `" + path + "`.");
  }

  std::vector<NodeRecord> nodes{};
  std::unordered_set<int> seen{};
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
    if(seen.count(node.nodeId) != 0) {
      fail("Duplicate node id `" + std::to_string(node.nodeId) + "`.");
    }
    seen.insert(node.nodeId);
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
  const vtkIdType pointCount = image->GetNumberOfPoints();
  for(auto &node : nodes) {
    if(node.nodeId < 0 || node.nodeId >= pointCount) {
      fail("Node id " + std::to_string(node.nodeId)
           + " is outside VTI point range [0, "
           + std::to_string(pointCount - 1) + "].");
    }
    image->GetPoint(node.nodeId, node.point.data());
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
  int rootCount = 0;
  for(auto &node : nodes) {
    node.children.clear();
    node.incidentArcIds.clear();
  }

  for(const auto &node : nodes) {
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
    fail("Expected exactly one root, got " + std::to_string(rootCount) + ".");
  }

  for(auto &node : nodes) {
    node.criticalType = classifyCriticalType(node);
  }
}

void computeSubtreeData(std::vector<NodeRecord> &nodes,
                        const std::unordered_map<int, int> &nodeIdToIndex) {
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

  std::vector<int> order{};
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

        const int nextNodeId = otherNodeId(arcs[currentArcId], currentNodeId);
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

  if(std::count(arcVisited.begin(), arcVisited.end(), 0) != 0) {
    fail("Some arcs were not assigned to branches.");
  }
  return branches;
}

std::array<int, 3> idToIJK(vtkImageData *image, vtkIdType pointId) {
  int dims[3];
  image->GetDimensions(dims);
  const vtkIdType sliceSize = static_cast<vtkIdType>(dims[0]) * dims[1];
  const int z = static_cast<int>(pointId / sliceSize);
  const vtkIdType inSlice = pointId % sliceSize;
  const int y = static_cast<int>(inSlice / dims[0]);
  const int x = static_cast<int>(inSlice % dims[0]);
  return {x, y, z};
}

vtkIdType ijkToId(int i, int j, int k, int dimX, int dimY) {
  return static_cast<vtkIdType>(k) * dimY * dimX
         + static_cast<vtkIdType>(j) * dimX + i;
}

void sortGridEdge(std::pair<vtkIdType, vtkIdType> &edge) {
  if(edge.first > edge.second) {
    std::swap(edge.first, edge.second);
  }
}

ActiveSubgraph buildActiveSubgraph(const NodeRecord &node,
                                   vtkImageData *image,
                                   vtkDataArray *scalarArray) {
  if(node.subtreeVertices.empty()) {
    fail("Node " + std::to_string(node.nodeId) + " has empty subtree.");
  }

  int dims[3];
  image->GetDimensions(dims);

  std::array<int, 3> minCoord{dims[0], dims[1], dims[2]};
  std::array<int, 3> maxCoord{0, 0, 0};
  std::unordered_set<int> localRegion(node.subtreeVertices.begin(), node.subtreeVertices.end());

  for(const auto vertexId : node.subtreeVertices) {
    const auto ijk = idToIJK(image, vertexId);
    for(int d = 0; d < 3; ++d) {
      minCoord[d] = std::min(minCoord[d], ijk[d]);
      maxCoord[d] = std::max(maxCoord[d], ijk[d]);
    }
  }

  ActiveSubgraph subgraph{};
  subgraph.bboxMin = {std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max()};
  subgraph.bboxMax = {std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest()};

  const double threshold = node.scalar;
  auto processEdge = [&](vtkIdType pid1, vtkIdType pid2) {
    const bool in1 = localRegion.count(static_cast<int>(pid1)) > 0;
    const bool in2 = localRegion.count(static_cast<int>(pid2)) > 0;
    if(!in1 && !in2) {
      return;
    }

    const double val1 = scalarArray->GetComponent(pid1, 0);
    const double val2 = scalarArray->GetComponent(pid2, 0);
    if((val1 <= threshold && val2 > threshold) || (val2 <= threshold && val1 > threshold)) {
      std::pair<vtkIdType, vtkIdType> edge{pid1, pid2};
      sortGridEdge(edge);
      subgraph.activeEdges.emplace_back(edge);
      subgraph.activeVertices.insert(pid1);
      subgraph.activeVertices.insert(pid2);
    }
  };

  for(int k = minCoord[2]; k <= maxCoord[2]; ++k) {
    for(int j = minCoord[1]; j <= maxCoord[1]; ++j) {
      for(int i = minCoord[0]; i <= maxCoord[0]; ++i) {
        const vtkIdType pid = ijkToId(i, j, k, dims[0], dims[1]);
        if(i + 1 < dims[0] && i < maxCoord[0]) {
          processEdge(pid, ijkToId(i + 1, j, k, dims[0], dims[1]));
        }
        if(j + 1 < dims[1] && j < maxCoord[1]) {
          processEdge(pid, ijkToId(i, j + 1, k, dims[0], dims[1]));
        }
        if(k + 1 < dims[2] && k < maxCoord[2]) {
          processEdge(pid, ijkToId(i, j, k + 1, dims[0], dims[1]));
        }
      }
    }
  }

  std::sort(subgraph.activeEdges.begin(), subgraph.activeEdges.end());
  subgraph.activeEdges.erase(
    std::unique(subgraph.activeEdges.begin(), subgraph.activeEdges.end()),
    subgraph.activeEdges.end());

  for(const auto pid : subgraph.activeVertices) {
    double point[3];
    image->GetPoint(pid, point);
    for(int d = 0; d < 3; ++d) {
      subgraph.centroid[d] += point[d];
      subgraph.bboxMin[d] = std::min(subgraph.bboxMin[d], point[d]);
      subgraph.bboxMax[d] = std::max(subgraph.bboxMax[d], point[d]);
    }
  }

  if(!subgraph.activeVertices.empty()) {
    for(int d = 0; d < 3; ++d) {
      subgraph.centroid[d] /= static_cast<double>(subgraph.activeVertices.size());
    }
  } else {
    const auto ijk = idToIJK(image, node.nodeId);
    subgraph.centroid = {static_cast<double>(ijk[0]),
                         static_cast<double>(ijk[1]),
                         static_cast<double>(ijk[2])};
    subgraph.bboxMin = subgraph.centroid;
    subgraph.bboxMax = subgraph.centroid;
  }

  const double dx = subgraph.bboxMax[0] - subgraph.bboxMin[0];
  const double dy = subgraph.bboxMax[1] - subgraph.bboxMin[1];
  const double dz = subgraph.bboxMax[2] - subgraph.bboxMin[2];
  subgraph.diagLength = std::sqrt(dx * dx + dy * dy + dz * dz);

  return subgraph;
}

double computeJaccardDistance(const ActiveSubgraph &a, const ActiveSubgraph &b) {
  size_t i = 0;
  size_t j = 0;
  size_t interSize = 0;
  while(i < a.activeEdges.size() && j < b.activeEdges.size()) {
    if(a.activeEdges[i] == b.activeEdges[j]) {
      ++interSize;
      ++i;
      ++j;
    } else if(a.activeEdges[i] < b.activeEdges[j]) {
      ++i;
    } else {
      ++j;
    }
  }

  const size_t unionSize = a.activeEdges.size() + b.activeEdges.size() - interSize;
  if(unionSize == 0) {
    return 0.0;
  }
  return 1.0 - static_cast<double>(interSize) / static_cast<double>(unionSize);
}

double computePositionDistance(const ActiveSubgraph &a, const ActiveSubgraph &b) {
  double dx = a.centroid[0] - b.centroid[0];
  double dy = a.centroid[1] - b.centroid[1];
  double dz = a.centroid[2] - b.centroid[2];
  const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  return dist / std::max(std::max(a.diagLength, b.diagLength), 1.0);
}

double computeBboxDistance(const ActiveSubgraph &a, const ActiveSubgraph &b) {
  return std::abs(a.diagLength - b.diagLength) / std::max(std::max(a.diagLength, b.diagLength), 1.0);
}

struct SimplifyResult {
  std::set<int> keptNodes{};
  std::vector<DistanceRecord> distances{};
};

SimplifyResult simplifyBranches(const std::vector<BranchRecord> &branches,
                                const std::vector<NodeRecord> &nodes,
                                const std::unordered_map<int, int> &nodeIdToIndex,
                                vtkImageData *image,
                                vtkDataArray *scalarArray,
                                const ProgramOptions &opt) {
  std::unordered_map<int, ActiveSubgraph> cache{};
  auto getSubgraph = [&](int nodeId) -> const ActiveSubgraph & {
    const auto it = cache.find(nodeId);
    if(it != cache.end()) {
      return it->second;
    }
    const auto inserted = cache.emplace(
      nodeId, buildActiveSubgraph(nodes.at(nodeIdToIndex.at(nodeId)), image, scalarArray));
    return inserted.first->second;
  };

  SimplifyResult result{};
  for(const auto &branch : branches) {
    if(branch.nodeIds.empty()) {
      continue;
    }

    result.keptNodes.insert(branch.nodeIds.front());
    result.keptNodes.insert(branch.nodeIds.back());
    if(branch.nodeIds.size() <= 2) {
      continue;
    }

    int refIndex = 0;
    for(size_t i = 1; i + 1 < branch.nodeIds.size(); ++i) {
      const int refNodeId = branch.nodeIds[refIndex];
      const int candidateNodeId = branch.nodeIds[i];
      const auto &refSubgraph = getSubgraph(refNodeId);
      const auto &candidateSubgraph = getSubgraph(candidateNodeId);

      const double distJ = computeJaccardDistance(refSubgraph, candidateSubgraph);
      const double distPos = computePositionDistance(refSubgraph, candidateSubgraph);
      const double distBbox = computeBboxDistance(refSubgraph, candidateSubgraph);
      const bool keep = distJ >= opt.epsilonJ
                        || distPos >= opt.epsilonPos
                        || distBbox >= opt.epsilonBbox;

      result.distances.push_back(
        {branch.branchId, refNodeId, candidateNodeId, distJ, distPos, distBbox, keep ? 1 : 0});
      if(keep) {
        result.keptNodes.insert(candidateNodeId);
        refIndex = static_cast<int>(i);
      }
    }
  }

  return result;
}

vtkSmartPointer<vtkUnstructuredGrid>
buildNodeGrid(const std::vector<NodeRecord> &nodes, const std::set<int> &keptNodes) {
  vtkNew<vtkUnstructuredGrid> grid{};
  vtkNew<vtkPoints> points{};
  vtkNew<vtkCellArray> cells{};
  vtkNew<vtkIntArray> nodeIdArray{};
  vtkNew<vtkDoubleArray> scalarArray{};
  vtkNew<vtkIntArray> criticalTypeArray{};
  vtkNew<vtkIntArray> vertexIdArray{};
  vtkNew<vtkDoubleArray> regionSizeArray{};
  vtkNew<vtkDoubleArray> regionSpanArray{};
  vtkNew<vtkIntArray> keepArray{};

  nodeIdArray->SetName("NodeId");
  scalarArray->SetName("Scalar");
  criticalTypeArray->SetName("CriticalType");
  vertexIdArray->SetName("VertexId");
  regionSizeArray->SetName("RegionSize");
  regionSpanArray->SetName("RegionSpan");
  keepArray->SetName("KeepNode");

  for(const auto &node : nodes) {
    const auto pointId = points->InsertNextPoint(node.point.data());
    vtkNew<vtkVertex> vertex{};
    vertex->GetPointIds()->SetId(0, pointId);
    cells->InsertNextCell(vertex);

    nodeIdArray->InsertNextValue(node.nodeId);
    scalarArray->InsertNextValue(node.scalar);
    criticalTypeArray->InsertNextValue(node.criticalType);
    vertexIdArray->InsertNextValue(node.nodeId);
    regionSizeArray->InsertNextValue(node.regionSize);
    regionSpanArray->InsertNextValue(node.regionSpan);
    keepArray->InsertNextValue(keptNodes.count(node.nodeId) > 0 ? 1 : 0);
  }

  grid->SetPoints(points);
  grid->SetCells(VTK_VERTEX, cells);
  grid->GetPointData()->AddArray(nodeIdArray);
  grid->GetPointData()->AddArray(scalarArray);
  grid->GetPointData()->AddArray(criticalTypeArray);
  grid->GetPointData()->AddArray(vertexIdArray);
  grid->GetPointData()->AddArray(regionSizeArray);
  grid->GetPointData()->AddArray(regionSpanArray);
  grid->GetPointData()->AddArray(keepArray);
  return grid;
}

vtkSmartPointer<vtkUnstructuredGrid>
buildArcGrid(const std::vector<BranchRecord> &branches,
             const std::vector<NodeRecord> &nodes,
             const std::unordered_map<int, int> &nodeIdToIndex,
             const std::set<int> &keptNodes) {
  vtkNew<vtkUnstructuredGrid> grid{};
  vtkNew<vtkPoints> points{};
  vtkNew<vtkCellArray> cells{};
  vtkNew<vtkIntArray> chainIdArray{};
  vtkNew<vtkIntArray> collapsedArray{};
  vtkNew<vtkIntArray> downNodeIdArray{};
  vtkNew<vtkIntArray> upNodeIdArray{};

  chainIdArray->SetName("ChainId");
  collapsedArray->SetName("CollapsedArcCount");
  downNodeIdArray->SetName("downNodeId");
  upNodeIdArray->SetName("upNodeId");

  vtkIdType pointOffset = 0;
  for(const auto &branch : branches) {
    std::vector<int> keptInBranch{};
    for(const auto nodeId : branch.nodeIds) {
      if(keptNodes.count(nodeId) > 0) {
        keptInBranch.emplace_back(nodeId);
      }
    }

    for(size_t i = 0; i + 1 < keptInBranch.size(); ++i) {
      const int downId = keptInBranch[i];
      const int upId = keptInBranch[i + 1];
      const auto &downNode = nodes.at(nodeIdToIndex.at(downId));
      const auto &upNode = nodes.at(nodeIdToIndex.at(upId));

      points->InsertNextPoint(downNode.point.data());
      points->InsertNextPoint(upNode.point.data());
      vtkIdType ids[2]{pointOffset, pointOffset + 1};
      cells->InsertNextCell(2, ids);

      chainIdArray->InsertNextValue(branch.branchId);
      collapsedArray->InsertNextValue(static_cast<int>(keptInBranch.size()) - 2);
      downNodeIdArray->InsertNextValue(downId);
      upNodeIdArray->InsertNextValue(upId);

      pointOffset += 2;
    }
  }

  grid->SetPoints(points);
  grid->SetCells(VTK_LINE, cells);
  grid->GetCellData()->AddArray(chainIdArray);
  grid->GetCellData()->AddArray(collapsedArray);
  grid->GetCellData()->AddArray(downNodeIdArray);
  grid->GetCellData()->AddArray(upNodeIdArray);
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
                const ProgramOptions &opt,
                const std::string &scalarArrayName,
                const std::vector<NodeRecord> &nodes,
                const std::vector<ArcRecord> &arcs,
                const std::vector<BranchRecord> &branches,
                const SimplifyResult &result) {
  std::ofstream out(path);
  if(!out) {
    fail("Failed to open stats file `" + path + "`.");
  }

  out << "mode=txt_tree_active_graph\n";
  out << "tree=" << opt.treePath << '\n';
  out << "image=" << opt.imagePath << '\n';
  out << "scalar_array=" << scalarArrayName << '\n';
  out << "epsilon_j=" << opt.epsilonJ << '\n';
  out << "epsilon_pos=" << opt.epsilonPos << '\n';
  out << "epsilon_bbox=" << opt.epsilonBbox << '\n';
  out << "original_nodes=" << nodes.size() << '\n';
  out << "kept_nodes=" << result.keptNodes.size() << '\n';
  out << "original_arcs=" << arcs.size() << '\n';
  out << "branches=" << branches.size() << '\n';
  out << '\n';
  out << "branchId,refNodeId,candidateNodeId,distJ,distPos,distBbox,keep\n";
  for(const auto &record : result.distances) {
    out << record.branchId << ','
        << record.refNodeId << ','
        << record.candidateNodeId << ','
        << record.distJ << ','
        << record.distPos << ','
        << record.distBbox << ','
        << record.keep << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opt = parseArgs(argc, argv);
    const auto outputDir = std::filesystem::path(opt.outputPrefix).parent_path();
    if(!outputDir.empty()) {
      std::filesystem::create_directories(outputDir);
    }

    auto image = readImage(opt.imagePath);
    auto *scalarArray = getScalarArray(image, opt.scalarArrayName);
    const std::string scalarArrayName = scalarArray->GetName() == nullptr
                                          ? std::string("<unnamed>")
                                          : std::string(scalarArray->GetName());

    auto nodes = readTreeTxt(opt.treePath);
    const auto nodeIdToIndex = buildNodeIndex(nodes);
    attachGeometryFromImage(nodes, image);
    buildTreeRelations(nodes, nodeIdToIndex);
    computeSubtreeData(nodes, nodeIdToIndex);
    auto arcs = buildArcs(nodes, nodeIdToIndex);
    auto branches = extractBranches(nodes, arcs, nodeIdToIndex, opt.treeType);
    const auto result = simplifyBranches(branches, nodes, nodeIdToIndex, image, scalarArray, opt);

    auto nodeGrid = buildNodeGrid(nodes, result.keptNodes);
    auto arcGrid = buildArcGrid(branches, nodes, nodeIdToIndex, result.keptNodes);

    writeGrid(nodeGrid, opt.outputPrefix + "_simplified_nodes.vtu");
    writeGrid(arcGrid, opt.outputPrefix + "_simplified_arcs.vtu");
    writeStats(opt.outputPrefix + "_stats.csv",
               opt,
               scalarArrayName,
               nodes,
               arcs,
               branches,
               result);

    std::cout << "Loaded nodes: " << nodes.size() << '\n';
    std::cout << "Loaded arcs: " << arcs.size() << '\n';
    std::cout << "Extracted branches: " << branches.size() << '\n';
    std::cout << "Kept nodes: " << result.keptNodes.size() << '\n';
    std::cout << "Wrote outputs with prefix: " << opt.outputPrefix << '\n';
    return 0;
  } catch(const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
