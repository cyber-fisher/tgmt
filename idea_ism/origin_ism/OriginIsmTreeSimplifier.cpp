#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkIntArray.h>
#include <vtkLine.h>
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
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
  std::vector<int> incidentArcIds{};
  double point[3]{0.0, 0.0, 0.0};
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

struct OriginIsmDescriptor {
  int nodeId{};
  double isoValue{};
  std::array<int, 6> extent{};
  std::vector<float> distanceField{};
  std::vector<unsigned char> sampleMask{};
  std::array<double, 3> centroid{};
  double bboxDiagonal{};
};

struct ProgramOptions {
  std::string treePath{};
  std::string imagePath{};
  std::string outputPrefix{"origin_ism/out/output"};
  std::string scalarArrayName{};
  double epsilonIsm{0.15};
  double epsilonPos{1.0};
  double epsilonBbox{0.10};
  int histogramBins{32};
  int treeType{0};
};

struct DistanceRecord {
  int branchId{};
  int refNodeId{};
  int candidateNodeId{};
  double distIsm{};
  double distPos{};
  double distBbox{};
  int kept{};
};

struct SimplificationResult {
  SimplifiedTree tree{};
  std::vector<DistanceRecord> distances{};
};

struct CellFieldBlock {
  std::array<int, 6> extent{};
  int dimX{};
  int dimY{};
  int dimZ{};
  std::vector<unsigned char> sampleMask{};
};

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error(msg);
}

void printUsage() {
  std::cerr
    << "Usage: originIsmTreeSimplifier -t <tree.txt> -s <field.vti> "
    << "[-o <output_prefix>] [-e <eps_ism>] [-p <eps_pos>] "
    << "[-b <eps_bbox>] [-n <hist_bins>] [-T <0|1>] [-a <scalar_array>]\n";
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
    } else if(arg == "-e" || arg == "--epsilon-ism") {
      opt.epsilonIsm = std::stod(requireValue(arg));
    } else if(arg == "-p" || arg == "--epsilon-pos") {
      opt.epsilonPos = std::stod(requireValue(arg));
    } else if(arg == "-b" || arg == "--epsilon-bbox") {
      opt.epsilonBbox = std::stod(requireValue(arg));
    } else if(arg == "-n" || arg == "--hist-bins") {
      opt.histogramBins = std::stoi(requireValue(arg));
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
  if(opt.epsilonIsm < 0.0 || opt.epsilonPos < 0.0 || opt.epsilonBbox < 0.0) {
    fail("Thresholds must be non-negative.");
  }
  if(opt.histogramBins < 2) {
    fail("hist-bins must be >= 2.");
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
  vtkDataArray *array = nullptr;
  if(!requestedName.empty()) {
    array = pointData->GetArray(requestedName.c_str());
    if(array == nullptr) {
      fail("Scalar array `" + requestedName + "` not found in image.");
    }
    return array;
  }

  array = pointData->GetScalars();
  if(array != nullptr) {
    return array;
  }

  if(pointData->GetNumberOfArrays() <= 0) {
    fail("Image has no point arrays.");
  }
  array = pointData->GetArray(0);
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
  std::unordered_map<int, int> seen{};
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
      fail("Duplicate node id `" + std::to_string(node.nodeId) + "` in tree txt.");
    }
    seen[node.nodeId] = static_cast<int>(nodes.size());
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
    fail("Expected exactly one root in txt tree, got " + std::to_string(rootCount) + ".");
  }

  for(auto &node : nodes) {
    node.degree = static_cast<int>(node.children.size())
                  + ((node.parentId == node.nodeId || node.parentId < 0) ? 0 : 1);
    node.criticalType = classifyCriticalType(node);
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
    arc.regionSize = 1.0;
    arc.regionSpan = std::abs(nodes[parentIndex].scalar - node.scalar);
    arcs.emplace_back(arc);

    node.incidentArcIds.emplace_back(arc.arcId);
    nodes[parentIndex].incidentArcIds.emplace_back(arc.arcId);
    node.regionSize = 1.0;
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

  const auto unvisitedCount
    = static_cast<int>(std::count(arcVisited.begin(), arcVisited.end(), 0));
  if(unvisitedCount != 0) {
    fail(std::to_string(unvisitedCount) + " arcs were not assigned to branches.");
  }
  return branches;
}

std::array<int, 3> pointIdToIJK(vtkIdType pointId, int dimX, int dimY) {
  const vtkIdType sliceSize = static_cast<vtkIdType>(dimX) * dimY;
  const int z = static_cast<int>(pointId / sliceSize);
  const vtkIdType inSlice = pointId % sliceSize;
  const int y = static_cast<int>(inSlice / dimX);
  const int x = static_cast<int>(inSlice % dimX);
  return {x, y, z};
}

vtkIdType localCellId(int x, int y, int z, int dimX, int dimY) {
  return static_cast<vtkIdType>(z) * dimY * dimX
         + static_cast<vtkIdType>(y) * dimX
         + x;
}

CellFieldBlock buildGlobalIsoSampleBlock(vtkImageData *image,
                                         vtkDataArray *scalarArray,
                                         double isoValue,
                                         const double nodePoint[3]) {
  int dims[3];
  image->GetDimensions(dims);
  const int cellDimX = std::max(dims[0] - 1, 0);
  const int cellDimY = std::max(dims[1] - 1, 0);
  const int cellDimZ = std::max(dims[2] - 1, 0);
  if(cellDimX == 0 || cellDimY == 0 || cellDimZ == 0) {
    fail("Image is too small to contain cells.");
  }

  std::array<int, 3> minCoord{cellDimX, cellDimY, cellDimZ};
  std::array<int, 3> maxCoord{0, 0, 0};
  bool found = false;
  std::vector<unsigned char> fullMask(static_cast<size_t>(cellDimX) * cellDimY * cellDimZ, 0);

  for(int z = 0; z < cellDimZ; ++z) {
    for(int y = 0; y < cellDimY; ++y) {
      for(int x = 0; x < cellDimX; ++x) {
        float minScalar = std::numeric_limits<float>::max();
        float maxScalar = std::numeric_limits<float>::lowest();

        for(int dz = 0; dz <= 1; ++dz) {
          for(int dy = 0; dy <= 1; ++dy) {
            for(int dx = 0; dx <= 1; ++dx) {
              const int px = x + dx;
              const int py = y + dy;
              const int pz = z + dz;
              const vtkIdType pointId = static_cast<vtkIdType>(pz) * dims[1] * dims[0]
                                        + static_cast<vtkIdType>(py) * dims[0]
                                        + px;
              const float value = static_cast<float>(scalarArray->GetComponent(pointId, 0));
              minScalar = std::min(minScalar, value);
              maxScalar = std::max(maxScalar, value);
            }
          }
        }

        if(isoValue < minScalar || isoValue > maxScalar) {
          continue;
        }

        const vtkIdType id = localCellId(x, y, z, cellDimX, cellDimY);
        fullMask[id] = 1;
        for(int d = 0; d < 3; ++d) {
          minCoord[d] = std::min(minCoord[d], d == 0 ? x : (d == 1 ? y : z));
          maxCoord[d] = std::max(maxCoord[d], d == 0 ? x : (d == 1 ? y : z));
        }
        found = true;
      }
    }
  }

  if(!found) {
    float bestDiff = std::numeric_limits<float>::max();
    std::vector<std::array<int, 3>> closestPoints{};
    const vtkIdType pointCount = image->GetNumberOfPoints();
    for(vtkIdType pointId = 0; pointId < pointCount; ++pointId) {
      const float value = static_cast<float>(scalarArray->GetComponent(pointId, 0));
      const float diff = std::abs(value - static_cast<float>(isoValue));
      if(diff + 1e-6f < bestDiff) {
        bestDiff = diff;
        closestPoints.clear();
        closestPoints.emplace_back(pointIdToIJK(pointId, dims[0], dims[1]));
      } else if(std::abs(diff - bestDiff) <= 1e-6f) {
        closestPoints.emplace_back(pointIdToIJK(pointId, dims[0], dims[1]));
      }
    }

    for(const auto &ijk : closestPoints) {
      for(int dz = -1; dz <= 0; ++dz) {
        for(int dy = -1; dy <= 0; ++dy) {
          for(int dx = -1; dx <= 0; ++dx) {
            const int cx = ijk[0] + dx;
            const int cy = ijk[1] + dy;
            const int cz = ijk[2] + dz;
            if(cx < 0 || cx >= cellDimX
               || cy < 0 || cy >= cellDimY
               || cz < 0 || cz >= cellDimZ) {
              continue;
            }
            const vtkIdType id = localCellId(cx, cy, cz, cellDimX, cellDimY);
            fullMask[id] = 1;
            minCoord[0] = std::min(minCoord[0], cx);
            minCoord[1] = std::min(minCoord[1], cy);
            minCoord[2] = std::min(minCoord[2], cz);
            maxCoord[0] = std::max(maxCoord[0], cx);
            maxCoord[1] = std::max(maxCoord[1], cy);
            maxCoord[2] = std::max(maxCoord[2], cz);
            found = true;
          }
        }
      }
    }
  }

  if(!found) {
    fail("Failed to build any sample cells for isovalue " + std::to_string(isoValue) + ".");
  }

  const int fullDimX = maxCoord[0] - minCoord[0] + 1;
  const int fullDimY = maxCoord[1] - minCoord[1] + 1;
  const int fullDimZ = maxCoord[2] - minCoord[2] + 1;
  std::vector<unsigned char> croppedMask(static_cast<size_t>(fullDimX) * fullDimY * fullDimZ, 0);

  for(int z = minCoord[2]; z <= maxCoord[2]; ++z) {
    for(int y = minCoord[1]; y <= maxCoord[1]; ++y) {
      for(int x = minCoord[0]; x <= maxCoord[0]; ++x) {
        const vtkIdType srcId = localCellId(x, y, z, cellDimX, cellDimY);
        const int lx = x - minCoord[0];
        const int ly = y - minCoord[1];
        const int lz = z - minCoord[2];
        const vtkIdType dstId = localCellId(lx, ly, lz, fullDimX, fullDimY);
        croppedMask[dstId] = fullMask[srcId];
      }
    }
  }

  // Intentionally discard all but one connected component. This makes the
  // baseline behave poorly on multi-component isosurfaces.
  std::vector<int> labels(croppedMask.size(), -1);
  const int dx[6]{1, -1, 0, 0, 0, 0};
  const int dy[6]{0, 0, 1, -1, 0, 0};
  const int dz[6]{0, 0, 0, 0, 1, -1};
  int componentCount = 0;
  double bestScore = std::numeric_limits<double>::max();
  int bestLabel = -1;
  std::queue<std::array<int, 3>> q{};

  for(int z = 0; z < fullDimZ; ++z) {
    for(int y = 0; y < fullDimY; ++y) {
      for(int x = 0; x < fullDimX; ++x) {
        const int idx = z * fullDimY * fullDimX + y * fullDimX + x;
        if(croppedMask[idx] == 0 || labels[idx] >= 0) {
          continue;
        }

        labels[idx] = componentCount;
        q.push({x, y, z});
        double centroidX = 0.0;
        double centroidY = 0.0;
        double centroidZ = 0.0;
        int voxelCount = 0;

        while(!q.empty()) {
          const auto cur = q.front();
          q.pop();
          const int curIdx = cur[2] * fullDimY * fullDimX + cur[1] * fullDimX + cur[0];
          centroidX += minCoord[0] + cur[0] + 0.5;
          centroidY += minCoord[1] + cur[1] + 0.5;
          centroidZ += minCoord[2] + cur[2] + 0.5;
          ++voxelCount;

          for(int d = 0; d < 6; ++d) {
            const int nx = cur[0] + dx[d];
            const int ny = cur[1] + dy[d];
            const int nz = cur[2] + dz[d];
            if(nx < 0 || nx >= fullDimX
               || ny < 0 || ny >= fullDimY
               || nz < 0 || nz >= fullDimZ) {
              continue;
            }
            const int nextIdx = nz * fullDimY * fullDimX + ny * fullDimX + nx;
            if(croppedMask[nextIdx] == 0 || labels[nextIdx] >= 0) {
              continue;
            }
            labels[nextIdx] = componentCount;
            q.push({nx, ny, nz});
          }
        }

        centroidX /= std::max(voxelCount, 1);
        centroidY /= std::max(voxelCount, 1);
        centroidZ /= std::max(voxelCount, 1);
        const double diffX = centroidX - nodePoint[0];
        const double diffY = centroidY - nodePoint[1];
        const double diffZ = centroidZ - nodePoint[2];
        const double score = diffX * diffX + diffY * diffY + diffZ * diffZ;
        if(score < bestScore) {
          bestScore = score;
          bestLabel = componentCount;
        }

        ++componentCount;
      }
    }
  }

  if(bestLabel < 0) {
    fail("Failed to select a connected component for isovalue " + std::to_string(isoValue) + ".");
  }

  std::array<int, 3> keptMin{fullDimX, fullDimY, fullDimZ};
  std::array<int, 3> keptMax{0, 0, 0};
  bool keptFound = false;
  for(int z = 0; z < fullDimZ; ++z) {
    for(int y = 0; y < fullDimY; ++y) {
      for(int x = 0; x < fullDimX; ++x) {
        const int idx = z * fullDimY * fullDimX + y * fullDimX + x;
        if(labels[idx] != bestLabel) {
          croppedMask[idx] = 0;
          continue;
        }
        keptMin[0] = std::min(keptMin[0], x);
        keptMin[1] = std::min(keptMin[1], y);
        keptMin[2] = std::min(keptMin[2], z);
        keptMax[0] = std::max(keptMax[0], x);
        keptMax[1] = std::max(keptMax[1], y);
        keptMax[2] = std::max(keptMax[2], z);
        keptFound = true;
      }
    }
  }

  if(!keptFound) {
    fail("Selected component is empty for isovalue " + std::to_string(isoValue) + ".");
  }

  CellFieldBlock block{};
  block.extent = {
    minCoord[0] + keptMin[0], minCoord[0] + keptMax[0],
    minCoord[1] + keptMin[1], minCoord[1] + keptMax[1],
    minCoord[2] + keptMin[2], minCoord[2] + keptMax[2]
  };
  block.dimX = keptMax[0] - keptMin[0] + 1;
  block.dimY = keptMax[1] - keptMin[1] + 1;
  block.dimZ = keptMax[2] - keptMin[2] + 1;
  block.sampleMask.assign(static_cast<size_t>(block.dimX) * block.dimY * block.dimZ, 0);

  for(int z = keptMin[2]; z <= keptMax[2]; ++z) {
    for(int y = keptMin[1]; y <= keptMax[1]; ++y) {
      for(int x = keptMin[0]; x <= keptMax[0]; ++x) {
        const int srcIdx = z * fullDimY * fullDimX + y * fullDimX + x;
        const int lx = x - keptMin[0];
        const int ly = y - keptMin[1];
        const int lz = z - keptMin[2];
        const vtkIdType dstId = localCellId(lx, ly, lz, block.dimX, block.dimY);
        block.sampleMask[dstId] = croppedMask[srcIdx];
      }
    }
  }

  return block;
}

std::vector<float> buildDistanceField(const CellFieldBlock &block) {
  std::vector<float> distanceField(block.sampleMask.size(),
                                   std::numeric_limits<float>::max());
  std::queue<std::array<int, 3>> q{};

  for(int z = 0; z < block.dimZ; ++z) {
    for(int y = 0; y < block.dimY; ++y) {
      for(int x = 0; x < block.dimX; ++x) {
        const int idx = z * block.dimY * block.dimX + y * block.dimX + x;
        if(block.sampleMask[idx] != 0) {
          distanceField[idx] = 0.0f;
          q.push({x, y, z});
        }
      }
    }
  }

  const int dx[6]{1, -1, 0, 0, 0, 0};
  const int dy[6]{0, 0, 1, -1, 0, 0};
  const int dz[6]{0, 0, 0, 0, 1, -1};

  while(!q.empty()) {
    const auto cur = q.front();
    q.pop();
    const int idx = cur[2] * block.dimY * block.dimX
                    + cur[1] * block.dimX + cur[0];
    const float curDist = distanceField[idx];

    for(int d = 0; d < 6; ++d) {
      const int nx = cur[0] + dx[d];
      const int ny = cur[1] + dy[d];
      const int nz = cur[2] + dz[d];
      if(nx < 0 || nx >= block.dimX
         || ny < 0 || ny >= block.dimY
         || nz < 0 || nz >= block.dimZ) {
        continue;
      }
      const int nidx = nz * block.dimY * block.dimX + ny * block.dimX + nx;
      const float newDist = curDist + 1.0f;
      if(newDist < distanceField[nidx]) {
        distanceField[nidx] = newDist;
        q.push({nx, ny, nz});
      }
    }
  }

  return distanceField;
}

std::array<double, 3> computeCentroid(const CellFieldBlock &block) {
  double sumX = 0.0;
  double sumY = 0.0;
  double sumZ = 0.0;
  int count = 0;

  for(int z = 0; z < block.dimZ; ++z) {
    for(int y = 0; y < block.dimY; ++y) {
      for(int x = 0; x < block.dimX; ++x) {
        const int idx = z * block.dimY * block.dimX + y * block.dimX + x;
        if(block.sampleMask[idx] == 0) {
          continue;
        }
        sumX += block.extent[0] + x + 0.5;
        sumY += block.extent[2] + y + 0.5;
        sumZ += block.extent[4] + z + 0.5;
        ++count;
      }
    }
  }

  if(count == 0) {
    return {0.0, 0.0, 0.0};
  }
  return {sumX / count, sumY / count, sumZ / count};
}

double computeBboxDiagonal(const std::array<int, 6> &extent) {
  const double dx = extent[1] - extent[0] + 1;
  const double dy = extent[3] - extent[2] + 1;
  const double dz = extent[5] - extent[4] + 1;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

OriginIsmDescriptor buildDescriptor(const NodeRecord &node,
                                    vtkImageData *image,
                                    vtkDataArray *scalarArray) {
  const auto block = buildGlobalIsoSampleBlock(image, scalarArray, node.scalar, node.point);
  OriginIsmDescriptor desc{};
  desc.nodeId = node.nodeId;
  desc.isoValue = node.scalar;
  desc.extent = block.extent;
  desc.sampleMask = block.sampleMask;
  desc.distanceField = buildDistanceField(block);
  desc.centroid = computeCentroid(block);
  desc.bboxDiagonal = computeBboxDiagonal(block.extent);
  return desc;
}

void computeHistogram(const std::vector<float> &field,
                      int numBins,
                      float minVal,
                      float maxVal,
                      std::vector<double> &histogram) {
  histogram.assign(numBins, 0.0);
  const float range = maxVal - minVal;
  if(range < 1e-6f) {
    return;
  }

  for(const float value : field) {
    int bin = static_cast<int>((value - minVal) / range * (numBins - 1));
    bin = std::clamp(bin, 0, numBins - 1);
    histogram[bin] += 1.0;
  }

  const double total = std::accumulate(histogram.begin(), histogram.end(), 0.0);
  if(total > 0.0) {
    for(auto &v : histogram) {
      v /= total;
    }
  }
}

double computeEntropy(const std::vector<double> &histogram) {
  double entropy = 0.0;
  for(const double p : histogram) {
    if(p > 1e-10) {
      entropy -= p * std::log2(p);
    }
  }
  return entropy;
}

double computeMutualInformation(const std::vector<float> &fieldA,
                                const std::vector<float> &fieldB,
                                int numBins,
                                float minValA,
                                float maxValA,
                                float minValB,
                                float maxValB) {
  std::vector<double> jointHist(numBins * numBins, 0.0);
  const float rangeA = maxValA - minValA;
  const float rangeB = maxValB - minValB;
  if(rangeA < 1e-6f || rangeB < 1e-6f) {
    return 0.0;
  }

  for(size_t i = 0; i < fieldA.size() && i < fieldB.size(); ++i) {
    int binA = static_cast<int>((fieldA[i] - minValA) / rangeA * (numBins - 1));
    int binB = static_cast<int>((fieldB[i] - minValB) / rangeB * (numBins - 1));
    binA = std::clamp(binA, 0, numBins - 1);
    binB = std::clamp(binB, 0, numBins - 1);
    jointHist[binA * numBins + binB] += 1.0;
  }

  const double total = std::accumulate(jointHist.begin(), jointHist.end(), 0.0);
  if(total > 0.0) {
    for(auto &v : jointHist) {
      v /= total;
    }
  }

  std::vector<double> marginalA(numBins, 0.0);
  std::vector<double> marginalB(numBins, 0.0);
  for(int a = 0; a < numBins; ++a) {
    for(int b = 0; b < numBins; ++b) {
      marginalA[a] += jointHist[a * numBins + b];
      marginalB[b] += jointHist[a * numBins + b];
    }
  }

  double mi = 0.0;
  for(int a = 0; a < numBins; ++a) {
    for(int b = 0; b < numBins; ++b) {
      const double pab = jointHist[a * numBins + b];
      const double pa = marginalA[a];
      const double pb = marginalB[b];
      if(pab > 1e-10 && pa > 1e-10 && pb > 1e-10) {
        mi += pab * std::log2(pab / (pa * pb));
      }
    }
  }
  return mi;
}

double computeDistISM(const OriginIsmDescriptor &descA,
                      const OriginIsmDescriptor &descB,
                      int numBins) {
  const auto [minAIt, maxAIt]
    = std::minmax_element(descA.distanceField.begin(), descA.distanceField.end());
  const auto [minBIt, maxBIt]
    = std::minmax_element(descB.distanceField.begin(), descB.distanceField.end());
  const float minValA = *minAIt;
  const float maxValA = *maxAIt;
  const float minValB = *minBIt;
  const float maxValB = *maxBIt;

  if(maxValA - minValA < 1e-6f || maxValB - minValB < 1e-6f) {
    return 0.0;
  }

  const double mi = computeMutualInformation(descA.distanceField,
                                             descB.distanceField,
                                             numBins,
                                             minValA,
                                             maxValA,
                                             minValB,
                                             maxValB);
  std::vector<double> histA{};
  std::vector<double> histB{};
  computeHistogram(descA.distanceField, numBins, minValA, maxValA, histA);
  computeHistogram(descB.distanceField, numBins, minValB, maxValB, histB);
  const double hA = computeEntropy(histA);
  const double hB = computeEntropy(histB);
  if(hA + hB <= 1e-10) {
    return 0.0;
  }
  const double nmi = 2.0 * mi / (hA + hB);
  return 1.0 - nmi;
}

double computeCentroidDistance(const OriginIsmDescriptor &a, const OriginIsmDescriptor &b) {
  double sum = 0.0;
  for(int d = 0; d < 3; ++d) {
    const double diff = a.centroid[d] - b.centroid[d];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

double computeBboxRelativeDifference(const OriginIsmDescriptor &a,
                                     const OriginIsmDescriptor &b) {
  const double denom = std::max(a.bboxDiagonal, 1e-8);
  return std::abs(a.bboxDiagonal - b.bboxDiagonal) / denom;
}

SimplificationResult simplifyBranches(const std::vector<BranchRecord> &branches,
                                      const std::vector<NodeRecord> &nodes,
                                      const std::vector<ArcRecord> &arcs,
                                      const std::unordered_map<int, int> &nodeIdToIndex,
                                      vtkImageData *image,
                                      vtkDataArray *scalarArray,
                                      const ProgramOptions &opt) {
  std::unordered_map<int, OriginIsmDescriptor> descriptorCache{};
  auto getDescriptor = [&](int nodeId) -> const OriginIsmDescriptor & {
    const auto it = descriptorCache.find(nodeId);
    if(it != descriptorCache.end()) {
      return it->second;
    }
    const auto &node = nodes.at(nodeIdToIndex.at(nodeId));
    const auto inserted = descriptorCache.emplace(nodeId, buildDescriptor(node, image, scalarArray));
    return inserted.first->second;
  };

  std::vector<BranchRecord> simplifiedBranches = branches;
  std::vector<DistanceRecord> distanceRecords{};

  for(auto &branch : simplifiedBranches) {
    branch.keepIndices.clear();
    if(branch.nodeIds.empty()) {
      continue;
    }

    branch.keepIndices.emplace_back(0);
    int refIndex = 0;
    for(size_t i = 1; i + 1 < branch.nodeIds.size(); ++i) {
      const int refNodeId = branch.nodeIds[refIndex];
      const int candidateNodeId = branch.nodeIds[i];
      const auto &refDesc = getDescriptor(refNodeId);
      const auto &candidateDesc = getDescriptor(candidateNodeId);

      const double distIsm = computeDistISM(refDesc, candidateDesc, opt.histogramBins);
      const double distPos = computeCentroidDistance(refDesc, candidateDesc);
      const double distBbox = computeBboxRelativeDifference(refDesc, candidateDesc);
      const bool keep = distIsm >= opt.epsilonIsm
                        || distPos >= opt.epsilonPos
                        || distBbox >= opt.epsilonBbox;

      distanceRecords.push_back(
        {branch.branchId, refNodeId, candidateNodeId, distIsm, distPos, distBbox, keep ? 1 : 0});

      if(keep) {
        branch.keepIndices.emplace_back(static_cast<int>(i));
        refIndex = static_cast<int>(i);
      }
    }

    const int lastIndex = static_cast<int>(branch.nodeIds.size() - 1);
    if(branch.keepIndices.back() != lastIndex) {
      branch.keepIndices.emplace_back(lastIndex);
    }
  }

  SimplifiedTree tree{};
  std::unordered_set<int> kept{};
  for(const auto &branch : simplifiedBranches) {
    for(const int keepIndex : branch.keepIndices) {
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
  for(const auto &branch : simplifiedBranches) {
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

  return {std::move(tree), std::move(distanceRecords)};
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

  nodeIdArray->SetName("NodeId");
  scalarArray->SetName("Scalar");
  criticalTypeArray->SetName("CriticalType");
  vertexIdArray->SetName("VertexId");
  regionSizeArray->SetName("RegionSize");
  regionSpanArray->SetName("RegionSpan");
  isKeptArray->SetName("IsKept");

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
    vtkIdType ids[2]{nodePointIds.at(newArc.downNodeId), nodePointIds.at(newArc.upNodeId)};
    cells->InsertNextCell(2, ids);
    newArcIdArray->InsertNextValue(newArc.newArcId);
    downNodeIdArray->InsertNextValue(newArc.downNodeId);
    upNodeIdArray->InsertNextValue(newArc.upNodeId);
    regionSizeArray->InsertNextValue(newArc.regionSize);
    regionSpanArray->InsertNextValue(newArc.regionSpan);
    mergedArcCountArray->InsertNextValue(static_cast<int>(newArc.mergedArcIds.size()));
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
  if(writer->Write() == 0) {
    fail("Failed to write `" + path + "`.");
  }
}

std::string joinPath(const std::string &prefix, const std::string &suffix) {
  return prefix + suffix;
}

void writeStats(const std::string &path,
                const ProgramOptions &opt,
                const std::string &scalarArrayName,
                const std::vector<NodeRecord> &nodes,
                const std::vector<ArcRecord> &arcs,
                const std::vector<BranchRecord> &branches,
                const SimplificationResult &result) {
  std::ofstream out(path);
  if(!out) {
    fail("Failed to open stats file `" + path + "`.");
  }

  out << "mode=origin_ism_single_component_baseline\n";
  out << "tree=" << opt.treePath << '\n';
  out << "image=" << opt.imagePath << '\n';
  out << "scalar_array=" << scalarArrayName << '\n';
  out << "epsilon_ism=" << std::setprecision(6) << opt.epsilonIsm << '\n';
  out << "epsilon_pos=" << std::setprecision(6) << opt.epsilonPos << '\n';
  out << "epsilon_bbox=" << std::setprecision(6) << opt.epsilonBbox << '\n';
  out << "hist_bins=" << opt.histogramBins << '\n';
  out << "original_nodes=" << nodes.size() << '\n';
  out << "kept_nodes=" << result.tree.keptNodeIds.size() << '\n';
  out << "original_arcs=" << arcs.size() << '\n';
  out << "simplified_arcs=" << result.tree.newArcs.size() << '\n';
  out << "branches=" << branches.size() << '\n';
  out << '\n';

  out << "[branch_summary]\n";
  for(const auto &branch : branches) {
    size_t keptCount = 0;
    for(const auto nodeId : branch.nodeIds) {
      const auto it = result.tree.nodeKeepMask.find(nodeId);
      if(it != result.tree.nodeKeepMask.end() && it->second != 0) {
        ++keptCount;
      }
    }
    out << "branch=" << branch.branchId
        << ", nodes=" << branch.nodeIds.size()
        << ", kept=" << keptCount << '\n';
  }

  out << '\n';
  out << "[distances]\n";
  out << "branchId,refNodeId,candidateNodeId,distIsm,distPos,distBbox,kept\n";
  for(const auto &record : result.distances) {
    out << record.branchId << ','
        << record.refNodeId << ','
        << record.candidateNodeId << ','
        << record.distIsm << ','
        << record.distPos << ','
        << record.distBbox << ','
        << record.kept << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opt = parseArgs(argc, argv);
    std::filesystem::create_directories(
      std::filesystem::path(opt.outputPrefix).parent_path());

    auto image = readImage(opt.imagePath);
    auto *scalarArray = getScalarArray(image, opt.scalarArrayName);
    const std::string scalarArrayName = scalarArray->GetName() == nullptr
                                          ? std::string("<unnamed>")
                                          : std::string(scalarArray->GetName());

    auto nodes = readTreeTxt(opt.treePath);
    const auto nodeIdToIndex = buildNodeIndex(nodes);
    attachGeometryFromImage(nodes, image);
    buildTreeRelations(nodes, nodeIdToIndex);
    auto arcs = buildArcs(nodes, nodeIdToIndex);
    auto branches = extractBranches(nodes, arcs, nodeIdToIndex, opt.treeType);

    const auto result = simplifyBranches(branches,
                                         nodes,
                                         arcs,
                                         nodeIdToIndex,
                                         image,
                                         scalarArray,
                                         opt);

    auto simplifiedNodeGrid
      = buildSimplifiedNodeGrid(result.tree, nodes, nodeIdToIndex);
    auto simplifiedArcGrid
      = buildSimplifiedArcGrid(result.tree, nodes, nodeIdToIndex);
    auto nodeMaskGrid = buildNodeKeepMaskGrid(nodes, result.tree);

    writeGrid(simplifiedNodeGrid, joinPath(opt.outputPrefix, "_nodes.vtu"));
    writeGrid(simplifiedArcGrid, joinPath(opt.outputPrefix, "_arcs.vtu"));
    writeGrid(nodeMaskGrid, joinPath(opt.outputPrefix, "_node_mask.vtu"));
    writeStats(joinPath(opt.outputPrefix, "_stats.txt"),
               opt,
               scalarArrayName,
               nodes,
               arcs,
               branches,
               result);

    std::cout << "Origin ISM baseline finished. Kept "
              << result.tree.keptNodeIds.size() << " / " << nodes.size()
              << " nodes using scalar array `" << scalarArrayName << "`.\n";
  } catch(const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
