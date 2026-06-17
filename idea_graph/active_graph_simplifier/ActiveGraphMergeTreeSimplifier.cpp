#include "ActiveGraphMergeTreeSimplifier.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <fstream>
#include <iostream>

namespace active_graph {

int ActiveGraphMergeTreeSimplifier::run(const ActiveGraphSimplificationParams &params) {
  std::cout << "=== ActiveGraphMergeTreeSimplifier ===" << std::endl;
  params_ = params;

  if (loadData(params) != 0) {
    std::cerr << "ERROR: Failed to load data" << std::endl;
    return -1;
  }

  if (buildTopologicalChains() != 0) {
    std::cerr << "ERROR: Failed to build topological chains" << std::endl;
    return -1;
  }

  if (simplify() != 0) {
    std::cerr << "ERROR: Failed to simplify" << std::endl;
    return -1;
  }

  if (writeOutputs(params) != 0) {
    std::cerr << "ERROR: Failed to write outputs" << std::endl;
    return -1;
  }

  std::cout << "=== Completed ===" << std::endl;
  return 0;
}

int ActiveGraphMergeTreeSimplifier::loadData(const ActiveGraphSimplificationParams &params) {
  std::cout << "Loading data..." << std::endl;
  postProcess_.setScalarFieldName(params.scalarFieldName);
  postProcess_.setGnnOptions(params.gnnHiddenDim, params.gnnSeed);
  postProcess_.setSimilarityMode(
      params.graphSimilarityMode == ActiveGraphSimplificationParams::GraphSimilarityMode::GNN
          ? ActiveGraphPostProcess::SimilarityMode::GNN
          : ActiveGraphPostProcess::SimilarityMode::Jaccard);

  if (io_.readTreeNodes(params.treeNodesPath) != 0) {
    return -1;
  }

  if (io_.readTreeArcs(params.treeArcsPath) != 0) {
    return -1;
  }

  if (io_.readSegmentation(params.segmentationPath) != 0) {
    return -1;
  }

  if (io_.buildAdjacency() != 0) {
    return -1;
  }

  std::cout << "Data loaded successfully" << std::endl;
  return 0;
}

int ActiveGraphMergeTreeSimplifier::buildTopologicalChains() {
  std::cout << "Extracting topological chains..." << std::endl;
  return postProcess_.extractTopologicalChains();
}

int ActiveGraphMergeTreeSimplifier::simplify() {
  std::cout << "Running greedy simplification..." << std::endl;
  return postProcess_.greedySimplify(params_.epsilonJ, params_.epsilonPos, params_.epsilonBbox);
}

int ActiveGraphMergeTreeSimplifier::writeOutputs(const ActiveGraphSimplificationParams &params) const {
  std::cout << "Writing outputs..." << std::endl;

  const auto &nodes = io_.getNodes();
  const auto &keptNodes = postProcess_.getKeptNodes();

  auto nodesUgrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

  auto points = vtkSmartPointer<vtkPoints>::New();
  points->SetNumberOfPoints(static_cast<vtkIdType>(nodes.size()));

  auto keepArray = vtkSmartPointer<vtkIntArray>::New();
  keepArray->SetName("KeepNode");
  keepArray->SetNumberOfTuples(static_cast<vtkIdType>(nodes.size()));

  vtkIdType idx = 0;
  std::map<int, vtkIdType> nodeIdToIdx;
  for (const auto &[nodeId, node] : nodes) {
    points->SetPoint(idx, node.position.data());
    keepArray->SetValue(idx, keptNodes.count(nodeId) ? 1 : 0);
    nodeIdToIdx[nodeId] = idx;
    idx++;
  }

  nodesUgrid->SetPoints(points);
  nodesUgrid->GetPointData()->AddArray(keepArray);

  // 为每个点创建一个 VTK_VERTEX 单元格，以便 ParaView 能显示点数据
  auto vertexCells = vtkSmartPointer<vtkCellArray>::New();
  for (vtkIdType i = 0; i < static_cast<vtkIdType>(nodes.size()); ++i) {
    vertexCells->InsertNextCell(1, &i);
  }
  nodesUgrid->SetCells(VTK_VERTEX, vertexCells);

  std::string nodesFile = params.outputPrefix + "_simplified_nodes.vtu";
  auto nodesWriter = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  nodesWriter->SetFileName(nodesFile.c_str());
  nodesWriter->SetInputData(nodesUgrid);
  nodesWriter->Write();

  std::cout << "Wrote simplified nodes to " << nodesFile << std::endl;

  const auto &arcs = io_.getArcs();
  const auto &chains = postProcess_.getChains();

  auto arcsUgrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

  auto arcPoints = vtkSmartPointer<vtkPoints>::New();
  auto arcLines = vtkSmartPointer<vtkCellArray>::New();

  auto chainIdArray = vtkSmartPointer<vtkIntArray>::New();
  chainIdArray->SetName("ChainId");
  auto collapsedArray = vtkSmartPointer<vtkIntArray>::New();
  collapsedArray->SetName("CollapsedArcCount");
  auto downNodeIdArray = vtkSmartPointer<vtkIntArray>::New();
  downNodeIdArray->SetName("downNodeId");
  auto upNodeIdArray = vtkSmartPointer<vtkIntArray>::New();
  upNodeIdArray->SetName("upNodeId");

  vtkIdType pointOffset = 0;
  for (const auto &chain : chains) {
    std::vector<int> keptInChain;
    for (int nodeId : chain.nodeIds) {
      if (keptNodes.count(nodeId)) {
        keptInChain.push_back(nodeId);
      }
    }

    for (size_t i = 0; i + 1 < keptInChain.size(); ++i) {
      int downId = keptInChain[i];
      int upId = keptInChain[i + 1];

      auto downIt = nodes.find(downId);
      auto upIt = nodes.find(upId);
      if (downIt == nodes.end() || upIt == nodes.end()) {
        continue;
      }

      arcPoints->InsertNextPoint(downIt->second.position.data());
      arcPoints->InsertNextPoint(upIt->second.position.data());

      auto line = vtkSmartPointer<vtkIdList>::New();
      line->InsertNextId(pointOffset);
      line->InsertNextId(pointOffset + 1);
      arcLines->InsertNextCell(line);

      chainIdArray->InsertNextValue(chain.chainId);
      collapsedArray->InsertNextValue(static_cast<int>(keptInChain.size()) - 2);
      downNodeIdArray->InsertNextValue(downId);
      upNodeIdArray->InsertNextValue(upId);

      pointOffset += 2;
    }
  }

  arcsUgrid->SetPoints(arcPoints);
  arcsUgrid->SetCells(VTK_LINE, arcLines);
  arcsUgrid->GetCellData()->AddArray(chainIdArray);
  arcsUgrid->GetCellData()->AddArray(collapsedArray);
  arcsUgrid->GetCellData()->AddArray(downNodeIdArray);
  arcsUgrid->GetCellData()->AddArray(upNodeIdArray);

  std::string arcsFile = params.outputPrefix + "_simplified_arcs.vtu";
  auto arcsWriter = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  arcsWriter->SetFileName(arcsFile.c_str());
  arcsWriter->SetInputData(arcsUgrid);
  arcsWriter->Write();

  std::cout << "Wrote simplified arcs to " << arcsFile << std::endl;

  std::string statsFile = params.outputPrefix + "_stats.csv";
  std::ofstream statsOut(statsFile);
  statsOut << "chainId,refNodeId,candidateNodeId," << postProcess_.getGraphDistanceName()
           << ",distPos,distBbox,keep" << std::endl;
  for (const auto &decision : postProcess_.getDecisions()) {
    statsOut << decision.chainId << "," << decision.refNodeId << ","
             << decision.candidateNodeId << "," << decision.distJ << ","
             << decision.distPos << "," << decision.distBbox << ","
             << (decision.keep ? 1 : 0) << std::endl;
  }
  statsOut.close();

  std::cout << "Wrote stats to " << statsFile << std::endl;

  return 0;
}

} // namespace active_graph
