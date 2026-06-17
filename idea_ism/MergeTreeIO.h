#pragma once

#include <string>
#include <vector>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

// Forward declarations
struct TreeNode;
struct TreeArc;
struct TopologicalChain;

class MergeTreeIO {
public:
  static std::vector<TreeNode> readNodesFromVTU(vtkUnstructuredGrid *grid);
  static std::vector<TreeArc> readArcsFromVTU(vtkUnstructuredGrid *grid);

  static void writeSimplifiedNodes(vtkUnstructuredGrid *originalGrid,
                                   const std::vector<int> &keptNodeIds,
                                   const std::string &path);

  static void writeGrid(vtkSmartPointer<vtkUnstructuredGrid> grid,
                        const std::string &path);

  static void writeStatsCSV(const std::string &path,
                            const std::vector<TopologicalChain> &chains,
                            const std::vector<int> &keptNodeIds,
                            const std::vector<double> &distToRef);
};