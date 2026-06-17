#pragma once

#include <vector>

// Forward declarations
struct IsmDescriptor;

class SimilarityMetrics {
public:
  static double computeDistISM(const IsmDescriptor &descA,
                               const IsmDescriptor &descB,
                               int numBins);

private:
  static void computeHistogram(const std::vector<float> &field,
                               int numBins,
                               float minVal,
                               float maxVal,
                               std::vector<double> &histogram);

  static double computeEntropy(const std::vector<double> &histogram);

  static double computeMutualInformation(const std::vector<float> &fieldA,
                                         const std::vector<float> &fieldB,
                                         int numBins,
                                         float minValA,
                                         float maxValA,
                                         float minValB,
                                         float maxValB);
};