#include "SimilarityMetrics.h"
#include "IsmMergeTreeSimplifier.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

double SimilarityMetrics::computeDistISM(const IsmDescriptor &descA,
                                         const IsmDescriptor &descB,
                                         int numBins) {
  // Find min/max of distance fields
  auto [minA, maxA] = std::minmax_element(descA.distanceField.begin(),
                                          descA.distanceField.end());
  auto [minB, maxB] = std::minmax_element(descB.distanceField.begin(),
                                          descB.distanceField.end());

  float minValA = *minA;
  float maxValA = *maxA;
  float minValB = *minB;
  float maxValB = *maxB;

  // Handle degenerate cases
  if (maxValA - minValA < 1e-6f || maxValB - minValB < 1e-6f) {
    return 0.0;
  }

  // Compute mutual information
  double mi = computeMutualInformation(descA.distanceField, descB.distanceField,
                                       numBins, minValA, maxValA, minValB, maxValB);

  // Compute individual entropies
  std::vector<double> histA, histB;
  computeHistogram(descA.distanceField, numBins, minValA, maxValA, histA);
  computeHistogram(descB.distanceField, numBins, minValB, maxValB, histB);

  double hA = computeEntropy(histA);
  double hB = computeEntropy(histB);

  // Compute NMI
  double nmi = 0.0;
  if (hA + hB > 1e-10) {
    nmi = 2.0 * mi / (hA + hB);
  }

  // DistISM = 1 - NMI
  return 1.0 - nmi;
}

void SimilarityMetrics::computeHistogram(const std::vector<float> &field,
                                          int numBins,
                                          float minVal,
                                          float maxVal,
                                          std::vector<double> &histogram) {
  histogram.resize(numBins, 0.0);

  float range = maxVal - minVal;
  if (range < 1e-6f) {
    return;
  }

  for (float val : field) {
    int bin = static_cast<int>((val - minVal) / range * (numBins - 1));
    bin = std::clamp(bin, 0, numBins - 1);
    histogram[bin] += 1.0;
  }

  // Normalize
  double total = std::accumulate(histogram.begin(), histogram.end(), 0.0);
  if (total > 0) {
    for (double &h : histogram) {
      h /= total;
    }
  }
}

double SimilarityMetrics::computeEntropy(const std::vector<double> &histogram) {
  double entropy = 0.0;
  for (double p : histogram) {
    if (p > 1e-10) {
      entropy -= p * std::log2(p);
    }
  }
  return entropy;
}

double SimilarityMetrics::computeMutualInformation(
    const std::vector<float> &fieldA,
    const std::vector<float> &fieldB,
    int numBins,
    float minValA,
    float maxValA,
    float minValB,
    float maxValB) {

  // Build joint histogram
  std::vector<double> jointHist(numBins * numBins, 0.0);

  float rangeA = maxValA - minValA;
  float rangeB = maxValB - minValB;

  if (rangeA < 1e-6f || rangeB < 1e-6f) {
    return 0.0;
  }

  for (size_t i = 0; i < fieldA.size() && i < fieldB.size(); i++) {
    int binA = static_cast<int>((fieldA[i] - minValA) / rangeA * (numBins - 1));
    int binB = static_cast<int>((fieldB[i] - minValB) / rangeB * (numBins - 1));
    binA = std::clamp(binA, 0, numBins - 1);
    binB = std::clamp(binB, 0, numBins - 1);
    jointHist[binA * numBins + binB] += 1.0;
  }

  // Normalize joint histogram
  double total = std::accumulate(jointHist.begin(), jointHist.end(), 0.0);
  if (total > 0) {
    for (double &h : jointHist) {
      h /= total;
    }
  }

  // Compute marginals
  std::vector<double> marginalA(numBins, 0.0);
  std::vector<double> marginalB(numBins, 0.0);

  for (int a = 0; a < numBins; a++) {
    for (int b = 0; b < numBins; b++) {
      marginalA[a] += jointHist[a * numBins + b];
      marginalB[b] += jointHist[a * numBins + b];
    }
  }

  // Compute MI: sum p(a,b) * log(p(a,b) / (p(a) * p(b)))
  double mi = 0.0;
  for (int a = 0; a < numBins; a++) {
    for (int b = 0; b < numBins; b++) {
      double pab = jointHist[a * numBins + b];
      double pa = marginalA[a];
      double pb = marginalB[b];

      if (pab > 1e-10 && pa > 1e-10 && pb > 1e-10) {
        mi += pab * std::log2(pab / (pa * pb));
      }
    }
  }

  return mi;
}