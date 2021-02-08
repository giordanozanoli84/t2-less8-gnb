//
// Created by LEI00017 on 27/01/2021.
//

#include <math.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "gnb.h"
#include "helpers.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

// Initializes GNB
GNB::GNB(int d, const vector<string>& possible_labels) : d_(d) {
  /**
   * @param d - int
   *   - is the dimensionality of the Xs
   */

  classes_ = possible_labels.size();

  for (int i=0; i < classes_; i++) {
    auto label = possible_labels[i];
    labels_dict_[label] = i;
    inverse_labels_dict_[i] = label;
    means_.emplace_back(Array1d::Zero(d));
    vars_.emplace_back(Array1d::Zero(d));
    online_sum_of_squares_.emplace_back(Array1d::Zero(d));
  }

  sizes_ = Array1d::Zero(classes_);
}

GNB::~GNB() = default;

void GNB::train(const vector<vector<double> > &data,
                const vector<string> &labels) {
  /**
   * Trains the classifier with N data points and labels. Keeps training when
   * new data is provided.
   *
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   */

  size_t N = labels.size();

  for (size_t i=0; i < N; i++) {
    Array1d x_i{d_};
    for (int j=0; j < d_; j++) {
      x_i(j) = data[i][j];
    }
    string l_i{labels[i]};
    int y_i{labels_dict_[l_i]};
    sizes_[y_i] += 1;
    // compute online statistics, means and sum of centered squares
    this->online_stats(x_i, y_i);
  }

  // compute priors
  priors_ = sizes_ / sizes_.sum();

  for (int i = 0; i < classes_; i++) {
    cout << means_[i] << endl;
    cout << vars_[i] << endl;
  }

}

string GNB::predict(const vector<double> &sample) {
  /**
   * Once trained, this method is called and expected to return
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   *
   */

  // vector of double to Array1d, ugly but works fine
  Array1d x{d_};
  for (int i = 0; i < d_; i++) {
    x[i] = sample[i];
  }

  // compute standard deviations for each class
  vector<Array1d> stds;
  for (int i = 0; i < classes_; i++) {
    stds.emplace_back(vars_[i].sqrt());
  }

  // compute likelihood for each class using norm pdf
  Array1d likelihoods = Array1d::Zero(classes_);
  for (int i = 0; i < classes_; i++) {
    likelihoods[i] = Helpers::normpdf1d(means_[i], stds[i], x).matrix().prod();
  }

  // compute posteriors. Do not normalize as we are not returning
  // probability but just the prediction
  auto posteriors = priors_ * likelihoods;

  // take the max index
  double argmax_ = -1;
  double max_ = 0.0;
  double p;
  for (int i = 0; i < classes_; i++) {
    p = posteriors[i];
    if (p > max_) {
      argmax_ = i;
      max_ = p;
    }
  }

  return inverse_labels_dict_[argmax_];
}

void GNB::online_stats(Array1d &new_vals, int class_) {
  /**
   * Implement Welford's online algorithm
   * (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
   */

  int n = sizes_[class_];

  // if first values, just update the mean equal to the values. Std is 0
  if (n < 2) {
    means_[class_] += new_vals;
    return;
  }

  // compute diff x_n - mu_n1
  auto delta1 = new_vals - means_[class_];

  // new mean
  means_[class_] += delta1 / n;

  // compute diff x_n - mu_n
  auto delta2 = new_vals - means_[class_];

  // new sum of centered squares
  online_sum_of_squares_[class_] += (n - 1.0) / n * delta1 * delta2;

  // new var
  vars_[class_] = online_sum_of_squares_[class_] / n;

}
