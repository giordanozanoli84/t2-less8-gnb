//
// Created by LEI00017 on 27/01/2021.
//

#ifndef GAUSSIAN_NAIVE_BAYES_GNB_H
#define GAUSSIAN_NAIVE_BAYES_GNB_H


#include <string>
#include <vector>
#include <map>
#include "Eigen/Dense"

using std::string;
using std::vector;
using std::map;
typedef Eigen::Array<double, 1, Eigen::Dynamic> Array1d;

class GNB {
 public:
  /**
   * Constructor
   */
  GNB(int d, const vector<string>& possible_labels);

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Train classifier
   */
  void train(const vector<vector<double> > &data,
             const vector<string> &labels);

  /**
   * Predict with trained classifier
   */
  string predict(const vector<double> &sample);

 private:
  vector<Array1d> means_;
  vector<Array1d> vars_;
  Array1d sizes_;
  Array1d priors_;
  map<string, int> labels_dict_;
  map<int, string> inverse_labels_dict_;
  int classes_;
  int d_;

  // auxiliary vars to compute online statistics
  void online_stats(Array1d &new_vals, int class_);
  vector<Array1d> online_sum_of_squares_;
};


#endif //GAUSSIAN_NAIVE_BAYES_GNB_H
