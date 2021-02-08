//
// Created by LEI00017 on 08/02/2021.
//

#ifndef GAUSSIAN_NAIVE_BAYES_HELPERS_H
#define GAUSSIAN_NAIVE_BAYES_HELPERS_H

//#define _USE_MATH_DEFINES
#include <cmath>
#include "Eigen/Dense"

typedef Eigen::Array<double, 1, Eigen::Dynamic> Array1d;
static const double STATIC_ONE_OVER_SQRT_2PI = 1.0 / sqrt(2.0*M_PI);

class Helpers {
 public:

  // norm pdf 1d
  static double normpdf1d(double mu, double sigma, double x);
  static Array1d normpdf1d(const Array1d& mu, const Array1d& sigma, const Array1d& x);

};


#endif //GAUSSIAN_NAIVE_BAYES_HELPERS_H
