//
// Created by LEI00017 on 08/02/2021.
//
#include <cmath>
#include "helpers.h"


double Helpers::normpdf1d(double mu, double sigma, double x) {

  if (sigma < 0.0) {
    // @fixme throw value error exception
    return -1;
  } else if (sigma == 0.0) {
    return x == mu ? 1.0 : 0.0;
  }

  double den = STATIC_ONE_OVER_SQRT_2PI / sigma;
  double num = exp(-0.5 * pow((x - mu) / sigma, 2));

  return num * den;
}

Array1d Helpers::normpdf1d(const Array1d& mu, const Array1d& sigma, const Array1d& x) {

  Array1d den = STATIC_ONE_OVER_SQRT_2PI / sigma;
  Array1d num = (-0.5 * ((x - mu) / sigma).pow(2)).exp();

  return num * den;
}


