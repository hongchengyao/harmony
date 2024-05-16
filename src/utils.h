#pragma once
#include "types.h"
#include <RcppArmadillo.h>

arma::mat kmeans_centers(const arma::mat& X, const int K);

MATTYPE safe_entropy(const MATTYPE& X);

MATTYPE harmony_pow(MATTYPE A, const VECTYPE& T);

VECTYPE calculate_norm(const MATTYPE& M);


int my_ceil(float num);


arma::vec find_lambda_cpp(const float alpha, const arma::vec& cluster_E);

arma::mat make_R_hard(const arma::mat& R);
arma::mat sampleIdxAndWeight(const arma::mat& R_hard, const arma::sp_mat& Phi, const float sample_num);