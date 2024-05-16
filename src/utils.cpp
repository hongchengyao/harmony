// [[Rcpp::depends(RcppArmadillo)]]
#include "utils.h"
#include "types.h"
#include <RcppArmadilloExtensions/sample.h>

//[[Rcpp::export]]
arma::mat kmeans_centers(const arma::mat& X, const int K) {
  
  // Environment 
  Rcpp::Environment stats_env("package:stats");
  // Cast function as callable from C++
  Rcpp::Function kmeans = stats_env["kmeans"];
  // Call the function and receive its list output
  Rcpp::List res = kmeans(Rcpp::_["x"] = X.t(),
                          Rcpp::_["centers"] = K,
                          Rcpp::_["iter.max"] = 25,
                          Rcpp::_["nstart"] = 10
                          );
  return res["centers"];
}


MATTYPE safe_entropy(const MATTYPE& X) {
  MATTYPE A = X % log(X);
  A.elem(find_nonfinite(A)).zeros();
  return(A);
}

// Overload pow to work on a MATTYPErix and vector
MATTYPE harmony_pow(MATTYPE A, const VECTYPE& T) {

  for (unsigned c = 0; c < A.n_cols; c++) {
    A.unsafe_col(c) = pow(A.unsafe_col(c), as_scalar(T.row(c)));
  }
  return(A);
}

VECTYPE calculate_norm(const MATTYPE& M) {
  VECTYPE x(M.n_cols);
  for(unsigned i = 0; i < M.n_cols; i++){
    x(i) = norm(M.col(i));
  }
  return x;
}


//https://stackoverflow.com/questions/8377412/ceil-function-how-can-we-implement-it-ourselves
int my_ceil(float num) {
    int inum = (int)num;
    if (num == (float)inum) {
        return inum;
    }
    return inum + 1;
}


// [[Rcpp::export]]
MATTYPE scaleRows_dgc(const VECTYPE& x, const VECTYPE& p, const VECTYPE& i, int ncol, int nrow, float thresh) {
  
    // (0) fill in non-zero elements
    MATTYPE res = arma::zeros<MATTYPE>(nrow, ncol);
    for (int c = 0; c < ncol; c++) {
        for (int j = p[c]; j < p[c + 1]; j++) {
            res(i[j], c) = x(j);
        }
    }

    // (1) compute means
    VECTYPE mean_vec = arma::zeros<VECTYPE>(nrow);
    for (int c = 0; c < ncol; c++) {
        for (int j = p[c]; j < p[c + 1]; j++) {
            mean_vec(i[j]) += x[j];
        }
    }
    mean_vec /= ncol;

    // (2) compute SDs
    VECTYPE sd_vec = arma::zeros<VECTYPE>(nrow);
    arma::uvec nz = arma::zeros<arma::uvec>(nrow);
    nz.fill(ncol);
    for (int c = 0; c < ncol; c++) {
        for (int j = p[c]; j < p[c + 1]; j++) {
            sd_vec(i[j]) += (x[j] - mean_vec(i[j])) * (x[j] - mean_vec(i[j])); // (x - mu)^2
            nz(i[j])--;
        }
    }

    // count for the zeros
    for (int r = 0; r < nrow; r++) {
        sd_vec(r) += nz(r) * mean_vec(r) * mean_vec(r);
    }

    sd_vec = arma::sqrt(sd_vec / (ncol - 1));

    // (3) scale values
    res.each_col() -= mean_vec;
    res.each_col() /= sd_vec;
    res.elem(find(res > thresh)).fill(thresh);
    res.elem(find(res < -thresh)).fill(-thresh);
    return res;
}


// [[Rcpp::export]]
arma::vec find_lambda_cpp(const float alpha, const arma::vec& cluster_E) {
  arma::vec lambda_dym_vec(cluster_E.n_rows + 1, arma::fill::zeros);
  lambda_dym_vec.subvec(1, lambda_dym_vec.n_rows - 1) = cluster_E * alpha;
  return lambda_dym_vec;
}



// [[Rcpp::export]]
arma::mat make_R_hard(const arma::mat& R){
    int nCols = R.n_cols;
    int nRows = R.n_rows;
    arma::uvec index(nCols);
    // Sample one index per column based on probabilities in each column
    for(int j = 0; j<nCols;j++){
        arma::vec probs = R.col(j);
        index(j) = Rcpp::RcppArmadillo::sample(arma::linspace<arma::uvec>(0, nRows-1, nRows), 1, FALSE, probs)(0);
    }
    arma::mat R_hard = arma::zeros<arma::mat>(nRows, nCols);
    for (int j = 0; j < nCols; j++){
        R_hard(index(j), j) = 1;
    }
    return R_hard;
}



// [[Rcpp::export]]
arma::mat sampleIdxAndWeight(const arma::mat& R_hard, const arma::sp_mat& Phi, const float sample_num){
    int nRows = R_hard.n_rows;
    int bRows = Phi.n_rows;

    std::vector<double> all_S;
    std::vector<double> all_weights;
    std::vector<double> all_batch;
    for(int b = 0; b< bRows; b++){
        // arma::rowvec Phi_b = Phi.row(b);
        arma::sp_mat Phi_b(Phi.n_cols, Phi.n_cols);
        Phi_b.diag() = Phi.row(b);
        // arma::mat R_hard_b = R_hard.each_row() % Phi_b;
        arma::mat R_hard_b = R_hard * Phi_b;
        for (int k = 0;k<nRows; k++){
            arma::uvec nonzeros = arma::find(R_hard_b.row(k) != 0);
            int nonzero_count = nonzeros.n_rows;
            double weight_bk = 1.0;
            if (nonzero_count > sample_num){
                weight_bk = nonzero_count / sample_num;
                nonzeros = arma::shuffle(nonzeros);
                nonzeros = nonzeros.subvec(0, sample_num - 1);
            }
            for (unsigned i = 0; i< nonzeros.n_rows; i++){
                all_S.push_back(nonzeros(i));
                all_weights.push_back(weight_bk);
                all_batch.push_back(b);
            }
        }
    }
    arma::mat result(all_S.size(),3);
    result.col(0) = arma::vec(all_S);
    result.col(1) = arma::vec(all_weights);
    result.col(2) = arma::vec(all_batch);
    return result;
}

