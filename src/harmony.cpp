#include <algorithm>
#include <chrono>

#include "harmony.h"
#include "types.h"
#include "utils.h"




harmony::harmony() :
    window_size(3),
    ran_setup(false),
    ran_init(false),
    lambda_estimation(false),
    verbose(false)
    
{}



void harmony::setup(const MATTYPE& __Z, const arma::sp_mat& __Phi,
                    // const VECTYPE __sigma, const VECTYPE __theta, const VECTYPE __lambda, const float __alpha, const int __max_iter_kmeans,
                    const VECTYPE __sigma, const VECTYPE __theta, const VECTYPE __lambda, const float __alpha, const int __max_iter_harmony,
                    const float __epsilon_kmeans, const float __epsilon_harmony,
                    const int __K, const float __block_size,
                    const std::vector<int>& __B_vec, const bool __verbose) {
    
  // Algorithm constants
  N = __Z.n_cols;
  B = __Phi.n_rows;
  d = __Z.n_rows;
  
  Z_orig = __Z;
  Z_cos = arma::normalise(__Z, 2, 0);
  Z_corr = zeros(size(Z_orig));

  
  Phi = __Phi;
  Phi_t = Phi.t();
  
  // Create index
  std::vector<unsigned>counters;
  arma::vec sizes(sum(Phi, 1));
  // std::cout << sizes << std::endl;
  for (unsigned i = 0; i < sizes.n_elem; i++) {
    arma::uvec a(int(sizes(i)));
    index.push_back(a);
    counters.push_back(0);
  }

  arma::sp_mat::const_iterator it =     Phi.begin();
  arma::sp_mat::const_iterator it_end = Phi.end();
  for(; it != it_end; ++it)
  {
    unsigned int row_idx = it.row();
    unsigned int col_idx = it.col();
    index[row_idx](counters[row_idx]++) = col_idx;
  }

  Pr_b = sum(Phi, 1) / N;

  
  epsilon_kmeans = __epsilon_kmeans;
  epsilon_harmony = __epsilon_harmony;

  // Hyperparameters
  K = __K;
  if (__lambda(0) == -1) {
    lambda_estimation = true;
  } else {
    lambda = __lambda;
  }
  B_vec = __B_vec;
  sigma = __sigma;

  if(__Z.n_cols < 6) {
    std::string error_message = "Refusing to run with less than 6 cells";
    Rcpp::stop(error_message);
  } else if (__Z.n_cols < 40) {
    Rcpp::warning("Too few cells. Setting block_size to 0.2");
    block_size = 0.2;
  } else {
    block_size = __block_size;
  } 
  theta = __theta;
  // max_iter_kmeans = __max_iter_kmeans;
  max_iter_harmony = __max_iter_harmony;

  verbose = __verbose;
  
  allocate_buffers();
  ran_setup = true;

  alpha = __alpha;
  
  
}


void harmony::allocate_buffers() {
  
  _scale_dist = zeros<MATTYPE>(K, N);
  dist_mat = zeros<MATTYPE>(K, N);
  O = E = zeros<MATTYPE>(K, B);
  
  // Hack: create matrix of ones by creating zeros and then add one!
  arma::sp_mat intcpt = zeros<arma::sp_mat>(1, N);
  intcpt = intcpt+1;
  
  Phi_moe = join_cols(intcpt, Phi);
  Phi_moe_t = Phi_moe.t();


  W = zeros<MATTYPE>(B + 1, d);
  
  // Need a W_cube to store all the beta during iterations
  W_cube = zeros<CUBETYPE>(B+1, d, K); 
}


void harmony::init_cluster_cpp() {

  Y = kmeans_centers(Z_cos, K).t();
  
  // Cosine normalization of data centrods
  Y = arma::normalise(Y, 2, 0);

  // (2) ASSIGN CLUSTER PROBABILITIES
  // using a nice property of cosine distance,
  // compute squared distance directly with cross product
  dist_mat = 2 * (1 - Y.t() * Z_cos);
  
  R = -dist_mat;
  R.each_col() /= sigma;
  R = exp(R);
  R.each_row() /= sum(R, 0);
  
  
  // (3) BATCH DIVERSITY STATISTICS
  E = sum(R, 1) * Pr_b.t();
  O = R * Phi_t;
  
  R_theta = R;  // Initialize R_theta to be R
  
  compute_objective();
  objective_harmony.push_back(objective_kmeans.back());
  
  dist_mat = 2 * (1 - Y.t() * Z_cos); // Z_cos was changed HCYAO: I think this line is unnecessary

  ran_init = true;
  
}


// Keep this function for now
void harmony::compute_objective() { 
  const float norm_const = 2000/((float)N);
  float kmeans_error = as_scalar(accu(R_theta % dist_mat));  
  float _entropy = as_scalar(accu(safe_entropy(R_theta).each_col() % sigma)); // NEW: vector sigma
  float _cross_entropy = as_scalar(
      accu((R_theta.each_col() % sigma) % ((arma::repmat(theta.t(), K, 1) % log((O + E) / E)) * Phi)));

  // Push back the data
  objective_kmeans.push_back((kmeans_error + _entropy + _cross_entropy) * norm_const);
  objective_kmeans_dist.push_back(kmeans_error * norm_const);
  objective_kmeans_entropy.push_back(_entropy * norm_const);
  objective_kmeans_cross.push_back(_cross_entropy * norm_const);
}

// Keep this function for now
bool harmony::check_convergence(int type) {
  float obj_new, obj_old;
  switch (type) {
    case 0: 
      // Clustering 
      // compute new window mean
      obj_old = 0;
      obj_new = 0;
      for (unsigned i = 0; i < window_size; i++) {
        obj_old += objective_kmeans[objective_kmeans.size() - 2 - i];
        obj_new += objective_kmeans[objective_kmeans.size() - 1 - i];
      }
      if ((obj_old - obj_new) / abs(obj_old) < epsilon_kmeans) {
        return(true); 
      } else {
        return(false);
      }
    case 1:
      // Harmony
      obj_old = objective_harmony[objective_harmony.size() - 2];
      obj_new = objective_harmony[objective_harmony.size() - 1];
      if ((obj_old - obj_new) / abs(obj_old) < epsilon_harmony) {
        return(true);              
      } else {
        return(false);              
      }
  }
  
  // gives warning if we don't give default return value
  return(true);
}


int harmony::main_loop_cpp(){
  moe_ridge_update_betas_cpp(); // update W_cube
  compute_objective();
  kmeans_rounds.push_back(1);
  objective_harmony.push_back(objective_kmeans.back());
  update_R0(); // update R
  update_R_theta(); // update R_theta and E and O by online algorithm
  
  return 0;
}


void harmony::moe_ridge_update_betas_cpp() {
  // CUBETYPE W_cube(B+1, d, K); // rows, cols, slices

  arma::sp_mat _Rk(N, N);
  arma::sp_mat lambda_mat(B + 1, B + 1);

  if (!lambda_estimation) {
    // Set lambda if we have to
    lambda_mat.diag() = lambda;
  }

  for (unsigned k = 0; k < K; k++) {
      _Rk.diag() = R_theta.row(k);
      if (lambda_estimation){
        lambda_mat.diag() = find_lambda_cpp(alpha, E.row(k).t()); 
      }
      arma::sp_mat Phi_Rk = Phi_moe * _Rk;
      W_cube.slice(k) = arma::inv(arma::mat(Phi_Rk * Phi_moe_t + lambda_mat)) * Phi_Rk * Z_orig.t();
  }
  Y = W_cube.row(0);
  Y = arma::normalise(Y, 2, 0);
  Z_cos = arma::normalise(Z_orig, 2, 0);
  dist_mat = 2 * (1 - Y.t() * Z_cos);
  // return W_cube;
}


// To update R_0
void harmony::update_R0(){
  for (unsigned k = 0; k < K; k++) { 
    Z_corr = Z_orig;
    W = W_cube.slice(k);
    Yk_t = W.row(0); // set intercept as centroid
    Yk_t = arma::normalise(Yk_t, 2);
    W.row(0).zeros(); // don't remove the intercept
    Z_corr -= W.t() * Phi_moe; // assume all cells belong to cluster k,  move all cells by W
    Z_cos = arma::normalise(Z_corr, 2, 0);  // NOTE, this could be slow as repeated K times
    R.row(k) = -2 * (1 - Yk_t * Z_cos); 
  }

  R.each_col() /= sigma;
  R = exp(R);
  R.each_row() /= sum(R, 0);

  E = sum(R, 1) * Pr_b.t();
  O = R * Phi_t;
  // I feel that I shouldn't update E and O here based on R because I think E and O is only
  // used for estimate R_theta and their value should be determined by R_theta
}


int harmony::update_R_theta(){
  // Generate the 0,N-1 indices
  uvec indices = linspace<uvec>(0, N - 1, N);
  update_order = shuffle(indices);
  
  // Inverse index
  uvec reverse_index(N, arma::fill::zeros);
  reverse_index.rows(update_order) = indices;
  
  // _scale_dist = -dist_mat; // K x N
  // _scale_dist.each_col() /= sigma; // NEW: vector sigma
  // _scale_dist = exp(_scale_dist);
  // _scale_dist = arma::normalise(_scale_dist, 1, 0); // HCYAO why do we want L1 normalization here?
  
  // GENERAL CASE: online updates, in blocks of size (N * block_size)
  unsigned n_blocks = (int)(my_ceil(1.0 / block_size));
  unsigned cells_per_block = unsigned(N * block_size);
  
  // Allocate new matrices
  MATTYPE R_randomized = R_theta.cols(update_order);
  arma::sp_mat Phi_randomized(Phi.cols(update_order));
  arma::sp_mat Phi_t_randomized(Phi_randomized.t());
  // Use R instead of _scale_dist, here R is just L1 normalized _scale_dist, 
  // which is the same as former code (but redundant), and won't affect result
  MATTYPE _scale_dist_randomized = R.cols(update_order); 
  for (unsigned i = 0; i < n_blocks; i++) {
    unsigned idx_min = i*cells_per_block;
    unsigned idx_max = ((i+1) * cells_per_block) - 1; // - 1 because of submat
    if (i == n_blocks-1) {
      // we are in the last block, so include everything. Up to 19
      // extra cells.
      idx_max = N - 1;
    }

    auto Rcells = R_randomized.submat(0, idx_min, R_randomized.n_rows - 1, idx_max);
    auto Phicells = Phi_randomized.submat(0, idx_min, Phi_randomized.n_rows - 1, idx_max);
    auto Phi_tcells = Phi_t_randomized.submat(idx_min, 0, idx_max, Phi_t_randomized.n_cols - 1);
    auto _scale_distcells = _scale_dist_randomized.submat(0, idx_min, _scale_dist_randomized.n_rows - 1, idx_max);

    // Step 1: remove cells
    E -= sum(Rcells, 1) * Pr_b.t(); // I feel that this is worng, should minus based on R0
    O -= Rcells * Phi_tcells;

    // Step 2: recompute R for removed cells
    Rcells = _scale_distcells;
    Rcells = Rcells % (harmony_pow(E/(O + E), theta) * Phicells);
    Rcells = normalise(Rcells, 1, 0); // L1 norm columns

    // Step 3: put cells back 
    E += sum(Rcells, 1) * Pr_b.t();
    O += Rcells * Phi_tcells;
  }
  this->R_theta = R_randomized.cols(reverse_index);
  return 0;
}

// get final Z_corr based on R (not R_theta)
// void harmony::moe_correct_ridge_cpp(){ 
//   arma::sp_mat _Rk(N, N);
//   arma::sp_mat lambda_mat(B + 1, B + 1);
//   if(!lambda_estimation) {
//     // Set lambda if we have to
//     lambda_mat.diag() = lambda;
//   }
//   Z_corr = Z_orig;
//   Progress p(K, verbose);
//   for (unsigned k = 0; k < K; k++) {
//     p.increment();
//     if (Progress::check_abort())
//       return;
//     if (lambda_estimation) {
//       lambda_mat.diag() = find_lambda_cpp(alpha, E.row(k).t());
//     }
//     _Rk.diag() = R.row(k);
//     arma::sp_mat Phi_Rk = Phi_moe * _Rk;

//     arma::mat inv_cov(arma::inv(arma::mat(Phi_Rk * Phi_moe_t + lambda_mat)));

//     // Calculate R-scaled PCs once
//     arma::mat Z_tmp = Z_orig.each_row() % R.row(k);
    
//     // Generate the betas contribution of the intercept using the data
//     // This erases whatever was written before in W
//     W = inv_cov.unsafe_col(0) * sum(Z_tmp, 1).t();

//     // Calculate betas by calculating each batch contribution
//     for(unsigned b=0; b < B; b++) {
//       // inv_conv is B+1xB+1 whereas index is B long
//       W += inv_cov.unsafe_col(b+1) * sum(Z_tmp.cols(index[b]), 1).t();
//     }
    
//     W.row(0).zeros(); // do not remove the intercept
//     Z_corr -= W.t() * Phi_Rk;
//   }
// }


void harmony::moe_correct_ridge_cpp(){ 
  arma::sp_mat _Rk(N, N);
  arma::sp_mat lambda_mat(B + 1, B + 1);
  if(!lambda_estimation) {
    // Set lambda if we have to
    lambda_mat.diag() = lambda;
  }
  Z_corr = Z_orig;
  Progress p(K, verbose);
  for (unsigned k = 0; k < K; k++) {
    p.increment();
    if (Progress::check_abort())
      return;
    if (lambda_estimation) {
      lambda_mat.diag() = find_lambda_cpp(alpha, E.row(k).t());
    }
    _Rk.diag() = R.row(k);
    arma::sp_mat Phi_Rk = Phi_moe * _Rk;

    arma::mat inv_cov(arma::inv(arma::mat(Phi_Rk * Phi_moe_t + lambda_mat)));

    // Calculate R-scaled PCs once
    arma::mat Z_tmp = Z_orig.each_row() % R.row(k);
    
    // Generate the betas contribution of the intercept using the data
    // This erases whatever was written before in W
    W = inv_cov.unsafe_col(0) * sum(Z_tmp, 1).t();

    // Calculate betas by calculating each batch contribution
    for(unsigned b=0; b < B; b++) {
      // inv_conv is B+1xB+1 whereas index is B long
      W += inv_cov.unsafe_col(b+1) * sum(Z_tmp.cols(index[b]), 1).t();
    }
    // W = arma::inv(arma::mat(Phi_Rk * Phi_moe_t + lambda_mat)) * Phi_Rk * Z_orig.t();
    W.row(0).zeros(); // do not remove the intercept
    Z_corr -= W.t() * Phi_Rk;
  }
}




RCPP_MODULE(harmony_module) {
  class_<harmony>("harmony")
      .constructor()
      .field("Z_corr", &harmony::Z_corr)
      .field("Z_cos", &harmony::Z_cos)
      .field("Z_orig", &harmony::Z_orig)
      .field("Phi", &harmony::Phi)
      .field("Phi_moe", &harmony::Phi_moe)
      .field("N", &harmony::N)
      .field("B", &harmony::B)
      .field("K", &harmony::K)
      .field("d", &harmony::d)
      .field("O", &harmony::O)
      .field("E", &harmony::E)
      .field("Y", &harmony::Y)
      .field("Pr_b", &harmony::Pr_b)
      .field("W", &harmony::W)
      .field("W_cube", &harmony::W_cube)
      .field("R", &harmony::R)
      .field("R_theta", &harmony::R_theta)
      .field("theta", &harmony::theta)
      .field("sigma", &harmony::sigma)
      .field("lambda", &harmony::lambda)
      .field("kmeans_rounds", &harmony::kmeans_rounds)
      .field("objective_kmeans", &harmony::objective_kmeans)
      .field("objective_kmeans_dist", &harmony::objective_kmeans_dist)
      .field("objective_kmeans_entropy", &harmony::objective_kmeans_entropy)
      .field("objective_kmeans_cross", &harmony::objective_kmeans_cross)    
      .field("objective_harmony", &harmony::objective_harmony)
      // .field("max_iter_kmeans", &harmony::max_iter_kmeans)
      .method("check_convergence", &harmony::check_convergence)
      .method("setup", &harmony::setup)
      .method("compute_objective", &harmony::compute_objective)
      .method("init_cluster_cpp", &harmony::init_cluster_cpp)
      // .method("cluster_cpp", &harmony::cluster_cpp)	  
      .method("moe_correct_ridge_cpp", &harmony::moe_correct_ridge_cpp)
      .method("moe_ridge_update_betas_cpp", &harmony::moe_ridge_update_betas_cpp)
      .method("update_R0", &harmony::update_R0)
      .method("update_R_theta", &harmony::update_R_theta)
      .method("main_loop_cpp", &harmony::main_loop_cpp)
      // .method("moe_ridge_get_betas_cpp", &harmony::moe_ridge_get_betas_cpp)
      .field("B_vec", &harmony::B_vec)
      .field("alpha", &harmony::alpha)
      ;
}