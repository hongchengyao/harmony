#' Harmony: fast, accurate, and robust single cell integration.
#'
#' Algorithm for single cell integration.
#'
#' @section Usage:
#'
#' 
#' ?RunHarmony to run Harmony on cell embeddings matrix, Seurat or
#' SingleCellExperiment objects.
#' 
#' @section Useful links:
#'
#' \enumerate{
#' \item Report bugs at \url{https://github.com/immunogenomics/harmony/issues}
#' \item Read the manuscript
#' \doi{10.1038/s41592-019-0619-0}
#' }
#'
#'
#' @name harmony
#' @docType package
#' @useDynLib harmony
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp loadModule
#' @importFrom methods new
#' @importFrom methods as
#' @importFrom methods is
#' @importFrom cowplot plot_grid
#' @importFrom rlang .data
#' @importFrom rlang `%||%`
#' @importFrom stats model.matrix
loadModule("harmony_module", TRUE)
NULL
