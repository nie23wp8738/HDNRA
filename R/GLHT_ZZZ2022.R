#' @title Test proposed by Zhu et al. (2022)
#' @description
#' Zhu et al. (2022)'s test for general linear hypothesis testing (GLHT) problem for high-dimensional data with assuming that underlying covariance matrices are the same.
#'
#' @usage glht_zzz2022(Y,X,C)
#' @param Y An \eqn{n\times p} response matrix obtained by independently observing a \eqn{p}-dimensional response variable for \eqn{n} subjects.
#' @param X A known \eqn{n\times k} full-rank design matrix with \eqn{\operatorname{rank}(\boldsymbol{G})=k<n-2}.
#' @param C A known matrix of size \eqn{q\times k} with \eqn{\operatorname{rank}(\boldsymbol{C})=q<k}.

#'
#' @details
#' A high-dimensional linear regression model can be expressed as
#' \deqn{\boldsymbol{Y}=\boldsymbol{X\Theta}+\boldsymbol{\epsilon},}
#' where \eqn{\Theta} is a \eqn{k\times p} unknown parameter matrix and \eqn{\boldsymbol{\epsilon}} is an \eqn{n\times p} error matrix.
#'
#' It is of interest to test the following GLHT problem
#' \deqn{H_0: \boldsymbol{C\Theta}=\boldsymbol{0}, \quad \text { vs. } H_1: \boldsymbol{C\Theta} \neq \boldsymbol{0}.}
#'
#' Zhu et al. (2022) proposed the following test statistic:
#' \deqn{T_{ZZZ}=\frac{(n-k-2)}{(n-k)pq}\operatorname{tr}(\boldsymbol{S}_h\boldsymbol{D}^{-1}),}
#' where \eqn{\boldsymbol{S}_h} and \eqn{\boldsymbol{S}_e} are the variation matrices due to the hypothesis and error, respectively, and \eqn{\boldsymbol{D}} is the diagonal matrix with the diagonal elements of \eqn{\boldsymbol{S}_e/(n-k)}.
#' They showed that under the null hypothesis, \eqn{T_{ZZZ}} and a chi-squared-type mixture have the same limiting distribution.

#' @references
#' \insertRef{Zhu_2023}{HDNRA}
#'
#' @return A  (list) object of  \code{S3} class \code{htest}  containing the following elements:
#' \describe{
#' \item{p.value}{the \eqn{p}-value of the test proposed by Zhu et al. (2022)}
#' \item{statistic}{the test statistic proposed by Zhu et al. (2022).}
#' \item{df}{estimated approximate degrees of freedom of Zhu et al. (2022)'s test.}
#' }

#' @examples
#' set.seed(1234)
#' k <- 3
#' q <- k - 1
#' p <- 50
#' n <- c(25, 30, 40)
#' rho <- 0.01
#' Theta <- matrix(rep(0, k * p), nrow = k)
#' X <- matrix(c(rep(1, n[1]), rep(0, sum(n)), rep(1, n[2]), rep(0, sum(n)), rep(1, n[3])),
#'   ncol = k, nrow = sum(n)
#' )
#' y <- (-2 * sqrt(1 - rho) + sqrt(4 * (1 - rho) + 4 * p * rho)) / (2 * p)
#' x <- y + sqrt((1 - rho))
#' Gamma <- matrix(rep(y, p * p), nrow = p)
#' diag(Gamma) <- rep(x, p)
#' U <- matrix(ncol = sum(n), nrow = p)
#' for (i in 1:sum(n)) {
#'   U[, i] <- rnorm(p, 0, 1)
#' }
#' Y <- X %*% Theta + t(U) %*% Gamma
#' C <- cbind(diag(q), -rep(1, q))
#' glht_zzz2022(Y, X, C)
#'
#' @concept nraglht
#' @export
glht_zzz2022 <- function(Y, X, C) {
  stats <- glht_zzz2022_cpp(Y, X, C)
  stat <- stats[1]
  df <- stats[2]
  n <- dim(Y)[1]
  k <- dim(X)[2]
  statnew <- stat * (n - k - 2) / (n - k)
  dhatnew <- df * (n - k)^2 / ((n - k - 2)^2)
  pvalue <- 1 - pchisq(dhatnew * statnew, dhatnew)
  names(statnew) <- "statistic"
  names(dhatnew) <- "df"
  res <- list(statistic = statnew, p.value = pvalue, parameter = dhatnew)
  class(res) <- "htest"
  return(res)
}
