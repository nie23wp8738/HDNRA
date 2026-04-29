#' @title
#' Normal-approximation-based test for k-sample linear hypothesis via random integration proposed by Li et al. (2025)
#' @description
#' Li et al. (2025)'s test for general linear hypothesis testing (GLHT) problem for high-dimensional data under heteroscedasticity.
#'
#' @usage LHNB2025.GLHTBF.NABT(Y, B, O, A, n, p)
#'
#' @param Y A list of \eqn{k} data matrices. The \eqn{i}th element represents the data matrix
#'   (\eqn{n_i \times p}) from the \eqn{i}th population with each row representing a \eqn{p}-dimensional observation.
#' @param B A vector of \eqn{k} coefficients \eqn{(B_1,\ldots,B_k)} specifying the linear combination of group mean vectors.
#' @param O A length-\eqn{p} vector used to form \eqn{\Omega = \mathrm{diag}(O_1^2,\ldots,O_p^2)}.
#' @param A A length-\eqn{p} vector used in \eqn{W = \Omega + A A^\top}.
#' @param n A vector of \eqn{k} sample sizes. The \eqn{i}th element represents the sample size of group \eqn{i}, \eqn{n_i}.
#' @param p The dimension of data.
#'
#'
#' @details
#' Suppose we have \eqn{k} independent high-dimensional samples
#' \deqn{\boldsymbol{Y}_{i1},\ldots,\boldsymbol{Y}_{in_i}\ \text{are i.i.d. with}\ \mathrm{E}(\boldsymbol{Y}_{i1})=\boldsymbol{\mu}_i,\
#' \mathrm{Cov}(\boldsymbol{Y}_{i1})=\boldsymbol{\Sigma}_i,\ i=1,\ldots,k,}
#' where the covariance matrices \eqn{\boldsymbol{\Sigma}_i} may differ across groups.
#'
#' It is of interest to test the k-sample linear hypothesis
#' \deqn{H_0:\ \sum_{i=1}^k B_i\boldsymbol{\mu}_i=\boldsymbol{0}\quad \text{vs.}\quad H_1:\ \sum_{i=1}^k B_i\boldsymbol{\mu}_i\neq\boldsymbol{0}.}
#'
#' Li et al. (2025) proposed a random-integration-based U-statistic \eqn{T_n} (Eq. (5) in the paper),
#' constructed using the weight matrix \eqn{\boldsymbol{W}=\boldsymbol{\Omega}+\boldsymbol{A}\boldsymbol{A}^\top} with
#' \eqn{\boldsymbol{\Omega}=\mathrm{diag}(O_1^2,\ldots,O_p^2)}.
#' They showed that the standardized statistic \eqn{Z=T_n/\sqrt{\hat{\sigma}^2}} is approximated by \eqn{N(0,1)} under \eqn{H_0}.
#'
#' A recommended default choice of tuning parameters is of the form
#' \eqn{A_1=\cdots=A_p=\sqrt{5}\,p^{-3/8}} and \eqn{O_k=\sqrt{\epsilon\left(1+\frac{2k}{3p}\right)}}, \eqn{k=1,\ldots,p}.
#'
#' @references
#' \insertRef{li2025test}{HDNRA}
#'
#' @return A list of class \code{"NRtest"} containing the results of the hypothesis test.
#'
#'
#' @examples
#' \donttest{
#' library("HDNRA")
#' data("corneal")
#'
#' # corneal: 150 x p, split into 4 groups (n_i x p)
#' group1 <- as.matrix(corneal[1:43,  ])      # normal
#' group2 <- as.matrix(corneal[44:57, ])      # unilateral suspect
#' group3 <- as.matrix(corneal[58:78, ])      # suspect map
#' group4 <- as.matrix(corneal[79:150,])      # clinical keratoconus
#'
#' Y <- list(group2, group3, group4)
#' n <- c(nrow(group2), nrow(group3), nrow(group4))
#' p <- ncol(group2)
#'
#' # One linear combination (example): B = (4, -1.5, -2.5)
#' B <- c(4, -1.5, -2.5)
#'
#' # Paper-style tuning parameters (example with eps = 2)
#' A <- rep(sqrt(5) * p^(-3/8), p)
#' O <- sqrt(2) * (1 + 2*(1:p)/(3*p))
#'
#' LHNB2025.GLHTBF.NABT(Y, B, O, A, n, p)
#' }
#'
#'
#' @concept glht
#' @export
LHNB2025.GLHTBF.NABT <- function(Y, B, O, A, n, p) {

  if (!is.list(Y) || length(Y) < 2) stop("Y must be a list of group matrices.")
  k <- length(Y)

  if (length(B) != k) stop("Length of B must equal length(Y).")
  if (length(n) != k) stop("Length of n must equal length(Y).")
  if (length(O) != p || length(A) != p) stop("O and A must have length p.")

  if (any(sapply(Y, ncol) != p)) stop("Each Y[[i]] must have p columns (n_i x p).")

  nY <- sapply(Y, nrow)
  if (any(nY < 4)) stop("Each group must have n_i >= 4.")
  if (any(nY != n)) stop("n must match nrow(Y[[i]]) for all i.")

  stats <- random_integration_test_cpp(Y, B, O, A, n, p)
  Z <- stats[1]
  Tn <- stats[2]
  sigma2 <- stats[3]

  pvalue <- pnorm(q = Z, mean = 0, sd = 1, lower.tail = FALSE, log.p = FALSE)

  hname  <- paste("Random integration test", sep = "")
  hname1 <- paste("Normal approximation", sep = "")

  null.value <- "0"
  attr(null.value, "names") <- "Linear combination of mean vectors"
  alternative <- "two.sided"

  sample.size <- setNames(n, paste0("n", seq_along(n)))

  out <- list(
    statistic = c("Z[RI]" = round(Z, 4)),
    parameter = c("Tn" = round(Tn, 4), "sigma^2" = round(sigma2, 4)),
    p.value = pvalue,
    method = hname,
    estimation.method = hname1,
    data.name = deparse(substitute(Y)),
    null.value = null.value,
    sample.size = sample.size,
    sample.dimension = p,
    alternative = alternative
  )

  class(out) <- "NRtest"
  return(out)
}

