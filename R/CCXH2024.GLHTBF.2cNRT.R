#' @title
#' Normal-reference-test with two-cumulant (2-c) matched $\\chi^2$-approximation for GLHT problem proposed by Cao et al. (2024)
#'
#' @description
#' Implements the scale-invariant test of Cao et al. (2024) for high-dimensional
#' linear hypotheses of \eqn{k}-sample mean vectors under heteroscedastic
#' covariance structures.
#'
#' @param Y A list of \eqn{k} data matrices. The \eqn{i}th element represents the data matrix
#'   (\eqn{n_i \times p}) from the \eqn{i}th population with each row representing a \eqn{p}-dimensional observation.
#' @param B A vector of \eqn{k} known scalars \eqn{(B_1,\ldots,B_k)} specifying the linear combination of mean vectors.
#' @param n A vector of \eqn{k} sample sizes. The \eqn{i}th element represents the sample size of group \eqn{i}, \eqn{n_i}.
#' @param p The dimension of data.
#' @param alpha Significance level used to report the critical value (default 0.05). P-value does not depend on alpha.
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
#' Cao et al. (2024) proposed the following scale-invariant test statistic:
#' \deqn{T = p^{-1}\boldsymbol{Y}^{\top}\boldsymbol{D}_\sigma^{-1}\boldsymbol{Y},\quad
#' \boldsymbol{Y}=\sqrt{n}\sum_{i=1}^k B_i\bar{\boldsymbol{Y}}_i,\quad n=\sum_{i=1}^k n_i,}
#' where \eqn{\bar{\boldsymbol{Y}}_i} is the sample mean vector of group \eqn{i} and \eqn{\boldsymbol{D}_\sigma} is the diagonal matrix formed from a pooled covariance estimator.
#' They showed that under the null hypothesis, \eqn{T} can be approximated by a Welch--Satterthwaite chi-square reference law \eqn{\chi^2_{\nu^*}/\nu^*},
#' where \eqn{\nu^*} is an adjusted degrees-of-freedom parameter.
#'
#' @references
#' \insertRef{cao2024scale}{HDNRA}
#'
#' @return A list of class \code{"NRtest"} containing the results of the hypothesis test.
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
#' Y <- list(group1, group2, group3, group4)
#' n <- c(nrow(group1), nrow(group2), nrow(group3), nrow(group4))
#' p <- ncol(group1)
#'
#' # Example linear combination (single contrast)
#' B <- c(-2, 1, 2, -1)
#'
#' CCXH2024.GLHTBF.2cNRT(Y, B, n, p, alpha = 0.05)
#' }
#'
#' @importFrom stats pchisq qchisq
#' @concept nraglht
#' @export
CCXH2024.GLHTBF.2cNRT <- function(Y, B, n, p, alpha = 0.05) {

  if (!is.list(Y) || length(Y) < 2) stop("Y must be a list of group matrices.")
  k <- length(Y)

  if (length(B) != k) stop("Length of B must equal length(Y).")
  if (length(n) != k) stop("Length of n must equal length(Y).")
  if (any(sapply(Y, ncol) != p)) stop("Each Y[[i]] must have p columns (n_i x p).")
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) || alpha <= 0 || alpha >= 1) {
    stop("alpha must be a number between 0 and 1.")
  }

  nY <- sapply(Y, nrow)
  if (any(nY < 4)) stop("Each group must have n_i >= 4.")
  if (any(nY != n)) stop("n must match nrow(Y[[i]]) for all i.")

  # Call the C++ implementation:
  # out = [T, nu_star, nu_hat, trR2_hat, c_old, c_new, c_star]
  stats <- cao2024_scale_invariant_cpp(Y, B, n, p)

  T       <- stats[1]
  nu_star <- stats[2]
  nu_hat  <- stats[3]
  trR2hat <- stats[4]
  c_old   <- stats[5]
  c_new   <- stats[6]
  c_star  <- stats[7]

  # Upper-tail p-value under the chi-square approximation.
  pvalue <- pchisq(q = nu_star * T, df = nu_star, ncp = 0,
                   lower.tail = FALSE, log.p = FALSE)

  # Critical value reported as extra information.
  crit <- qchisq(p = 1 - alpha, df = nu_star) / nu_star

  hname  <- paste("Cao et al. (2024)'s scale-invariant test", sep = "")
  hname1 <- paste("Welch-Satterthwaite chi-square approximation", sep = "")

  null.value <- "0"
  attr(null.value, "names") <- "Linear combination of mean vectors"
  alternative <- "two.sided"

  sample.size <- setNames(n, paste0("n", seq_along(n)))

  out <- list(
    statistic = c("T[SI]" = round(T, 4)),
    parameter = c("nu_star" = round(nu_star, 4),
                  "nu_hat"  = round(nu_hat, 4),
                  "trR2_hat" = round(trR2hat, 4),
                  "c_old"   = round(c_old, 4),
                  "c_new"   = round(c_new, 4),
                  "c_star"  = round(c_star, 4),
                  "crit"    = round(crit, 4)),
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

