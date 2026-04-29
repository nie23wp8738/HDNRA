#' @title
#' F-approximation-based F-type test for GLHT problem under heteroscedasticity
#' @description
#' An F-type normal reference test for the high-dimensional general linear hypothesis testing (GLHT)
#' problem under heteroscedasticity. The null distribution is approximated by an F distribution
#' using Welch--Satterthwaite (W--S) chi-square approximations.
#'
#' @usage WZ2026.GLHTBF.2cNRT(Y, G, n, p)
#' @param Y A list of \eqn{k} data matrices. The \eqn{i}th element represents the data matrix
#'   (\eqn{n_i \times p}) from the \eqn{i}th population with each row representing a \eqn{p}-dimensional observation.
#' @param G A known full-rank coefficient matrix (\eqn{q \times k}) with \eqn{\operatorname{rank}(\boldsymbol{G}) < k}.
#' @param n A vector of \eqn{k} sample sizes. The \eqn{i}th element represents the sample size of group \eqn{i}, \eqn{n_i}.
#' @param p The dimension of data.
#'
#' @details
#' The test statistic is of F-type form
#' \deqn{F_{n,p} = \frac{\|\boldsymbol{C\hat\mu}\|^2}{\operatorname{tr}(\widehat{\Omega}_n)}.}
#' The degrees of freedom are estimated by matching the first two cumulants via W--S approximation,
#' resulting in an \eqn{F_{\hat d_1, \hat d_2}} reference distribution.
#'
#' @references
#' Wang, P. and Zhu, T. (preprint). An F-type Test for Heteroscedastic General Linear Hypothesis Testing
#' Problem for High Dimensional Data: a Normal Reference Approach.
#'
#' @return A list of class \code{"NRtest"} containing the results of the hypothesis test.
#'
#'
#'
#' @examples
#' library("HDNRA")
#' data("corneal")
#' dim(corneal)
#' group1 <- as.matrix(corneal[1:43, ]) ## normal group
#' group2 <- as.matrix(corneal[44:57, ]) ## unilateral suspect group
#' group3 <- as.matrix(corneal[58:78, ]) ## suspect map group
#' group4 <- as.matrix(corneal[79:150, ]) ## clinical keratoconus group
#' p <- dim(corneal)[2]
#' Y <- list()
#' k <- 4
#' Y[[1]] <- group1
#' Y[[2]] <- group2
#' Y[[3]] <- group3
#' Y[[4]] <- group4
#' n <- c(nrow(Y[[1]]),nrow(Y[[2]]),nrow(Y[[3]]),nrow(Y[[4]]))
#' G <- cbind(diag(k-1),rep(-1,k-1))
#' WZ2026.GLHTBF.2cNRT(Y,G,n,p)
#'
#'
#' @concept glht
#' @export
WZ2026.GLHTBF.2cNRT <- function(Y, G, n, p) {
  if (!is.list(Y) || length(Y) < 2) stop("Y must be a list of group matrices.")
  if (length(n) != length(Y)) stop("Length of n must equal length(Y).")
  if (any(sapply(Y, ncol) != p)) stop("Each Y[[i]] must have p columns (n_i x p).")

  stats <- wz2026_glhtbf_2cnrt_cpp(Y, G, n, p)
  stat <- stats[1]
  d1 <- stats[2]
  d2 <- stats[3]
  bias <- stats[4]

  # Upper-tail p-value for F_{d1,d2}
  pvalue <- pf(q = stat, df1 = d1, df2 = d2, lower.tail = FALSE, log.p = FALSE)

  hname  <- paste("F-type normal reference test", sep = "")
  hname1 <- paste("F-approximation (W-S chi^2 matching)", sep = "")

  null.value <- "true"
  attr(null.value, "names") <- "The general linear hypothesis"
  alternative <- "two.sided"

  out <- list(
    statistic = c("F[WZ]" = round(stat, 4)),
    parameter = c("df1" = round(d1, 4), "df2" = round(d2, 4), "bias" = round(bias, 4)),
    p.value = pvalue,
    method = hname,
    estimation.method = hname1,
    data.name = deparse(substitute(Y)),
    null.value = null.value,
    sample.size = c(n = n),
    sample.dimension = p,
    alternative = alternative
  )

  class(out) <- "NRtest"
  return(out)
}
