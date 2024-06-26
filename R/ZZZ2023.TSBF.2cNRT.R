#' @title
#' Normal-reference-test with two-cumulant (2-c) matched χ^2-approximation for two-sample BF problem proposed by Zhang et al. (2023)
#' @description
#' Zhang et al. (2023)'s test for testing equality of two-sample high-dimensional mean vectors without assuming that two covariance matrices are the same.

#' @usage ZZZ2023.TSBF.2cNRT(y1, y2, cutoff)
#' @param y1 The data matrix (\eqn{n_1 \times p}) from the first population. Each row represents a \eqn{p}-dimensional observation.
#' @param y2 The data matrix (\eqn{n_2 \times p}) from the second population. Each row represents a \eqn{p}-dimensional observation.
#' @param cutoff An empirical criterion for applying the adjustment coefficient
#' @details
#' Suppose we have two independent high-dimensional samples:
#' \deqn{
#' \boldsymbol{y}_{i1},\ldots,\boldsymbol{y}_{in_i}, \;\operatorname{are \; i.i.d. \; with}\; \operatorname{E}(\boldsymbol{y}_{i1})=\boldsymbol{\mu}_i,\; \operatorname{Cov}(\boldsymbol{y}_{i1})=\boldsymbol{\Sigma}_i,i=1,2.
#' }
#' The primary object is to test
#' \deqn{H_{0}: \boldsymbol{\mu}_1 = \boldsymbol{\mu}_2\; \operatorname{versus}\; H_{1}: \boldsymbol{\mu}_1 \neq \boldsymbol{\mu}_2.}
#' Zhang et al.(2023) proposed the following test statistic:
#' \deqn{T_{ZZZ}=\frac{n_1 n_2}{np}(\bar{\boldsymbol{y}}_1-\bar{\boldsymbol{y}}_2)^{\top} \hat{\boldsymbol{D}}_n^{-1}(\bar{\boldsymbol{y}}_1-\bar{\boldsymbol{y}}_2),}
#' where \eqn{\bar{\boldsymbol{y}}_{i},i=1,2} are the sample mean vectors, and \eqn{\hat{\boldsymbol{D}}_n=\operatorname{diag}(\hat{\boldsymbol{\Sigma}}_1/n+\hat{\boldsymbol{\Sigma}}_2/n)} with \eqn{n=n_1+n_2}.

#' They showed that under the null hypothesis, \eqn{T_{ZZZ}} and a chi-squared-type mixture have the same limiting distribution.
#'
#' @references
#' \insertRef{zhang2023two}{HDNRA}
#'
#' @return A list of class \code{"NRtest"} containing the results of the hypothesis test. See the help file for \code{\link{NRtest.object}} for details.


#' @examples
#' library("HDNRA")
#' data("COVID19")
#' dim(COVID19)
#' group1 <- as.matrix(COVID19[c(2:19, 82:87), ]) ## healthy group
#' group2 <- as.matrix(COVID19[-c(1:19, 82:87), ]) ## COVID-19 patients
#' ZZZ2023.TSBF.2cNRT(group1,group2,cutoff=1.2)

#'
#' @concept nrats
#' @export
ZZZ2023.TSBF.2cNRT <- function(y1, y2, cutoff) {
  if (ncol(y1) != ncol(y2)) {
    stop("y1 and y2 must have the same dimension!")
  }

  # Calculate test statistics using the provided C++ function
  stats <- tsbf_zzz2023_cpp(y1, y2)
  stat <- stats[1]
  dhat <- stats[2]
  cpn <- stats[3]

  ### SI test
  if (cpn <= cutoff) {
    d <- dhat / cpn
    pvalue <- 1 - pchisq(d * stat, d)
  } else {
    d <- dhat
    pvalue <- 1 - pchisq(d * stat, d)
  }

  # Prepare the result as an NRtest object
  hname <- paste("Zhang et al. (2023)'s test", sep = "")
  hname1 <- paste("2-c matched chi^2-approximation", sep = "")

  null.value  <- "0"
  attr(null.value, "names") <- "Difference between two mean vectors"
  alternative <- "two.sided"

  out <- list(
    statistic = c("T[ZZZ]" = round(stat,4)),
    parameter = c("df" = round(d,4), "cpn" = round(cpn,4)), # Include additional parameters as needed
    p.value = pvalue,
    method = hname,
    estimation.method = hname1,
    data.name = paste(deparse(substitute(y1)), " and ", deparse(substitute(y2)), sep = ""),
    null.value = null.value,
    sample.size = c(n1 = nrow(y1), n2 = nrow(y2)),
    sample.dimension = ncol(y1),
    alternative = alternative
  )

  class(out) <- "NRtest"
  return(out)
}

