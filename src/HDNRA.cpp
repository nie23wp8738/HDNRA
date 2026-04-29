// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// Helper function to compute inverse using Cholesky decomposition
arma::mat cholesky_inverse(const arma::mat &X) {
  arma::mat L = arma::chol(X, "lower");  // Performs Cholesky decomposition, returns lower triangular matrix
  arma::mat Linv = arma::inv(L);  // Computes the inverse of the lower triangular matrix L
  return Linv.t() * Linv;  // Returns the inverse of the input matrix X
}

// Test proposed by Bai and Saranadasa (1996)
// [[Rcpp::export]]
double bs1996_ts_nabt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int p = y1.n_cols;

  arma::rowvec mu1 = arma::mean(y1, 0);
  arma::mat z1 = y1.t() - arma::repmat(mu1.t(), 1, n1);
  arma::rowvec mu2 = arma::mean(y2, 0);
  arma::mat z2 = y2.t() - arma::repmat(mu2.t(), 1, n2);

  int n = n1 + n2 - 2;
  double invtau = double(n1 * n2) / (n1 + n2);
  double stat = arma::as_scalar((mu1 - mu2) * (mu1 - mu2).t());
  arma::mat z = arma::join_horiz(z1, z2);
  arma::mat S;

  if (p <= n) {
    S = (z * z.t()) / n; // tr(Sigmahat)
  } else {
    S = (z.t() * z) / n;
  }

  double trSn = arma::trace(S); // tr(Sigmahat)
  double trSn2 = arma::accu(S % S); // tr(Sigmahat^2)
  double Bn2 = double(n * n) / ((n + 2) * (n - 1)) * (trSn2 - trSn * trSn / n); // unbiased estimator for B
  double statn = (invtau * stat - trSn) / sqrt(2 * double(n + 1) / n * Bn2);

  return statn;
}

// Test proposed by Chen and Qin (2010)
// [[Rcpp::export]]
arma::vec cq2010_tsbf_nabt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;

  arma::mat y1xy1t = y1 * y1.t();
  double sumx1tx1inej = arma::accu(y1xy1t) - arma::trace(y1xy1t);
  arma::mat y2xy2t = y2 * y2.t();
  double sumx2tx2inej = arma::accu(y2xy2t) - arma::trace(y2xy2t);
  arma::mat y1xy2t = y1 * y2.t();
  double sumx1tx2 = arma::accu(y1xy2t);
  double Tn = sumx1tx1inej / (n1 * (n1 - 1)) + sumx2tx2inej / (n2 * (n2 - 1)) - 2 * sumx1tx2 / (n1 * n2);

  double tempsum = 0.0;
  arma::rowvec y1sum = arma::sum(y1, 0); // add columns
  int n1_1 = n1 - 1;
  int n2_1 = n2 - 1;
  int n1_2 = n1 - 2;
  int n2_2 = n2 - 2;

  // Parallelize the first loop using OpenMP
#pragma omp parallel for reduction(+:tempsum)
  for (int j = 0; j < n1; j++) {
    arma::rowvec y1_j = y1.row(j);
    for (int k = j + 1; k < n1; k++) {
      arma::rowvec y1_k = y1.row(k);
      arma::rowvec meanxiexjk = (y1sum - y1_j - y1_k) / n1_2;
      tempsum += arma::as_scalar(y1_k * (y1_j - meanxiexjk).t() * y1_j * (y1_k - meanxiexjk).t());
    }
  }

  double trsigma12hat = (tempsum + tempsum) / (n1 * n1_1);
  tempsum = 0.0;
  arma::rowvec y2sum = arma::sum(y2, 0); // add columns

  // Parallelize the second loop using OpenMP
#pragma omp parallel for reduction(+:tempsum)
  for (int j = 0; j < n2; j++) {
    arma::rowvec y2_j = y2.row(j);
    for (int k = j + 1; k < n2; k++) {
      arma::rowvec y2_k = y2.row(k);
      arma::rowvec meanxiexjk = (y2sum - y2_j - y2_k) / n2_2;
      tempsum += arma::as_scalar(y2_k * (y2_j - meanxiexjk).t() * y2_j * (y2_k - meanxiexjk).t());
    }
  }

  double trsigma22hat = (tempsum + tempsum) / (n2 * n2_1);
  tempsum = 0.0;

  // Parallelize the third loop using OpenMP
#pragma omp parallel for reduction(+:tempsum)
  for (int l = 0; l < n1; l++) {
    arma::rowvec y1_l = y1.row(l);
    arma::rowvec meanx1exl = (y1sum - y1_l) / n1_1;
    for (int k = 0; k < n2; k++) {
      arma::rowvec y2_k = y2.row(k);
      arma::rowvec meanx2exk = (y2sum - y2_k) / n2_1;
      tempsum += arma::as_scalar(y2_k * (y1_l - meanx1exl).t() * y1_l * (y2_k - meanx2exk).t());
    }
  }

  double trsigma1sigma2hat = tempsum / (n1 * n2);
  double sigman12hat = 2.0 / (n1 * n1_1) * trsigma12hat + 2.0 / (n2 * n2_1) * trsigma22hat + 4.0 / (n1 * n2) * trsigma1sigma2hat;
  double stat = Tn / sqrt(sigman12hat); // Qn

  arma::vec stats(2);
  stats(0) = stat;
  stats(1) = Tn;
  return stats;
}

// Test proposed by Zhang et al. (2020)
// [[Rcpp::export]]
arma::vec zgzc2020_ts_2cnrt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int p = y1.n_cols;

  arma::rowvec mu1 = arma::mean(y1, 0);
  arma::mat z1 = y1.t() - arma::repmat(mu1.t(), 1, n1);
  arma::rowvec mu2 = arma::mean(y2, 0);
  arma::mat z2 = y2.t() - arma::repmat(mu2.t(), 1, n2);
  int n = n1 + n2;
  double stat = double(n1 * n2) / n * arma::as_scalar((mu1 - mu2) * (mu1 - mu2).t());

  arma::mat z = join_horiz(z1, z2);
  arma::mat S;
  if (p <= n) {
    S = (z * z.t()) / (n - 2); // tr(Sigmahat)
  } else {
    S = (z.t() * z) / (n - 2);
  }

  double B = arma::accu(S % S); // tr(Sigmahat^2)
  double A = arma::trace(S); // tr(Sigmahat)
  double uA0 = A;
  double uA = double((n - 1) * (n - 2)) / (n * (n - 3)) * (A * A - 2 * B / (n - 1)); // unbiased estimator for A^2
  double uB = double((n - 2) * (n - 2)) / (n * (n - 3)) * (B - A * A / (n - 2)); // unbiased estimator for B
  double beta = uB / uA0;
  double df = uA / uB;
  double statn = stat / beta; // normalizing
  double statstd = (stat - beta * df) / sqrt(2 * beta * beta * df);

  arma::vec stats(5);
  stats(0) = stat;
  stats(1) = statn;
  stats(2) = beta;
  stats(3) = df;
  stats(4) = statstd;
  return stats;
}

// Test proposed by Zhang et al. (2021).
// [[Rcpp::export]]
arma::vec zzgz2021_tsbf_2cnrt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int p = y1.n_cols;

  arma::rowvec mu1 = arma::mean(y1, 0);
  arma::mat z1 = y1.t() - arma::repmat(mu1.t(), 1, n1);
  arma::rowvec mu2 = arma::mean(y2, 0);
  arma::mat z2 = y2.t() - arma::repmat(mu2.t(), 1, n2);
  int n = n1 + n2;
  double stat = double(n1 * n2) / n * arma::as_scalar((mu1 - mu2) * (mu1 - mu2).t());

  arma::mat S1, S2;
  double B1, B2, B12;
  if (p < n) {
    S1 = (z1 * z1.t()) / (n1 - 1);
    B1 = arma::accu(S1 % S1);
    S2 = (z2 * z2.t()) / (n2 - 1);
    B2 = arma::accu(S2 % S2);
    B12 = arma::trace(z1 * z1.t() * z2 * z2.t()) / ((n1 - 1) * (n2 - 1));
  } else {
    S1 = (z1.t() * z1) / (n1 - 1);
    B1 = arma::accu(S1 % S1);
    S2 = (z2.t() * z2) / (n2 - 1);
    B2 = arma::accu(S2 % S2);
    B12 = arma::trace(z1.t() * z2 * z2.t() * z1) / ((n1 - 1) * (n2 - 1));
  }

  double A1 = arma::trace(S1);
  double A2 = arma::trace(S2);
  double uA0 = (n2 * A1 + n1 * A2) / n; // tr(Sigma)
  double A12 = A1 * A2;
  double uA1 = double(n1 * (n1 - 1)) / ((n1 + 1) * (n1 - 2)) * (A1 * A1 - 2 * B1 / n1); // unbiased estimator for A1^2
  double uB1 = double((n1 - 1) * (n1 - 1)) / ((n1 + 1) * (n1 - 2)) * (B1 - A1 * A1 / (n1 - 1)); // unbiased estimator for B1
  double uA2 = double(n2 * (n2 - 1)) / ((n2 + 1) * (n2 - 2)) * (A2 * A2 - 2 * B2 / n2); // unbiased estimator for A2^2
  double uB2 = double((n2 - 1) * (n2 - 1)) / ((n2 + 1) * (n2 - 2)) * (B2 - A2 * A2 / (n2 - 1)); // unbiased estimator for B2

  double uA = (double(n2 * n2) / (n * n) * uA1 + 2 * double(n1 * n2) / (n * n) * A12 + double(n1 * n1) / (n * n) * uA2); // tr(Sigma)^2
  double uB = (double(n2 * n2) / (n * n) * uB1 + 2 * double(n1 * n2) / (n * n) * B12 + double(n1 * n1) / (n * n) * uB2); // tr(Sigma^2)

  double beta = uB / uA0;
  double df = uA / uB;
  double statn = stat / beta; // normalizing
  double statstd = (stat - beta * df) / sqrt(2 * beta * beta * df);

  arma::vec stats(5);
  stats(0) = stat;
  stats(1) = statn;
  stats(2) = beta;
  stats(3) = df;
  stats(4) = statstd;
  return stats;
}

// Test proposed by Zhang and Zhu (2022)
// [[Rcpp::export]]
arma::vec zz2022_ts_3cnrt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int p = y1.n_cols;

  arma::rowvec mu1 = arma::mean(y1, 0);
  arma::mat z1 = y1.t() - arma::repmat(mu1.t(), 1, n1);
  arma::rowvec mu2 = arma::mean(y2, 0);
  arma::mat z2 = y2.t() - arma::repmat(mu2.t(), 1, n2);
  int n = n1 + n2;
  double invtau = double(n1 * n2) / n;
  double stat = arma::as_scalar((mu1 - mu2) * (mu1 - mu2).t());

  arma::mat z = join_horiz(z1, z2);
  arma::mat S;
  if (p <= n) {
    S = (z * z.t()) / (n - 2); // tr(Sigmahat)
  } else {
    S = (z.t() * z) / (n - 2);
  }

  double trSn = arma::trace(S); // tr(Sigmahat)
  double trSn2 = arma::accu(S % S); // tr(Sigmahat^2)
  double trSn3 = arma::trace(S * S * S); // tr(Sigmahat^3)
  double htrSn2 = pow((n - 2), 2) * (trSn2 - trSn * trSn / (n - 2)) / (n - 3) / n;
  double htrSn3 = pow((n - 2), 4) * (trSn3 - 3 * trSn * trSn2 / (n - 2) + 2 * pow(trSn, 3) / pow((n - 2), 2)) / (n * n - n - 6) / (n * (n - 4));

  double statn = invtau * stat - trSn;
  double beta0 = -(n - 1) * htrSn2 * htrSn2 / htrSn3 / (n - 3);
  double beta1 = (n - 3) * htrSn3 / htrSn2 / (n - 2);
  double d = (n - 1) * (n - 2) * pow(htrSn2, 3) / pow(htrSn3, 2) / pow((n - 3), 2);
  double statstd = statn / sqrt(2 * (n - 1) * htrSn2 / (n - 2));

  arma::vec stats(5);
  stats(0) = statn;
  stats(1) = beta0;
  stats(2) = beta1;
  stats(3) = d;
  stats(4) = statstd;
  return stats;
}

// Test proposed by Zhang and Zhu (2022)
// [[Rcpp::export]]
arma::vec zz2022_tsbf_3cnrt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int p = y1.n_cols;

  arma::rowvec mu1 = arma::mean(y1, 0);
  arma::mat z1 = y1.t() - arma::repmat(mu1.t(), 1, n1);
  arma::rowvec mu2 = arma::mean(y2, 0);
  arma::mat z2 = y2.t() - arma::repmat(mu2.t(), 1, n2);
  int n = n1 + n2;
  arma::mat S1, S2;
  double B1, B2, B12, D12, D21;

  if (p < n) {
    S1 = (z1 * z1.t()) / (n1 - 1);
    B1 = arma::accu(S1 % S1);
    S2 = (z2 * z2.t()) / (n2 - 1);
    B2 = arma::accu(S2 % S2);
    B12 = arma::trace(z1 * z1.t() * z2 * z2.t()) / ((n1 - 1) * (n2 - 1));
    D12 = arma::trace(z1 * z1.t() * z1 * z1.t() * z2 * z2.t()) / ((n1 - 1) * (n1 - 1) * (n2 - 1));
    D21 = arma::trace(z1 * z1.t() * z2 * z2.t() * z2 * z2.t()) / ((n1 - 1) * (n2 - 1) * (n2 - 1));
  } else {
    S1 = (z1.t() * z1) / (n1 - 1);
    B1 = arma::accu(S1 % S1);
    S2 = (z2.t() * z2) / (n2 - 1);
    B2 = arma::accu(S2 % S2);
    B12 = arma::trace(z1.t() * z2 * z2.t() * z1) / ((n1 - 1) * (n2 - 1));
    D12 = arma::trace(z1.t() * z1 * z1.t() * z2 * z2.t() * z1) / ((n1 - 1) * (n1 - 1) * (n2 - 1));
    D21 = arma::trace(z1.t() * z2 * z2.t() * z2 * z2.t() * z1) / ((n1 - 1) * (n2 - 1) * (n2 - 1));
  }

  double A1 = arma::trace(S1);
  double A2 = arma::trace(S2);
  double stat = arma::as_scalar((mu1 - mu2) * (mu1 - mu2).t()) - (A1 / n1 + A2 / n2);

  double uB1 = double((n1 - 1) * (n1 - 1)) / ((n1 + 1) * (n1 - 2)) * (B1 - A1 * A1 / (n1 - 1));
  double uB2 = double((n2 - 1) * (n2 - 1)) / ((n2 + 1) * (n2 - 2)) * (B2 - A2 * A2 / (n2 - 1));

  double K2 = 2 * (uB1 / (n1 * (n1 - 1)) + 2 * B12 / (n1 * n2) + uB2 / (n2 * (n2 - 1)));

  double C1 = arma::trace(S1 * S1 * S1);
  double C2 = arma::trace(S2 * S2 * S2);

  double c1 = pow((n1 - 1), 4) / (n1 * n1 + n1 - 6) / (n1 * n1 - 2 * n1 - 3);
  double c2 = pow((n2 - 1), 4) / (n2 * n2 + n2 - 6) / (n2 * n2 - 2 * n2 - 3);

  double uC1 = c1 * (C1 - 3 * A1 * B1 / (n1 - 1) + 2 * pow(A1, 3) / ((n1 - 1) * (n1 - 1)));
  double uC2 = c2 * (C2 - 3 * A2 * B2 / (n2 - 1) + 2 * pow(A2, 3) / ((n2 - 1) * (n2 - 1)));

  double uD12 = (n1 - 1) * ((n1 - 1) * D12 - B12 * A1) / (n1 - 2) / (n1 + 1);
  double uD21 = (n2 - 1) * ((n2 - 1) * D21 - B12 * A2) / (n2 - 2) / (n2 + 1);

  double K3 = 8 * ((n1 - 2) * uC1 / pow(n1 * (n1 - 1), 2) + 3 * uD12 / (n1 * n1 * n2) + 3 * uD21 / (n1 * n2 * n2) + (n2 - 2) * uC2 / pow(n2 * (n2 - 1), 2));

  double beta0 = -2 * K2 * K2 / K3;
  double beta1 = K3 / (4 * K2);
  double d = 8 * pow(K2, 3) / (K3 * K3);

  double statstd = stat / sqrt(K2);

  arma::vec stats(5);
  stats(0) = stat;
  stats(1) = beta0;
  stats(2) = beta1;
  stats(3) = d;
  stats(4) = statstd;
  return stats;
}

// Test proposed by Zhu et al.(2023)
// [[Rcpp::export]]
arma::vec zwz2023_tsbf_2cnrt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int p = y1.n_cols;

  arma::rowvec mu1 = arma::mean(y1, 0);
  arma::mat z1 = y1.t() - arma::repmat(mu1.t(), 1, n1);
  arma::rowvec mu2 = arma::mean(y2, 0);
  arma::mat z2 = y2.t() - arma::repmat(mu2.t(), 1, n2);
  int n = n1 + n2;
  arma::mat S1, S2;
  double B1, B2, B12;

  if (p < n) {
    S1 = (z1 * z1.t()) / (n1 - 1);
    B1 = arma::accu(S1 % S1);
    S2 = (z2 * z2.t()) / (n2 - 1);
    B2 = arma::accu(S2 % S2);
    B12 = arma::trace(z1 * z1.t() * z2 * z2.t()) / ((n1 - 1) * (n2 - 1));
  } else {
    S1 = (z1.t() * z1) / (n1 - 1);
    B1 = arma::accu(S1 % S1);
    S2 = (z2.t() * z2) / (n2 - 1);
    B2 = arma::accu(S2 % S2);
    B12 = arma::trace(z1.t() * z2 * z2.t() * z1) / ((n1 - 1) * (n2 - 1));
  }

  double A1 = arma::trace(S1);
  double A2 = arma::trace(S2);
  double stat = arma::as_scalar((mu1 - mu2) * (mu1 - mu2).t()) / (A1 / n1 + A2 / n2);

  double A12 = A1 * A2;
  double uA1 = double(n1 * (n1 - 1)) / ((n1 + 1) * (n1 - 2)) * (A1 * A1 - 2 * B1 / n1); // unbiased estimator for A1^2
  double uB1 = double((n1 - 1) * (n1 - 1)) / ((n1 + 1) * (n1 - 2)) * (B1 - A1 * A1 / (n1 - 1)); // unbiased estimator for B1
  double uA2 = double(n2 * (n2 - 1)) / ((n2 + 1) * (n2 - 2)) * (A2 * A2 - 2 * B2 / n2); // unbiased estimator for A2^2
  double uB2 = double((n2 - 1) * (n2 - 1)) / ((n2 + 1) * (n2 - 2)) * (B2 - A2 * A2 / (n2 - 1)); // unbiased estimator for B2

  double uA = uA1 / (n1 * n1) + 2 * A12 / (n1 * n2) + uA2 / (n2 * n2);
  double uB = uB1 / (n1 * n1) + 2 * B12 / (n1 * n2) + uB2 / (n2 * n2);
  double d1 = uA / uB;
  double uC = uB1 / (n1 * n1 * (n1 - 1)) + uB2 / (n2 * n2 * (n2 - 1));
  double d2 = uA / uC;

  arma::vec stats(3);
  stats(0) = stat;
  stats(1) = d1;
  stats(2) = d2;
  return stats;
}

// Two-sample scale-invariant tests
// Test proposed by Srivastava and Du (2008)
// [[Rcpp::export]]
arma::vec sd2008_ts_nabt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int n = n1 + n2 - 2;
  int p = y1.n_cols;

  arma::rowvec bary1 = arma::mean(y1, 0);
  arma::rowvec bary2 = arma::mean(y2, 0);

  arma::mat x1 = y1.each_row() - bary1;
  arma::mat x2 = y2.each_row() - bary2;

  arma::vec diagSvec1 = arma::var(x1.t(), 0, 1); // variance of x1
  arma::vec diagSvec2 = arma::var(x2.t(), 0, 1); // variance of x2

  arma::mat x = join_horiz(x1.t(), x2.t());

  arma::vec sqrtdiagSvec = arma::sqrt((diagSvec1 * (n1 - 1) + diagSvec2 * (n2 - 1)) / n); // D^{-1/2}
  sqrtdiagSvec.elem(find(sqrtdiagSvec < pow(10, -10))).fill(pow(10, -10));
  arma::vec meandiff = (bary1.t() - bary2.t());
  meandiff.each_col() /= sqrtdiagSvec; // D^{-1/2}(bary1-bary2)

  double Tnp = double(n1 * n2) / (n1 + n2) * arma::dot(meandiff, meandiff) / p;
  x.each_col() /= sqrtdiagSvec;
  arma::mat R;
  double trR2;

  if (n1 < p || n2 < p) {
    R = (x.t() * x) / n; // tr(Sigmahat)
    trR2 = arma::accu(R % R);
  } else {
    R = (x * x.t()) / n;
    trR2 = arma::accu(R % R);
  }

  double cpn = 1 + trR2 / pow(sqrt(p), 3);
  double TSD = (p * Tnp - n * p / (n - 2)) / sqrt(2 * (trR2 - p * p / n) * cpn);

  arma::vec values(2);
  values(0) = TSD;
  values(1) = cpn;
  return values;
}

// Test proposed by Srivastava et al.(2013)
// [[Rcpp::export]]
arma::vec skk2013_tsbf_nabt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int n = n1 + n2 - 2;
  int p = y1.n_cols;

  arma::rowvec bary1 = arma::mean(y1, 0);
  arma::rowvec bary2 = arma::mean(y2, 0);

  arma::mat x1 = y1.each_row() - bary1;
  arma::mat x2 = y2.each_row() - bary2;

  arma::vec diagSvec1 = arma::var(x1.t(), 0, 1); // variance of x1
  arma::vec diagSvec2 = arma::var(x2.t(), 0, 1); // variance of x2

  arma::vec sqrtdiagSvec = arma::sqrt((diagSvec1 * n2 + diagSvec2 * n1) / n); // D^{-1/2}
  sqrtdiagSvec.elem(find(sqrtdiagSvec < pow(10, -10))).fill(pow(10, -10));
  arma::vec meandiff = (bary1.t() - bary2.t());
  meandiff.each_col() /= sqrtdiagSvec; // D^{-1/2}(bary1-bary2)

  double Tnp = double(n1 * n2) / n * arma::dot(meandiff, meandiff) / p;
  arma::mat x1t = x1.t();
  arma::mat x2t = x2.t();
  arma::mat w1 = x1t.each_col() / sqrtdiagSvec;
  arma::mat w2 = x2t.each_col() / sqrtdiagSvec;
  arma::mat R1;
  arma::mat R2;
  double B1, B2, B12;

  if (n1 < p || n2 < p) {
    R1 = (w1.t() * w1) / (n1 - 1); // tr(Sigmahat)
    R2 = (w2.t() * w2) / (n2 - 1);
    B1 = arma::accu(R1 % R1);
    B2 = arma::accu(R2 % R2);
    B12 = arma::trace(w2.t() * w1 * w1.t() * w2) / ((n1 - 1) * (n2 - 1));
  } else {
    R1 = (w1 * w1.t()) / (n1 - 1); // tr(Sigmahat)
    R2 = (w2 * w2.t()) / (n2 - 1);
    B1 = arma::accu(R1 % R1);
    B2 = arma::accu(R2 % R2);
    B12 = arma::trace(w1 * w1.t() * w2 * w2.t()) / ((n1 - 1) * (n2 - 1));
  }

  arma::vec vecw1 = arma::vectorise(w1);
  arma::vec vecw2 = arma::vectorise(w2);
  double trinvDS1 = arma::dot(vecw1, vecw1) / (n1 - 1); // = arma::trace(invDs*S1)
  double trinvDS2 = arma::dot(vecw2, vecw2) / (n2 - 1); // = arma::trace(invDs*S2)

  double trR2 = (n2 * n2 * B1 + n1 * n1 * B2 + 2 * n1 * n2 * B12) / (n * n);
  double cpn = 1 + trR2 / pow(sqrt(p), 3);
  double sigma2 = ((n2 * n2 * trinvDS1 * trinvDS1) / (n1 - 1) + (n1 * n1 * trinvDS2 * trinvDS2) / (n2 - 1)) / (n * n);
  sigma2 = 2 * (trR2 - sigma2);

  double TSKK = (p * Tnp - p) / sqrt(sigma2 * cpn);

  arma::vec values(3);
  values(0) = TSKK;
  values(1) = sigma2;
  values(2) = cpn;
  return values;
}

// Test proposed by Zhang et al. (2020)
// [[Rcpp::export]]
arma::vec zzz2020_ts_2cnrt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int n = n1 + n2 - 2;
  int p = y1.n_cols;

  arma::rowvec bary1 = arma::mean(y1, 0);
  arma::rowvec bary2 = arma::mean(y2, 0);

  arma::mat x1 = y1.each_row() - bary1;
  arma::mat x2 = y2.each_row() - bary2;

  arma::vec diagSvec1 = arma::var(x1.t(), 0, 1); // variance of x1
  arma::vec diagSvec2 = arma::var(x2.t(), 0, 1); // variance of x2

  arma::mat x = join_horiz(x1.t(), x2.t());

  arma::vec sqrtdiagSvec = arma::sqrt((diagSvec1 * (n1 - 1) + diagSvec2 * (n2 - 1)) / n); // D^{-1/2}
  sqrtdiagSvec.elem(find(sqrtdiagSvec < pow(10, -10))).fill(pow(10, -10));
  arma::vec meandiff = (bary1.t() - bary2.t());
  meandiff.each_col() /= sqrtdiagSvec; // D^{-1/2}(bary1-bary2)

  double Tnp = double(n1 * n2) / (n1 + n2) * arma::dot(meandiff, meandiff) / p;
  x.each_col() /= sqrtdiagSvec;
  arma::mat R;
  double trR2;

  if (n1 < p || n2 < p) {
    R = (x.t() * x) / n; // tr(Sigmahat)
    trR2 = arma::accu(R % R);
  } else {
    R = (x * x.t()) / n;
    trR2 = arma::accu(R % R);
  }

  double trR2hat = pow(n, 2) * (trR2 - pow(trace(R), 2) / n) / (n1 + n2) / (n - 1);
  double dhat = p * p / trR2hat;

  arma::vec values(2);
  values(0) = Tnp;
  values(1) = dhat;
  return values;
}

// Test proposed by Zhang et al. (2023)
// [[Rcpp::export]]
arma::vec zzz2023_tsbf_2cnrt_cpp(const arma::mat &y1, const arma::mat &y2) {
  int n1 = y1.n_rows;
  int n2 = y2.n_rows;
  int n = n1 + n2 - 2;
  int p = y1.n_cols;

  arma::rowvec bary1 = arma::mean(y1, 0);
  arma::rowvec bary2 = arma::mean(y2, 0);

  arma::mat x1 = y1.each_row() - bary1;
  arma::mat x2 = y2.each_row() - bary2;

  arma::vec diagSvec1 = arma::var(x1.t(), 0, 1); // variance of x1
  arma::vec diagSvec2 = arma::var(x2.t(), 0, 1); // variance of x2

  arma::vec sqrtdiagSvec = arma::sqrt((diagSvec1 * n2 + diagSvec2 * n1) / n); // D^{-1/2}
  sqrtdiagSvec.elem(find(sqrtdiagSvec < pow(10, -10))).fill(pow(10, -10));
  arma::vec meandiff = (bary1.t() - bary2.t());
  meandiff.each_col() /= sqrtdiagSvec; // D^{-1/2}(bary1-bary2)

  double Tnp = double(n1 * n2) / n * arma::dot(meandiff, meandiff) / p;
  arma::mat x1t = x1.t();
  arma::mat x2t = x2.t();
  arma::mat w1 = x1t.each_col() / sqrtdiagSvec;
  arma::mat w2 = x2t.each_col() / sqrtdiagSvec;
  arma::mat R1;
  arma::mat R2;
  double B1, B2, B12;

  if (n1 < p || n2 < p) {
    R1 = (w1.t() * w1) / (n1 - 1); // tr(Sigmahat)
    R2 = (w2.t() * w2) / (n2 - 1);
    B1 = arma::accu(R1 % R1);
    B2 = arma::accu(R2 % R2);
    B12 = arma::trace(w2.t() * w1 * w1.t() * w2) / ((n1 - 1) * (n2 - 1));
  } else {
    R1 = (w1 * w1.t()) / (n1 - 1); // tr(Sigmahat)
    R2 = (w2 * w2.t()) / (n2 - 1);
    B1 = arma::accu(R1 % R1);
    B2 = arma::accu(R2 % R2);
    B12 = arma::trace(w1 * w1.t() * w2 * w2.t()) / ((n1 - 1) * (n2 - 1));
  }

  double trR12 = (n1 - 1) * (n1 - 1) * (B1 - pow(trace(R1), 2) / (n1 - 1)) / (n1 - 2) / (n1 + 1);
  double trR22 = (n2 - 1) * (n2 - 1) * (B2 - pow(trace(R2), 2) / (n2 - 1)) / (n2 - 2) / (n2 + 1);

  double trR2 = (n2 * n2 * B1 + n1 * n1 * B2 + 2 * n1 * n2 * B12) / (n * n);
  double cpn = 1 + trR2 / pow(sqrt(p), 3);

  double trRn2 = (n2 * n2 * trR12 + n1 * n1 * trR22 + 2 * n1 * n2 * B12) / (n * n);
  double dhat = p * p / trRn2;

  arma::vec values(3);
  values(0) = Tnp;
  values(1) = dhat;
  values(2) = cpn;
  return values;
}



// One-way MANOVA
// Test proposed by Schott (2007)
// [[Rcpp::export]]
arma::vec s2007_ks_nabt_cpp(List Y, const arma::vec &n, int p) {
  int g = Y.size(); // number of classes
  int h = g - 1;
  int ss = sum(n);
  int e = ss - g;
  arma::mat ybar(p, g);

  // Precompute necessary values
  arma::vec index = arma::cumsum(n);
  arma::vec ind = arma::zeros(g + 1);
  std::copy(index.begin(), index.end(), ind.begin() + 1);
  arma::mat Ymat(ss, p, arma::fill::zeros);

  #pragma omp parallel for
  for (int i = 0; i < g; i++) {
    arma::mat yi = Y[i];
    Ymat.rows(ind[i], ind[i + 1] - 1) = yi;
    arma::colvec mu = arma::mean(yi.t(), 1);
    ybar.col(i) = mu;
  }

  arma::mat P(ss,ss,fill::zeros);
  arma::mat X(ss,g,fill::zeros);
  arma::vec index1 = arma::cumsum(n);

  arma::vec index0(index1.n_elem + 1);
  index0(0) = 0;
  index0.subvec(1, index1.n_elem) = index1;

  for(int i=0;i<g;++i){
    arma::mat I (n(i), n(i),arma::fill::ones);
    P.submat(index0(i), index0(i), (index0(i + 1)-1), (index0(i + 1)-1)) = I / n(i);
    arma::colvec v(n(i), arma::fill::ones);
    X.submat(index0(i),i,(index0(i + 1)-1),i) = v;
  }


  arma::mat identity = arma::eye(g - 1, g - 1);
  arma::mat column = -arma::ones<arma::mat>(g - 1, 1);
  arma::mat C = join_horiz(identity, column);
  arma::mat XtXinv = diagmat(1/n);
  arma::mat XtXinvCt = X*XtXinv*C.t();

  arma::mat Hmat = XtXinvCt*cholesky_inverse(C*XtXinv*C.t())*XtXinvCt.t();


  arma::mat H1 = arma::eye(ss, ss) - P;


  double trH, trE,trE2;
  if (p < ss) {
    arma::mat E = Ymat.t() * H1 * Ymat;
    arma::mat H = Ymat.t() * Hmat * Ymat;
    trH = arma::trace(H);
    trE= arma::trace(E);
    trE2 = arma::trace(E*E);
  }else{
    arma::mat E1 =  H1 * Ymat*Ymat.t();
    arma::mat H1 =  Hmat * Ymat*Ymat.t();
    trH = arma::trace(H1);
    trE= arma::trace(E1);
    trE2 = arma::trace(E1*E1);
  }


  double tnp = (trH / h - trE / e) / sqrt(ss - 1);
  double a = (trE2 - pow(trE, 2) / e) / (e + 2) / (e - 1);
  double sigmahat2 = 2 * a / e / h;
  double sigmahat = sqrt(sigmahat2);

  arma::vec stats(2);
  stats(0) = tnp;
  stats(1) = sigmahat;
  return stats;
}

// General linear hypothesis testing (GLHT) problem
// Test proposed by Fujikoshi et al. (2004)
// [[Rcpp::export]]
double fhw2004_glht_nabt_cpp(List Y, const arma::mat &X, const arma::mat &C, const arma::vec &n, int p) {
  int k = Y.size(); // number of classes
  int q = rank(C);
  int ss = sum(n);
  arma::mat Ymat(ss, p);

  arma::vec index = cumsum(n);
  arma::vec ind = zeros(k + 1);
  std::copy(index.begin(), index.end(), ind.begin() + 1);
  for (int i = 0; i < k; i++) {
    arma::mat yi = Y[i];
    Ymat.rows(ind[i], ind[i + 1] - 1) = yi;
  }

  arma::mat XtXinv = cholesky_inverse(X.t() * X);
  arma::mat H = X * XtXinv * C.t() * cholesky_inverse(C * XtXinv * C.t()) * C * XtXinv * X.t();

  arma::mat P = X * XtXinv * X.t();
  arma::mat I = eye(ss, ss);

  double trSh,trSe, trSe2;
  if (p < ss) {
    // Calculate Tnp
    trSh = arma::trace(Ymat.t() * H * Ymat);
    arma::mat Se = Ymat.t() * (I - P) * Ymat;
    trSe = arma::trace(Se);
    trSe2 = arma::trace(Se*Se);
  }else{
    trSh = arma::trace(H * Ymat*Ymat.t());
    arma::mat H1 = arma::eye(ss, ss) - P;
    arma::mat R1 = H1*Ymat*Ymat.t();
    trSe = arma::trace((I - P) * Ymat*Ymat.t());
    trSe2 = arma::trace(R1*R1);
  }

  double sigmaD = sqrt(2 * q * (trSe2 / pow(ss - k, 2) - pow(trSe, 2) / pow(ss - k, 3)) / p) / (trSe / (ss - k) / p);
  double TFHW = sqrt(p) * ((ss - k) * trSh / trSe - q) / sigmaD;
  return TFHW;
}

// Test proposed by Srivastava and Fujikoshi (2006)
// [[Rcpp::export]]
double sf2006_glht_nabt_cpp(List Y, const arma::mat &X, const arma::mat &C, const arma::vec &n, int p) {
  int k = Y.size(); // number of classes
  int q = rank(C);
  int ss = sum(n); // total sample size
  arma::mat Ymat(ss, p); // initialize Y matrix

  // Cumulative index for splitting Y
  arma::vec index = cumsum(n);
  arma::vec ind = arma::zeros(k + 1);
  std::copy(index.begin(), index.end(), ind.begin() + 1);

  // Fill Ymat with class data from list Y
  for (int i = 0; i < k; i++) {
    arma::mat yi = Y[i];
    Ymat.rows(ind[i], ind[i + 1] - 1) = yi;
  }

  // Apply regularization to XtX for numerical stability
  arma::mat XtXinv = cholesky_inverse(X.t() * X);
  arma::mat H = X * XtXinv * C.t() * cholesky_inverse(C * XtXinv * C.t()) * C * XtXinv * X.t();
  arma::mat P = X * XtXinv * X.t();
  arma::mat I = arma::eye(ss, ss); // Identity matrix

  double trSh, trSe, trSe2;

  if (p < ss) {
    // If the number of features is less than total samples
    trSh = arma::trace(Ymat.t() * H * Ymat);
    arma::mat Se = Ymat.t() * (I - P) * Ymat;
    trSe = arma::trace(Se);
    trSe2 = arma::trace(Se * Se);
  } else {
    // If the number of features is greater than or equal to total samples
    trSh = arma::trace(H * Ymat * Ymat.t());
    arma::mat H1 = I - P;
    arma::mat R1 = H1 * Ymat * Ymat.t();
    trSe = arma::trace((I - P) * Ymat * Ymat.t());
    trSe2 = arma::trace(R1 * R1);
  }

  // Compute a2 and TSF
  double a2 = (trSe2 - std::pow(trSe, 2) / (ss - k)) / (ss - k - 1) / (ss - k + 2) / p;
  double TSF = (trSh / std::sqrt(p) - q * trSe / std::sqrt(ss - k) / std::sqrt((ss - k) * p)) /
    std::sqrt(2 * q * a2 * (1 + q / (ss - k)));

  return TSF;
}


// Test proposed by Yamada and Srivastava (2012)
// [[Rcpp::export]]
arma::vec ys2012_glht_nabt_cpp(const Rcpp::List& Y, const arma::mat& X, const arma::mat& C, const arma::vec& n, int p) {
  int k = Y.size(); // number of classes
  int q = C.n_rows; // rank of C should be its row number
  int ss = arma::sum(n);

  // Precompute necessary values
  arma::vec index = arma::cumsum(n);
  arma::vec ind = arma::zeros(k + 1);
  std::copy(index.begin(), index.end(), ind.begin() + 1);

  arma::mat Ymat(ss, p, arma::fill::zeros);

  // Parallelize data filling for performance optimization
#pragma omp parallel for
  for (int i = 0; i < k; i++) {
    arma::mat yi = Y[i];
    Ymat.rows(ind[i], ind[i + 1] - 1) = yi;
  }

  // Compute XtX and its inverse using Cholesky decomposition with regularization
  arma::mat XtX = X.t() * X;
  arma::mat XtXinv = cholesky_inverse(XtX); // Regularization to avoid singular matrix

  // Precompute XtXinv * C.t() and C * XtXinv * C.t() * inv
  arma::mat XtXinvC = XtXinv * C.t();
  arma::mat invC_XtXinvC = cholesky_inverse(C * XtXinvC); // Regularization for stability

  // Compute H matrix
  arma::mat H = X * XtXinvC * invC_XtXinvC * XtXinvC.t() * X.t();

  // Calculate Sh and Se (Error and projection matrices)
  arma::mat Ymat_t = Ymat.t();
  arma::mat Sh = Ymat_t * H * Ymat;
  arma::mat P = X * XtXinv * X.t();
  arma::mat Se = Ymat_t * (arma::eye(ss, ss) - P) * Ymat;

  // Calculate Sigma and its inverse diagonal (if needed)
  arma::mat Sigma = Se / (ss - k);
  arma::vec Sigma_diag = Sigma.diag();

  // Prevent very small values in Sigma_diag to avoid NaN issues
  Sigma_diag = arma::clamp(Sigma_diag, 1e-10, arma::datum::inf);

  arma::vec invSigma_diag = 1 / Sigma_diag;
  arma::mat invD = arma::diagmat(invSigma_diag);

  double Tnp, trRhat2, TYS;

  // Different paths based on p and ss
  if (p < ss) {
    // Path for p < ss
    Tnp = arma::trace(Sh.each_col() % invSigma_diag) / (p * q);
    arma::mat invDSigma = invD * Sigma;
    trRhat2 = arma::trace(invDSigma * invDSigma);
  } else {
    // Path for p >= ss
    Tnp = arma::trace(H * Ymat * invD * Ymat_t) / (p * q);
    arma::mat H1 = arma::eye(ss, ss) - P;
    arma::mat R1 = H1 * Ymat * invD * Ymat_t;
    trRhat2 = arma::trace(R1 * R1) / ((ss - k) * (ss - k));
  }

  // Calculate cpn based on trRhat2 (common part)
  double cpn = 1 + trRhat2 / sqrt(pow(p, 3));

  // Calculate TYS statistic (using trRhat2 and cpn)
  TYS = (p * q * Tnp - (ss - k) * p * q / (ss - k - 2)) / sqrt(2 * q * (trRhat2 - p * p / (ss - k)) * cpn);

  // Return the results
  arma::vec values(2);
  values(0) = TYS;
  values(1) = cpn;
  return values;
}

// Test proposed by Zhou et al. (2017)
// Input: Y[i] is n_i x p  (same convention as wzz2026)
// [[Rcpp::export]]
arma::vec zgz2017_glhtbf_nabt_cpp(List Y, const arma::mat &tG, const arma::vec &n, int p) {

  int k  = Y.size();
  int ss = (int)arma::sum(n);

  if ((int)n.n_elem != k) Rcpp::stop("Length of n must equal Y.size().");
  if (p <= 0) Rcpp::stop("p must be positive.");
  if ((int)tG.n_cols != k) Rcpp::stop("tG must have k columns.");

  // ---- Preconvert + center once (avoid proxy + repeated work) ----
  std::vector<arma::mat> Yv(k);
  std::vector<arma::mat> Z(k);             // n_i x p centered
  arma::mat hatM(p, k, arma::fill::zeros); // p x k means

  for (int i=0; i<k; ++i) {
    Yv[i] = Rcpp::as<arma::mat>(Y[i]); // n_i x p
    if ((int)Yv[i].n_rows != (int)n(i)) Rcpp::stop("Each Y[i] must have n(i) rows.");
    if ((int)Yv[i].n_cols != p)         Rcpp::stop("Each Y[i] must have p columns.");
    if (n(i) < 4) Rcpp::stop("Each group size n(i) must be >= 4.");

    arma::rowvec mui = arma::mean(Yv[i], 0); // 1 x p
    hatM.col(i) = mui.t();
    Z[i] = Yv[i].each_row() - mui;           // n_i x p
  }

  // ---- H = tG^T (tG D tG^T)^-1 tG ----
  arma::mat D = arma::diagmat(1.0 / n);
  arma::mat tmp = tG * D * tG.t();
  tmp.diag() += 1e-12; // tiny ridge for stability
  arma::mat H = tG.t() * cholesky_inverse(tmp) * tG;

  arma::vec A(k, arma::fill::zeros);      // tr(Si)
  arma::vec B(k, arma::fill::zeros);      // tr(Si^2)
  arma::vec Q(k, arma::fill::zeros);      // (1/(ni-1))*sum ||x||^4
  arma::mat Bij(k, k, arma::fill::zeros); // tr(Si Sj)
  double trOmegan = 0.0;

  // ---- Q(i) is same under both branches; compute from centered Z[i] ----
  for (int i=0; i<k; ++i) {
    double ni1 = (double)n(i) - 1.0;
    arma::vec rownorm2 = arma::sum(Z[i] % Z[i], 1);   // length n_i
    Q(i) = arma::dot(rownorm2, rownorm2) / ni1;       // sum ||x||^4 /(ni-1)
  }

  // ------------------------------------------------------------
  // Case 1: p < ss  -> build p x p covariances Si
  // Case 2: p >= ss -> Gram / Frobenius identities (HDLSS)
  // ------------------------------------------------------------
  if (p < ss) {
    std::vector<arma::mat> S(k);

    for (int i=0; i<k; ++i) {
      double ni1 = (double)n(i) - 1.0;
      arma::mat Si = (Z[i].t() * Z[i]) / ni1;   // p x p
      S[i] = Si;
      A(i) = arma::trace(Si);
      B(i) = arma::accu(Si % Si);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<k; ++i) {
      for (int j=i; j<k; ++j) {
        double val = arma::accu(S[i] % S[j]);   // tr(Si Sj)
        Bij(i,j) = val;
        Bij(j,i) = val;
      }
    }

  } else {
    // HDLSS branch: use Gram/Frobenius
    std::vector<arma::mat> G(k);

    for (int i=0; i<k; ++i) {
      double ni1 = (double)n(i) - 1.0;
      G[i] = Z[i] * Z[i].t();                   // n_i x n_i
      A(i) = arma::trace(G[i]) / ni1;
      B(i) = arma::accu(G[i] % G[i]) / (ni1*ni1);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<k; ++i) {
      for (int j=i; j<k; ++j) {
        double ni1 = (double)n(i) - 1.0;
        double nj1 = (double)n(j) - 1.0;
        arma::mat C = Z[i] * Z[j].t();          // n_i x n_j
        double fro2 = arma::accu(C % C);
        double val  = fro2 / (ni1 * nj1);       // tr(Si Sj)
        Bij(i,j) = val;
        Bij(j,i) = val;
      }
    }
  }

  // tr(Omega_hat) = sum_i h_ii * tr(Si)/n_i
  for (int i=0; i<k; ++i) {
    trOmegan += H(i,i) * A(i) / (double)n(i);
  }

  // Tnp = tr(hatM H hatM^T) - tr(Omega_hat)
  arma::mat Msmall = hatM.t() * hatM; // k x k
  double Tnp = arma::trace(H * Msmall) - trOmegan;

  // unbiased uB(i) for tr(Sigma_i^2) as in your original code
  arma::vec uB(k, arma::fill::zeros);
  double K2s1 = 0.0;
  arma::mat K2s2mat(k, k, arma::fill::zeros);

  for (int i=0; i<k; ++i) {
    double ni = (double)n(i);
    uB(i) = (ni - 1.0) * ((ni - 1.0) * (ni - 2.0) * B(i) + A(i) * A(i) - ni * Q(i))
      / (ni * (ni - 2.0) * (ni - 3.0));

    K2s1 += H(i,i) * H(i,i) * uB(i) / (ni * (ni - 1.0));

    for (int j=0; j<k; ++j) {
      K2s2mat(i,j) = H(i,j) * H(i,j) * Bij(i,j) / (ni * (double)n(j));
    }
  }

  double K2s2 = arma::accu(K2s2mat) - arma::sum(K2s2mat.diag());
  double sigma2 = 2.0 * (K2s1 + K2s2);

  arma::vec stats(2);
  stats(0) = Tnp;
  stats(1) = sigma2;
  return stats;
}


// Test proposed by Zhang et al. (2017)
// [[Rcpp::export]]
arma::vec zgz2017_glht_2cnrt_cpp(List Y, const arma::mat &tG, const arma::vec &n, int p) {
  int k = Y.size(); // number of classes
  int q = rank(tG);
  int ss = sum(n);
  arma::mat D = diagmat(1 / n);
  arma::mat H = tG.t() * cholesky_inverse(tG * D * tG.t()) * tG;
  arma::mat hatM(p, k);

  // Precompute necessary values
  arma::vec index = arma::cumsum(n);
  arma::vec ind = arma::zeros(k + 1);
  std::copy(index.begin(), index.end(), ind.begin() + 1);
  arma::mat Ymat(ss, p, arma::fill::zeros);

#pragma omp parallel for
  for (int i = 0; i < k; i++) {
    arma::mat yi = Y[i];
    Ymat.rows(ind[i], ind[i + 1] - 1) = yi;
    arma::colvec mu = arma::mean(yi.t(), 1);
    hatM.col(i) = mu;
  }


  arma::mat P(ss,ss,fill::zeros);
  arma::vec index1 = arma::cumsum(n);

  arma::vec index0(index1.n_elem + 1);
  index0(0) = 0;
  index0.subvec(1, index1.n_elem) = index1;

  for(int i=0;i<k;++i){
    arma::mat I (n(i), n(i),arma::fill::ones);
    P.submat(index0(i), index0(i), (index0(i + 1)-1), (index0(i + 1)-1)) = I / n(i);
  }

  arma::mat H1 = arma::eye(ss, ss) - P;

  double Tnp, trSn,trSn2;
  if (p < ss) {
    // Calculate Tnp
    Tnp = trace(hatM * H * hatM.t());
    // Calculate trRhat2 and trR2
    arma::mat S = Ymat.t() * H1 * Ymat/(ss-k);
    trSn= arma::trace(S);
    trSn2 = arma::trace(S*S);
  }else{
    Tnp = trace(H * hatM.t()*hatM);
    arma::mat S1 =  H1 * Ymat*Ymat.t()/(ss-k);
    trSn= arma::trace(S1);
    trSn2 = arma::trace(S1*S1);
  }
  double uA = (ss - k) * (ss - k + 1) * (pow(trSn, 2) - 2 * trSn2 / (ss - k + 1)) / (ss - k - 1) / (ss - k + 2);
  double uB = pow(ss - k, 2) * (trSn2 - pow(trSn, 2) / (ss - k)) / (ss - k - 1) / (ss - k + 2);

  double beta = uB / trSn;
  double d = q * uA / uB;
  arma::vec stats(4);
  stats(0) = Tnp;
  stats(1) = beta;
  stats(2) = d;
  return stats;
}


// Test proposed by Zhang et al. (2022)  [ZZG 2-c NRT]
// Input: Y[i] is n_i x p
// [[Rcpp::export]]
arma::vec zzg2022_glhtbf_2cnrt_cpp(List Y, const arma::mat &tG, const arma::vec &n, int p) {

  int k  = Y.size();
  int ss = (int)arma::sum(n);

  if ((int)n.n_elem != k) Rcpp::stop("Length of n must equal Y.size().");
  if (p <= 0) Rcpp::stop("p must be positive.");
  if ((int)tG.n_cols != k) Rcpp::stop("tG must have k columns.");

  // ---- Preconvert + center once ----
  std::vector<arma::mat> Yv(k);
  std::vector<arma::mat> Z(k);             // n_i x p centered
  arma::mat hatM(p, k, arma::fill::zeros); // p x k means

  for (int i=0; i<k; ++i) {
    Yv[i] = Rcpp::as<arma::mat>(Y[i]); // n_i x p
    if ((int)Yv[i].n_rows != (int)n(i)) Rcpp::stop("Each Y[i] must have n(i) rows.");
    if ((int)Yv[i].n_cols != p)         Rcpp::stop("Each Y[i] must have p columns.");
    if (n(i) < 4) Rcpp::stop("Each group size n(i) must be >= 4.");

    arma::rowvec mui = arma::mean(Yv[i], 0);
    hatM.col(i) = mui.t();
    Z[i] = Yv[i].each_row() - mui;
  }

  // ---- H, D, D2, Amat ----
  arma::mat D  = arma::diagmat(1.0 / n);
  arma::mat D2 = arma::diagmat(1.0 / arma::sqrt(n));

  arma::mat tmp = tG * D * tG.t();
  tmp.diag() += 1e-12;
  arma::mat H = tG.t() * cholesky_inverse(tmp) * tG;

  arma::mat Amat = D2 * H * D2;

  arma::vec A(k, arma::fill::zeros);      // tr(Si)
  arma::vec B(k, arma::fill::zeros);      // tr(Si^2)
  arma::mat Bij(k, k, arma::fill::zeros); // tr(Si Sj)

  // ---- A,B,Bij in two branches ----
  if (p < ss) {
    std::vector<arma::mat> S(k);

    for (int i=0; i<k; ++i) {
      double ni1 = (double)n(i) - 1.0;
      arma::mat Si = (Z[i].t() * Z[i]) / ni1; // p x p
      S[i] = Si;
      A(i) = arma::trace(Si);
      B(i) = arma::accu(Si % Si);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<k; ++i) {
      for (int j=i; j<k; ++j) {
        double val = arma::accu(S[i] % S[j]); // tr(Si Sj)
        Bij(i,j) = val;
        Bij(j,i) = val;
      }
    }

  } else {
    std::vector<arma::mat> G(k);

    for (int i=0; i<k; ++i) {
      double ni1 = (double)n(i) - 1.0;
      G[i] = Z[i] * Z[i].t(); // n_i x n_i
      A(i) = arma::trace(G[i]) / ni1;
      B(i) = arma::accu(G[i] % G[i]) / (ni1*ni1);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<k; ++i) {
      for (int j=i; j<k; ++j) {
        double ni1 = (double)n(i) - 1.0;
        double nj1 = (double)n(j) - 1.0;
        arma::mat C = Z[i] * Z[j].t();      // n_i x n_j
        double fro2 = arma::accu(C % C);
        double val  = fro2 / (ni1 * nj1);   // tr(Si Sj)
        Bij(i,j) = val;
        Bij(j,i) = val;
      }
    }
  }

  // ---- Tnp ----
  arma::mat Msmall = hatM.t() * hatM;      // k x k
  double Tnp = arma::trace(H * Msmall);

  // ---- uB, uC, traces ----
  arma::vec uB(k, arma::fill::zeros);
  arma::vec uC(k, arma::fill::zeros);
  arma::mat S1(k, k, arma::fill::zeros);
  arma::mat S2(k, k, arma::fill::zeros);

  double trOmegan = 0.0;
  double tr2Omegans1 = 0.0;
  double trOmegan2s1 = 0.0;

  for (int i=0; i<k; ++i) {
    double ni = (double)n(i);

    uB(i) = (ni - 1.0) * (ni - 1.0) * (B(i) - A(i)*A(i)/(ni - 1.0)) / ((ni - 2.0)*(ni + 1.0));
    uC(i) = ni * (ni - 1.0) * (A(i)*A(i) - 2.0*B(i)/ni) / ((ni - 2.0)*(ni + 1.0));

    trOmegan     += Amat(i,i) * A(i);
    tr2Omegans1  += Amat(i,i) * Amat(i,i) * uC(i);
    trOmegan2s1  += Amat(i,i) * Amat(i,i) * uB(i);

    for (int j=0; j<k; ++j) {
      S1(i,j) = Amat(i,i) * Amat(j,j) * A(i) * A(j);
      S2(i,j) = Amat(i,j) * Amat(i,j) * Bij(i,j);
    }
  }

  double tr2Omegan = tr2Omegans1 + arma::accu(S1) - arma::sum(S1.diag());
  double trOmegan2 = trOmegan2s1 + arma::accu(S2) - arma::sum(S2.diag());

  double beta = trOmegan2 / trOmegan;
  double d    = tr2Omegan / trOmegan2;

  arma::vec stats(3);
  stats(0) = Tnp;
  stats(1) = beta;
  stats(2) = d;
  return stats;
}



// Test proposed by Zhang and Zhu (2022)  (3-c matched NRT)
// Input: Y[i] is n_i x p
// [[Rcpp::export]]
arma::vec zz2022_glht_3cnrt_cpp(List Y, const arma::mat &tG, const arma::vec &n, int p) {

  int k  = Y.size();
  int ss = (int)arma::sum(n);
  int q  = (int)arma::rank(tG);

  if ((int)n.n_elem != k) Rcpp::stop("Length of n must equal Y.size().");
  if (p <= 0) Rcpp::stop("p must be positive.");
  if ((int)tG.n_cols != k) Rcpp::stop("tG must have k columns (k = Y.size()).");
  if (ss <= k) Rcpp::stop("Total sample size ss must be > k.");

  // ---- H = tG^T (tG D tG^T)^{-1} tG  ----
  arma::mat D = arma::diagmat(1.0 / n);
  arma::mat tmp = tG * D * tG.t();
  tmp.diag() += 1e-12; // tiny ridge for stability
  arma::mat H = tG.t() * cholesky_inverse(tmp) * tG;

  // ---- Preconvert + center once; build hatM (p x k) and Zall (p x ss) ----
  std::vector<arma::mat> Yv(k);
  arma::mat hatM(p, k, arma::fill::zeros);
  arma::mat Zall(p, ss, arma::fill::zeros);   // concatenated centered columns (p x ss)

  int col_start = 0;
  for (int i = 0; i < k; ++i) {
    Yv[i] = Rcpp::as<arma::mat>(Y[i]);         // n_i x p
    int ni = (int)n(i);

    if (ni < 4) Rcpp::stop("Each group size n(i) must be >= 4.");
    if ((int)Yv[i].n_rows != ni) Rcpp::stop("Each Y[i] must have n(i) rows.");
    if ((int)Yv[i].n_cols != p)  Rcpp::stop("Each Y[i] must have p columns.");

    // transpose to p x n_i (same orientation as your original code)
    arma::mat YiT = Yv[i].t();                 // p x n_i
    arma::colvec mui = arma::mean(YiT, 1);     // p x 1
    hatM.col(i) = mui;

    // center and place into Zall block
    // Zi = YiT - mui * 1^T
    Zall.cols(col_start, col_start + ni - 1) = YiT.each_col() - mui;

    col_start += ni;
  }
  if (col_start != ss) Rcpp::stop("Internal error: concatenation size mismatch.");

  // ---- Build S (same as your original) ----
  arma::mat S;
  double ss_k = (double)(ss - k);

  if (p < ss) {
    // p x p
    S = (Zall * Zall.t()) / ss_k;
  } else {
    // ss x ss
    S = (Zall.t() * Zall) / ss_k;
  }

  // ---- Tnp (keep original definition) ----
  double trS = arma::trace(S);
  double Tnp = arma::trace(hatM * H * hatM.t()) - (double)q * trS;

  // ---- 3-c matching ingredients ----
  double trS2 = arma::accu(S % S);
  double trS3 = arma::trace(S * S * S);

  // unbiased-type transforms (same formulas as your original)
  double htrS2 = std::pow(ss_k, 2.0) * (trS2 - trS * trS / ss_k) / (ss_k - 1.0) / (ss_k + 2.0);

  double denom1 = (std::pow(ss_k, 2.0) + 2.0 * ss_k - 3.0);
  double denom2 = (std::pow(ss_k, 2.0) - 4.0);
  double htrS3 = std::pow(ss_k, 4.0) *
    (trS3 - 3.0 * trS * trS2 / ss_k + 2.0 * std::pow(trS, 3.0) / (ss_k * ss_k))
    / denom1 / denom2;

  double ss_k_minus_q = (double)(ss - k - q);
  if (!(htrS2 > 0.0)) Rcpp::stop("htrS2 is non-positive; cannot form 3-c matched approximation.");
  if (!(htrS3 > 0.0)) Rcpp::stop("htrS3 is non-positive; cannot form 3-c matched approximation.");
  if (!(ss_k_minus_q > 0.0)) Rcpp::stop("ss - k - q must be positive for 3-c matched approximation.");

  double beta0 = - (double)q * (ss_k + (double)q) * (htrS2 * htrS2) / htrS3 / ss_k_minus_q;
  double beta1 = ss_k_minus_q * htrS3 / htrS2 / ss_k;
  double d     = (double)q * ss_k * (ss_k + (double)q) * std::pow(htrS2, 3.0)
    / std::pow(htrS3, 2.0) / (ss_k_minus_q * ss_k_minus_q);

  arma::vec stats(4);
  stats(0) = Tnp;
  stats(1) = beta0;
  stats(2) = beta1;
  stats(3) = d;
  return stats;
}


// Test proposed by Zhang and Zhu (2022)
// 3-c matched chi-square approximation for heteroscedastic GLHT
// Input: Y[i] is n_i x p
// [[Rcpp::export]]
arma::vec zz2022_glhtbf_3cnrt_cpp(const Rcpp::List& Y,
                                  const arma::mat& tG,
                                  const arma::vec& n,
                                  int p) {

  int k  = (int)Y.size();
  int ss = (int)arma::sum(n);

  if ((int)n.n_elem != k) Rcpp::stop("Length of n must equal Y.size().");
  if (p <= 0)             Rcpp::stop("p must be positive.");
  if ((int)tG.n_cols != k) Rcpp::stop("tG must have k columns.");

  for (int i=0; i<k; ++i) {
    if (n(i) < 4) Rcpp::stop("Each group size n(i) must be >= 4.");
  }

  // ---- H = tG^T (tG D tG^T)^{-1} tG ----
  arma::mat D = arma::diagmat(1.0 / n);
  arma::mat H = tG.t() * cholesky_inverse(tG * D * tG.t()) * tG;

  // ---- Preconvert + center once (Z[i] stored as p x n_i, like your original) ----
  arma::mat hatM(p, k, arma::fill::zeros);
  std::vector<arma::mat> Yv(k);
  std::vector<arma::mat> Z(k);          // p x n_i centered
  std::vector<arma::mat> Graw(k);       // n_i x n_i, Graw[i] = Z[i].t() * Z[i] (unscaled), for HDLSS Dij

  for (int i=0; i<k; ++i) {
    Yv[i] = Rcpp::as<arma::mat>(Y[i]);          // n_i x p
    int ni = (int)n(i);
    if ((int)Yv[i].n_rows != ni) Rcpp::stop("Each Y[i] must have n(i) rows.");
    if ((int)Yv[i].n_cols != p)  Rcpp::stop("Each Y[i] must have p columns.");

    arma::mat yi = Yv[i].t();                   // p x n_i
    arma::colvec mui = arma::mean(yi, 1);       // p x 1
    hatM.col(i) = mui;
    Z[i] = yi.each_col() - mui;                 // p x n_i
    // keep raw Gram for HDLSS Dij
    Graw[i] = Z[i].t() * Z[i];                  // n_i x n_i
  }

  bool use_p = (p < ss);

  arma::vec A(k, arma::fill::zeros);
  arma::vec B(k, arma::fill::zeros);
  arma::vec C(k, arma::fill::zeros);
  arma::mat Bij(k, k, arma::fill::zeros);
  arma::mat Dij(k, k, arma::fill::zeros);

  double trOmegan = 0.0;

  // =========================
  // 1) Compute A,B,C and trOmegan
  // =========================
  if (use_p) {
    // Build S_i in p-space
    std::vector<arma::mat> S(k);  // p x p
    for (int i=0; i<k; ++i) {
      double ni1 = (double)n(i) - 1.0;
      S[i] = (Z[i] * Z[i].t()) / ni1;           // p x p

      A(i) = arma::trace(S[i]);
      B(i) = arma::accu(S[i] % S[i]);
      C(i) = arma::trace(S[i] * S[i] * S[i]);

      trOmegan += H(i,i) * A(i) / (double)n(i);
    }

    // Bij(i,j) = tr(Si Sj), Dij(i,j) = tr(Si Si Sj)
    for (int i=0; i<k; ++i) {
      for (int j=0; j<k; ++j) {
        Bij(i,j) = arma::accu(S[i] % S[j]);                 // tr(Si Sj)
        Dij(i,j) = arma::trace(S[i] * S[i] * S[j]);         // tr(Si^2 Sj)
      }
    }

    // K3s3 will use tr(Si Sj Sr)
    // (computed later)

  } else {
    // HDLSS: build S_i in sample space (n_i x n_i)
    std::vector<arma::mat> S(k);  // n_i x n_i
    for (int i=0; i<k; ++i) {
      double ni1 = (double)n(i) - 1.0;
      S[i] = Graw[i] / ni1;                                 // n_i x n_i

      A(i) = arma::trace(S[i]);
      B(i) = arma::accu(S[i] % S[i]);
      C(i) = arma::trace(S[i] * S[i] * S[i]);

      trOmegan += H(i,i) * A(i) / (double)n(i);
    }

    // Bij(i,j) = ||Zi^T Zj||_F^2 / ((ni-1)(nj-1))
    // Dij(i,j) = tr( (Zi^T Zi) * (Zi^T Zj)(Zj^T Zi) ) / ((ni-1)^2 (nj-1))
    for (int i=0; i<k; ++i) {
      double ni  = (double)n(i);
      double ni1 = ni - 1.0;

      for (int j=0; j<k; ++j) {
        double nj  = (double)n(j);
        double nj1 = nj - 1.0;

        arma::mat Cij = Z[i].t() * Z[j];                    // n_i x n_j
        double fro2 = arma::accu(Cij % Cij);                // ||Cij||_F^2
        Bij(i,j) = fro2 / (ni1 * nj1);

        // Dij numerator: tr( (Zi^T Zi) * (Cij Cij^T) )
        arma::mat M = Cij * Cij.t();                         // n_i x n_i (symmetric)
        double tr_val = arma::accu(Graw[i] % M);              // tr(Graw_i * M)
        Dij(i,j) = tr_val / (ni1*ni1*nj1);
      }
    }
  }

  // =========================
  // 2) Test statistic
  // =========================
  double Tnp = arma::trace(hatM * H * hatM.t()) - trOmegan;

  // =========================
  // 3) uB, uC, K2s1, K3s1
  // =========================
  arma::vec uB(k, arma::fill::zeros);
  arma::vec uC(k, arma::fill::zeros);
  double K2s1 = 0.0, K3s1 = 0.0;

  for (int i=0; i<k; ++i) {
    double ni  = (double)n(i);
    double ni1 = ni - 1.0;

    double ccoef = std::pow(ni1, 4.0) / ((ni*ni + ni - 6.0) * (ni*ni - 2.0*ni - 3.0));

    uB(i) = (ni1*ni1) * (B(i) - A(i)*A(i)/ni1) / ((ni - 2.0) * (ni + 1.0));
    uC(i) = ccoef * (C(i) - 3.0*A(i)*B(i)/ni1 + 2.0*std::pow(A(i),3.0)/(ni1*ni1));

    K2s1 += H(i,i) * H(i,i) * uB(i) / (ni * ni1);
    K3s1 += std::pow(H(i,i),3.0) * (ni - 2.0) * uC(i) / (ni*ni * ni1*ni1);
  }

  // =========================
  // 4) K2s2, K3s2
  // =========================
  arma::mat K2s2mat(k, k, arma::fill::zeros);
  arma::mat K3s2mat(k, k, arma::fill::zeros);

  for (int i=0; i<k; ++i) {
    double ni  = (double)n(i);
    double ni1 = ni - 1.0;

    for (int j=0; j<k; ++j) {
      double nj = (double)n(j);

      double Cij_unb = ni1 * (ni1 * Dij(i,j) - Bij(i,j) * A(i)) / ((ni - 2.0) * (ni + 1.0));

      K2s2mat(i,j) = H(i,j) * H(i,j) * Bij(i,j) / (ni * nj);
      K3s2mat(i,j) = H(i,i) * H(i,j) * H(i,j) * Cij_unb / (ni*ni * nj);
    }
  }

  double K2s2 = arma::accu(K2s2mat) - arma::sum(K2s2mat.diag());
  double K3s2 = arma::accu(K3s2mat) - arma::sum(K3s2mat.diag());

  // =========================
  // 5) K3s3 (k is tiny; compute directly; reuse cached matrices)
  // =========================
  double K3s3 = 0.0;

  if (use_p) {
    // Use S_i = Zi Zi^T/(ni-1) implicitly via Bij/Dij path is not stored now;
    // compute tri = tr(Si Sj Sr) directly from cached Z:
    // tri = tr( Zi Zi^T Zj Zj^T Zr Zr^T ) / ((ni-1)(nj-1)(nr-1))
    // We compute it as tr(Si Sj Sr) by building Si,Sj,Sr on the fly (p is small here).
    for (int i=2; i<k; ++i) {
      for (int j=1; j<i; ++j) {
        for (int r=0; r<j; ++r) {
          double ni1 = (double)n(i) - 1.0;
          double nj1 = (double)n(j) - 1.0;
          double nr1 = (double)n(r) - 1.0;

          arma::mat Si = (Z[i] * Z[i].t()) / ni1;
          arma::mat Sj = (Z[j] * Z[j].t()) / nj1;
          arma::mat Sr = (Z[r] * Z[r].t()) / nr1;

          double tri = arma::trace(Si * Sj * Sr);
          K3s3 += H(i,j) * H(j,r) * H(r,i) * tri / ((double)n(i) * (double)n(j) * (double)n(r));
        }
      }
    }

  } else {
    // HDLSS: tri = trace( (Zi^T Zj)(Zj^T Zr)(Zr^T Zi) ) / ((ni-1)(nj-1)(nr-1))
    for (int i=2; i<k; ++i) {
      for (int j=1; j<i; ++j) {
        for (int r=0; r<j; ++r) {
          double ni1 = (double)n(i) - 1.0;
          double nj1 = (double)n(j) - 1.0;
          double nr1 = (double)n(r) - 1.0;

          arma::mat Cij = Z[i].t() * Z[j];   // n_i x n_j
          arma::mat Cjr = Z[j].t() * Z[r];   // n_j x n_r
          arma::mat Cri = Z[r].t() * Z[i];   // n_r x n_i

          double tri = arma::trace(Cij * Cjr * Cri) / (ni1 * nj1 * nr1);
          K3s3 += H(i,j) * H(j,r) * H(r,i) * tri / ((double)n(i) * (double)n(j) * (double)n(r));
        }
      }
    }
  }

  // =========================
  // 6) K2, K3, beta0, beta1, d
  // =========================
  double K2 = 2.0 * (K2s1 + K2s2);
  double K3 = 8.0 * (K3s1 + 3.0 * K3s2 + 6.0 * K3s3);

  double beta0 = -2.0 * K2 * K2 / K3;
  double beta1 = K3 / (4.0 * K2);
  double d     = 8.0 * K2 * K2 * K2 / (K3 * K3);

  arma::vec stats(4);
  stats(0) = Tnp;
  stats(1) = beta0;
  stats(2) = beta1;
  stats(3) = d;
  return stats;
}


// Test proposed by Zhang and Zhu (2022)
// [[Rcpp::export]]
arma::vec zzz2022_glht_2cnrt_cpp(const Rcpp::List& Y, const arma::mat& X, const arma::mat& C, const arma::vec& n, int p) {
  int k = Y.size(); // number of classes
  int q = C.n_rows; // rank of C should be its row number
  int ss = arma::sum(n);

  // Precompute necessary values
  arma::vec index = arma::cumsum(n);
  arma::vec ind = arma::zeros(k + 1);
  std::copy(index.begin(), index.end(), ind.begin() + 1);

  arma::mat Ymat(ss, p, arma::fill::zeros);

  // Fill Ymat with data
#pragma omp parallel for
  for (int i = 0; i < k; i++) {
    arma::mat yi = Y[i];
    Ymat.rows(ind[i], ind[i + 1] - 1) = yi;
  }

  // Calculate XtXinv using Cholesky decomposition
  arma::mat XtX = X.t() * X;
  arma::mat XtXinv = cholesky_inverse(XtX); // Regularization removed

  // Precompute XtXinv * C.t() and C * XtXinv * C.t() * inv
  arma::mat XtXinvC = XtXinv * C.t();
  arma::mat invC_XtXinvC = cholesky_inverse(C * XtXinvC); // Regularization removed

  // Compute H matrix
  arma::mat H = X * XtXinvC * invC_XtXinvC * XtXinvC.t() * X.t();

  // Calculate Sh and Se
  arma::mat Ymat_t = Ymat.t();
  arma::mat Sh = Ymat_t * H * Ymat;
  arma::mat P = X * XtXinv * X.t();
  arma::mat Se = Ymat_t * (arma::eye(ss, ss) - P) * Ymat;

  // Calculate Sigma and its inverse diagonal
  arma::mat Sigma = Se / (ss - k);
  arma::vec Sigma_diag = Sigma.diag();
  Sigma_diag = arma::clamp(Sigma_diag, 1e-10, arma::datum::inf);// Prevent very small values
  arma::vec invSigma_diag = 1 / Sigma_diag;
  arma::mat invD = arma::diagmat(invSigma_diag);

  double Tnp, trRhat2;
  if (p < ss) {
    // Calculate Tnp
    Tnp = arma::trace(Sh.each_col() % invSigma_diag) / (p * q);
    // Calculate trRhat2 and trR2
    arma::mat invDSigma = invD * Sigma;
    trRhat2 = arma::trace(invDSigma * invDSigma);
  } else {
    Tnp = arma::trace(H * Ymat * invD * Ymat_t) / (p * q);
    arma::mat H1 = arma::eye(ss, ss) - P;
    arma::mat R1 = H1 * Ymat * invD * Ymat_t;
    trRhat2 = arma::trace(R1 * R1) / ((ss - k) * (ss - k));
  }

  // Calculate trR2 and ensure it is positive
  double trR2 = (ss - k) * (ss - k) * (trRhat2 - p * p / (ss - k)) / ((ss - k - 1) * (ss - k + 2));
  if (trR2 <= 1e-6) {
    double adjustment = std::max(1e-6, 0.01 * fabs(trRhat2));
    Rcpp::Rcout << "Warning: trR2 is very small or non-positive. Applying adaptive adjustment.\n";
    trR2 += adjustment; // Apply adaptive adjustment to ensure trR2 is sufficiently positive
  }



  // Calculate degrees of freedom and ensure it is positive
  double hatd = p * p * q / trR2;
  if (hatd <= 1e-6) {
    Rcpp::Rcout << "Warning: Degrees of freedom (hatd) is non-positive. Applying adjustment.\n";
    hatd = std::max(1e-6, 0.01 * fabs(hatd)); // Ensure hatd is positive, with an adaptive adjustment
  }


  // Return the results
  arma::vec values(2);
  values(0) = Tnp;
  values(1) = hatd;
  return values;
}




//F-type test for GLHT problem (input Y[i] is n_i x p)
// [[Rcpp::export]]
arma::vec wz2026_glhtbf_2cnrt_cpp(List Y, const arma::mat &tG, const arma::vec &n, int p){
  int k  = Y.size();
  int ss = arma::sum(n);

  if ((int)n.n_elem != k) Rcpp::stop("Length of n must equal Y.size().");
  if (p <= 0) Rcpp::stop("p must be positive.");
  if ((int)tG.n_cols != k) Rcpp::stop("tG must have k columns (k = Y.size()).");

  // Convert List elements to arma::mat explicitly (avoid proxy issues)
  std::vector<arma::mat> Yv(k);
  std::vector<arma::mat> Z(k);
  arma::mat hatM(p, k, arma::fill::zeros);

  for (int i = 0; i < k; ++i){
    Yv[i] = Rcpp::as<arma::mat>(Y[i]);  // n_i x p
    if (n(i) < 4) Rcpp::stop("Each group size n(i) must be >= 4 (needed for unbiased estimators).");
    if ((int)Yv[i].n_cols != p) Rcpp::stop("Each Y[i] must be an n_i x p matrix (cols = p).");
    if ((int)Yv[i].n_rows != (int)n(i)) Rcpp::stop("Each Y[i] must have n(i) rows.");

    arma::rowvec mui = arma::mean(Yv[i], 0);      // 1 x p
    hatM.col(i) = mui.t();                        // p x 1
    Z[i] = Yv[i].each_row() - mui;                // n_i x p
  }

  // D = diag(1/n)
  arma::mat D = arma::diagmat(1.0 / n);

  // H = tG^T (tG D tG^T)^{-1} tG, add a tiny ridge for numerical stability if needed
  arma::mat tmp = tG * D * tG.t();
  // optional ridge (very small, helps chol if near-singular)
  tmp.diag() += 1e-12;
  arma::mat inv_mat = cholesky_inverse(tmp);
  arma::mat H = tG.t() * inv_mat * tG;

  arma::vec A(k, arma::fill::zeros);       // tr(Si)
  arma::vec B(k, arma::fill::zeros);       // tr(Si^2)
  arma::mat Bij(k, k, arma::fill::zeros);  // tr(Si Sj)

  // ------------------------------------------------------------
  // Case 1: p < ss  -> build p x p covariances Si
  // Case 2: p >= ss -> use Gram / Frobenius identities (HDLSS)
  // ------------------------------------------------------------
  if (p < ss) {
    std::vector<arma::mat> S(k);
    for(int i=0; i<k; ++i){
      double ni = (double)n(i);
      arma::mat Si = (Z[i].t() * Z[i]) / (ni - 1.0); // p x p
      S[i] = Si;
      A(i) = arma::trace(Si);
      B(i) = arma::accu(Si % Si);
    }

    // Bij(i,j) = tr(Si Sj) = accu(Si % Sj) for symmetric matrices
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0; i<k; ++i){
      for(int j=i; j<k; ++j){
        double val = arma::accu(S[i] % S[j]);
        Bij(i,j) = val;
        Bij(j,i) = val;
      }
    }

  } else {
    // HDLSS branch
    for(int i=0; i<k; ++i){
      double ni = (double)n(i);
      arma::mat Gi = Z[i] * Z[i].t(); // n_i x n_i

      A(i) = arma::trace(Gi) / (ni - 1.0);
      B(i) = arma::accu(Gi % Gi) / ((ni - 1.0) * (ni - 1.0));
    }

    // Bij(i,j) = ||Z_i Z_j^T||_F^2 / ((n_i-1)(n_j-1))
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0; i<k; ++i){
      for(int j=i; j<k; ++j){
        double ni = (double)n(i);
        double nj = (double)n(j);
        arma::mat C = Z[i] * Z[j].t();
        double fro2 = arma::accu(C % C);
        double val  = fro2 / ((ni - 1.0) * (nj - 1.0));
        Bij(i,j) = val;
        Bij(j,i) = val;
      }
    }
  }

  // tr(Omega_hat) = sum_i h_ii * tr(Si)/n_i
  double trOmegan = 0.0;
  for(int i=0; i<k; ++i){
    trOmegan += H(i,i) * A(i) / (double)n(i);
  }
  if (!(trOmegan > 0.0)) Rcpp::stop("trOmegan is non-positive; cannot form F-type statistic.");

  // numerator = tr(H * hatM^T * hatM)
  arma::mat Msmall = hatM.t() * hatM;      // k x k
  double num = arma::trace(H * Msmall);
  double Tnp = num / trOmegan;

  // ---- d1, d2 parts ----
  arma::mat S1(k, k, arma::fill::zeros);
  arma::mat S2(k, k, arma::fill::zeros);

  double tr2Omegans1 = 0.0;
  double trOmegan2s1 = 0.0;
  double d2d = 0.0;

  for(int i=0; i<k; ++i){
    double Ai = A(i);
    double Bi = B(i);
    double ni = (double)n(i);

    double tmp_uB = (ni - 1.0)*(ni - 1.0)*(Bi - Ai*Ai/(ni - 1.0))
      / ((ni - 2.0)*(ni + 1.0));
    double tmp_uC = ni*(ni - 1.0)*(Ai*Ai - 2.0*Bi/ni)
      / ((ni - 2.0)*(ni + 1.0));

    double Hii = H(i,i);
    tr2Omegans1 += Hii*Hii*tmp_uC/(ni*ni);
    trOmegan2s1 += Hii*Hii*tmp_uB/(ni*ni);

    // IMPORTANT: denominator for d2 uses h_ii^2
    d2d += Hii*Hii*tmp_uB/(ni*ni*(ni - 1.0));
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i=0; i<k; ++i){
    double Ai = A(i);
    double Hii = H(i,i);
    for(int j=0; j<k; ++j){
      double Aj = A(j);
      S1(i,j) = Hii * H(j,j) * Ai * Aj / ((double)n(i)*(double)n(j));
      S2(i,j) = H(i,j)*H(i,j)*Bij(i,j) / ((double)n(i)*(double)n(j));
    }
  }

  double tr2Omegan = tr2Omegans1 + arma::accu(S1) - arma::sum(S1.diag());
  double trOmegan2 = trOmegan2s1 + arma::accu(S2) - arma::sum(S2.diag());

  if (!(trOmegan2 > 0.0)) Rcpp::stop("tr(Omega_n^2) estimator is non-positive.");
  if (!(d2d > 0.0))       Rcpp::stop("d2 denominator component is non-positive.");

  double d1 = tr2Omegan / trOmegan2;
  double d2 = tr2Omegan / d2d;

  arma::vec stats(4);
  stats(0) = Tnp;
  stats(1) = d1;
  stats(2) = d2;
  stats(3) = 4.0 * trOmegan / d2;
  return stats;
}



// Test via random integration (Li et al. 2024): T_n + variance estimate + standardization
// Input Y[i] is n_i x p (rows are samples)
// B: coefficient vector (length k)
// O: omega vector (length p)
// A: alpha vector (length p)
// n: group sizes (length k), for interface consistency with other GLHT tests
// [[Rcpp::export]]
arma::vec random_integration_test_cpp(const Rcpp::List &Y,
                                          const arma::vec &B,
                                          const arma::vec &O,
                                          const arma::vec &A,
                                          const arma::vec &n,
                                          int p)
{
  int k = Y.size();
  if ((int)B.n_elem != k) Rcpp::stop("Length of B must equal Y.size().");
  if ((int)n.n_elem != k) Rcpp::stop("Length of n must equal Y.size().");
  if ((int)O.n_elem != p || (int)A.n_elem != p) Rcpp::stop("Length of O/A must equal p.");
  if (p <= 0) Rcpp::stop("p must be positive.");

  // Convert list -> vector<mat> first (avoid proxy issues)
  std::vector<arma::mat> X(k), Z(k);
  arma::ivec n_i(k);
  int ss = 0;

  for (int i = 0; i < k; ++i) {
    arma::mat Xi = Rcpp::as<arma::mat>(Y[i]); // n_i x p
    if ((int)Xi.n_cols != p) Rcpp::stop("Each Y[i] must be an n_i x p matrix (cols = p).");
    if ((int)Xi.n_rows < 4)  Rcpp::stop("Each group must have n_i >= 4.");
    if ((int)Xi.n_rows != (int)n(i)) Rcpp::stop("n(i) must match nrow(Y[[i]]).");

    X[i] = Xi;
    n_i(i) = (int)Xi.n_rows;
    ss += n_i(i);

    arma::rowvec mui = arma::mean(Xi, 0);
    Z[i] = Xi.each_row() - mui;
  }

  arma::vec O2 = arma::square(O);

  // Right-multiply by W: XW = X * W, W = diag(O^2) + A*A'
  auto right_mult_W = [&](const arma::mat &Xmat) -> arma::mat {
    arma::mat XW = Xmat;
    XW.each_row() %= O2.t();              // X * diag(O^2)
    XW += (Xmat * A) * A.t();             // + (X*A)*A^T
    return XW;
  };

  // -----------------------------
  // (1) Tn
  // -----------------------------
  double Tn = 0.0;

  // within-group terms
  for (int i = 0; i < k; ++i) {
    int ni = n_i(i);
    arma::mat XW = right_mult_W(X[i]);
    arma::mat M  = XW * X[i].t();
    double sum_offdiag = arma::accu(M) - arma::trace(M);
    Tn += (B(i) * B(i)) * sum_offdiag / (double)(ni * (ni - 1));
  }

  // between-group terms
  double Tn_between = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:Tn_between) schedule(static)
#endif
  for (int i = 0; i < k; ++i) {
    for (int l = i + 1; l < k; ++l) {
      int ni = n_i(i), nl = n_i(l);
      arma::mat XlW = right_mult_W(X[l]);
      arma::mat M   = X[i] * XlW.t();
      double sum_all = arma::accu(M);
      Tn_between += 2.0 * B(i) * B(l) * sum_all / (double)(ni * nl);
    }
  }
  Tn += Tn_between;

  // -----------------------------
  // (2) sigmahat2 with p < ss / p >= ss split
  // -----------------------------
  double sigmahat2 = 0.0;

  if (p < ss) {
    std::vector<arma::mat> Si(k);
    arma::vec trWS(k, arma::fill::zeros);
    arma::vec trWSWS(k, arma::fill::zeros);

    for (int i = 0; i < k; ++i) {
      double ni = (double)n_i(i);
      Si[i] = (Z[i].t() * Z[i]) / (ni - 1.0);

      double term_diag  = arma::dot(O2, Si[i].diag());
      double term_rank1 = arma::as_scalar(A.t() * Si[i] * A);
      trWS(i) = term_diag + term_rank1;

      arma::mat WSi = Si[i];
      WSi.each_row() %= O2.t();                 // Omega * Si
      WSi += A * (A.t() * Si[i]);               // + (A A^T) * Si  => W * Si
      trWSWS(i) = arma::accu(WSi % WSi.t());    // tr(WSi*WSi)

    }

    // group-wise
    for (int i = 0; i < k; ++i) {
      int ni = n_i(i);

      arma::mat ZW = right_mult_W(Z[i]);
      arma::mat Gi = ZW * Z[i].t();
      double diag_sq_sum = arma::dot(Gi.diag(), Gi.diag());

      double trcov_i_sq =
        - diag_sq_sum / (double)((ni - 2) * (ni - 3))
        + (double)((ni - 1) * (ni - 1)) * trWSWS(i) / (double)(ni * (ni - 3))
        + (double)(ni - 1) * (trWS(i) * trWS(i)) / (double)(ni * (ni - 2) * (ni - 3));

        sigmahat2 += 2.0 * std::pow(B(i), 4.0) * trcov_i_sq / (double)(ni * (ni - 1));
    }

    // cross part
    double cross_sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:cross_sum) schedule(static)
#endif
    for (int i = 0; i < k; ++i) {
      for (int l = i + 1; l < k; ++l) {
        int ni = n_i(i), nl = n_i(l);

        arma::mat WSi = Si[i];
        WSi.each_row() %= O2.t();
        WSi += A * (A.t() * Si[i]);

        arma::mat WSl = Si[l];
        WSl.each_row() %= O2.t();
        WSl += A * (A.t() * Si[l]);

        double trWSiWSl = arma::accu(WSi % WSl.t()); // tr(WSi * WSl) = tr(W Si W Sl)

        cross_sum += 4.0 * (B(i) * B(i)) * (B(l) * B(l)) * trWSiWSl / (double)(ni * nl);
      }
    }
    sigmahat2 += cross_sum;

  } else {
    // HDLSS branch
    for (int i = 0; i < k; ++i) {
      int ni = n_i(i);

      arma::mat ZW = right_mult_W(Z[i]);
      arma::mat Gi = ZW * Z[i].t();

      double diag_sq_sum = arma::dot(Gi.diag(), Gi.diag());
      double trWSi = arma::trace(Gi) / (double)(ni - 1);
      double trGi2 = arma::accu(Gi % Gi);
      double trWSiWSi = trGi2 / (double)((ni - 1) * (ni - 1));

      double trcov_i_sq =
        - diag_sq_sum / (double)((ni - 2) * (ni - 3))
        + (double)((ni - 1) * (ni - 1)) * trWSiWSi / (double)(ni * (ni - 3))
        + (double)(ni - 1) * (trWSi * trWSi) / (double)(ni * (ni - 2) * (ni - 3));

        sigmahat2 += 2.0 * std::pow(B(i), 4.0) * trcov_i_sq / (double)(ni * (ni - 1));
    }

    double cross_sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:cross_sum) schedule(static)
#endif
    for (int i = 0; i < k; ++i) {
      for (int l = i + 1; l < k; ++l) {
        int ni = n_i(i), nl = n_i(l);
        arma::mat ZlW = right_mult_W(Z[l]);
        arma::mat Gil = Z[i] * ZlW.t();
        double trWSiWSl = arma::accu(Gil % Gil) / (double)((ni - 1) * (nl - 1));
        cross_sum += 4.0 * (B(i) * B(i)) * (B(l) * B(l)) * trWSiWSl / (double)(ni * nl);
      }
    }
    sigmahat2 += cross_sum;
  }

  if (!(sigmahat2 > 1e-12)) Rcpp::stop("Estimated variance (sigmahat2) is non-positive or too small.");

  double Zscore = Tn / std::sqrt(sigmahat2);

  arma::vec out(3);
  out(0) = Zscore;
  out(1) = Tn;
  out(2) = sigmahat2;
  return out;
}




// Scale-invariant test (Cao et al., 2024)
// Input Y[i] is n_i x p (rows are samples)
// B: coefficient vector (length k)
// n: group sizes (length k), for interface consistency with other GLHT tests
// Returns: [T, nu_star, nu_hat, trR2_hat, c_old, c_new, c_star]
// [[Rcpp::export]]
arma::vec cao2024_scale_invariant_cpp(const Rcpp::List &Y,
                                      const arma::vec &B,
                                      const arma::vec &n,
                                      int p)
{
  int k = Y.size();
  if ((int)B.n_elem != k) Rcpp::stop("Length of B must equal Y.size().");
  if ((int)n.n_elem != k) Rcpp::stop("Length of n must equal Y.size().");
  if (p <= 0) Rcpp::stop("p must be positive.");

  std::vector<arma::mat> X(k), Z(k);
  std::vector<arma::rowvec> meanRow(k);
  arma::ivec n_i(k);
  int ss = 0;

  // Read data, check dimensions, and cache centered matrices
  for (int i = 0; i < k; ++i) {
    int ni = (int)n(i);
    if (ni < 4) Rcpp::stop("Each group must have n_i >= 4.");

    arma::mat Xi = Rcpp::as<arma::mat>(Y[i]); // n_i x p
    if ((int)Xi.n_cols != p) Rcpp::stop("Each Y[i] must be an n_i x p matrix (cols = p).");
    if ((int)Xi.n_rows != ni) Rcpp::stop("n(i) must match nrow(Y[[i]]).");

    X[i] = Xi;
    n_i(i) = ni;
    ss += ni;

    meanRow[i] = arma::mean(Xi, 0);
    Z[i] = Xi.each_row() - meanRow[i];
  }

  // Yvec = sqrt(ss) * sum_i B_i * mean_i  (p x 1)
  arma::colvec Yvec(p, arma::fill::zeros);
  for (int i = 0; i < k; ++i) Yvec += B(i) * meanRow[i].t();
  Yvec *= std::sqrt((double)ss);

  // diag(S_i): sample variance per feature for group i
  std::vector<arma::vec> diagSi(k);
  for (int i = 0; i < k; ++i) {
    arma::rowvec vrow = arma::sum(arma::square(Z[i]), 0) / (double)(n_i(i) - 1);
    diagSi[i] = vrow.t(); // p x 1
  }

  // a_i = ss * B_i^2 / n_i
  arma::vec a(k, arma::fill::zeros);
  for (int i = 0; i < k; ++i) a(i) = (double)ss * B(i) * B(i) / (double)n_i(i);

  // Dhat_diag = diag( sum_i a_i S_i ) = sum_i a_i * diag(S_i)
  arma::vec Dhat_diag(p, arma::fill::zeros);
  for (int i = 0; i < k; ++i) Dhat_diag += a(i) * diagSi[i];
  for (int j = 0; j < p; ++j) if (Dhat_diag(j) <= 1e-12) Dhat_diag(j) = 1e-12;

  arma::vec invD     = 1.0 / Dhat_diag;
  arma::vec invSqrtD = 1.0 / arma::sqrt(Dhat_diag);

  // T = p^{-1} Y^T D^{-1} Y
  double T = arma::dot(invD, arma::square(Yvec)) / (double)p;

  // U_i = Z_i * D^{-1/2} (column scaling)
  std::vector<arma::mat> U(k);
  for (int i = 0; i < k; ++i) {
    arma::mat Ui = Z[i];
    Ui.each_row() %= invSqrtD.t(); // scale columns
    U[i] = Ui;                     // n_i x p
  }

  // Vcat stacks weighted U_i rows: Vcat is ss x p
  arma::mat Vcat(ss, p, arma::fill::zeros);
  int row0 = 0;
  for (int i = 0; i < k; ++i) {
    int ni = n_i(i);
    double wi = std::sqrt(a(i) / (double)(ni - 1));
    Vcat.rows(row0, row0 + ni - 1) = wi * U[i];
    row0 += ni;
  }

  // lambda_max = lambda_1(Rhat) via smaller matrix
  double lambda_max = 0.0;
  if (p < ss) {
    arma::mat Rhat = Vcat.t() * Vcat;           // p x p
    arma::vec eigR = arma::eig_sym(Rhat);
    lambda_max = eigR.max();
  } else {
    arma::mat Kmat = Vcat * Vcat.t();           // ss x ss
    arma::vec eigK = arma::eig_sym(Kmat);
    lambda_max = eigK.max();
  }
  if (lambda_max < 0.0) lambda_max = 0.0;

  // -----------------------------
  // Estimate tr(R^2): trR2_hat
  // -----------------------------
  arma::vec trRi(k, arma::fill::zeros);
  arma::vec trRi2(k, arma::fill::zeros);

  for (int i = 0; i < k; ++i) {
    int ni = n_i(i);

    // tr(R_i) = ||U_i||_F^2 / (n_i-1)
    double fro2 = arma::accu(U[i] % U[i]);
    trRi(i) = fro2 / (double)(ni - 1);

    // tr(R_i^2) via Gi = U_i U_i^T (n_i x n_i): tr(Gi^2) = sum(Gi^2 entries)
    arma::mat Gi = U[i] * U[i].t(); // n_i x n_i
    double trGi2 = arma::accu(Gi % Gi);
    trRi2(i) = trGi2 / (double)((ni - 1) * (ni - 1));
  }

  double trR2_hat = 0.0;

  // diagonal blocks
  for (int i = 0; i < k; ++i) {
    int ni = n_i(i);
    double coef = (double)(ni - 1) * (double)(ni - 1) / ((double)(ni - 2) * (double)(ni + 1));
    double trRi2_unb = coef * (trRi2(i) - (trRi(i) * trRi(i)) / (double)(ni - 1));
    if (trRi2_unb < 0.0) trRi2_unb = 0.0; // numerical guard
    trR2_hat += (a(i) * a(i)) * trRi2_unb;
  }

  // off-diagonal blocks (parallel)
  double trR2_hat_off = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:trR2_hat_off) schedule(static)
#endif
  for (int i = 0; i < k; ++i) {
    for (int j = i + 1; j < k; ++j) {
      int ni = n_i(i), nj = n_i(j);
      arma::mat Gij = U[i] * U[j].t(); // n_i x n_j
      double fro2 = arma::accu(Gij % Gij);
      double trRij = fro2 / (double)((ni - 1) * (nj - 1));
      trR2_hat_off += 2.0 * a(i) * a(j) * trRij;
    }
  }
  trR2_hat += trR2_hat_off;

  if (!(trR2_hat > 1e-12)) Rcpp::stop("Estimated tr(R^2) is non-positive or too small.");

  // -----------------------------
  // Adjustment coefficients using the same tr(R^2) estimator
  // -----------------------------
  double c_old  = 1.0 + trR2_hat / std::pow((double)p, 1.5);              // c_{n,p}
  double c_new  = 1.0 + lambda_max / std::sqrt((double)p * trR2_hat);     // \bar c_{n,p}
  double c_star = (c_old <= 1.2) ? c_old : c_new;                         // c^*_{n,p}

  double nu_hat  = (double)p * (double)p / trR2_hat;
  double nu_star = nu_hat / c_star;
  if (!(nu_star > 0.0)) Rcpp::stop("nu_star is non-positive.");

  arma::vec out(7);
  out(0) = T;
  out(1) = nu_star;
  out(2) = nu_hat;
  out(3) = trR2_hat;
  out(4) = c_old;
  out(5) = c_new;
  out(6) = c_star;
  return out;
}


