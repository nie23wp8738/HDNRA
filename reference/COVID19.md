# HDNRA_data COVID19

A COVID19 data set from NCBI with ID GSE152641. The data set profiled
peripheral blood from 24 healthy controls and 62 prospectively enrolled
patients with community-acquired lower respiratory tract infection by
SARS-COV-2 within the first 24 hours of hospital admission using RNA
sequencing.

## Usage

``` r
data(COVID19)
```

## Format

### 'COVID19'

A data frame with 86 observations on the following 2 groups.

- healthy group1:

  row 2 to row 19, and row 82 to 87, in total 24 healthy controls

- patients group2:

  row 20 to 81, in total 62 prospectively enrolled patients

## References

Thair SA, He YD, Hasin-Brumshtein Y, Sakaram S, Pandya R, Toh J, Rawling
D, Remmel M, Coyle S, Dalekos GN, others (2021). “Transcriptomic
similarities and differences in host response between SARS-CoV-2 and
other viral infections.” *Iscience*, **24**(1).
[doi:10.1016/j.isci.2020.101947](https://doi.org/10.1016/j.isci.2020.101947)
.

## Examples

``` r
library(HDNRA)
data(COVID19)
dim(COVID19)
#> [1]    87 20460
group1 <- as.matrix(COVID19[c(2:19, 82:87), ]) ## healthy group
dim(group1)
#> [1]    24 20460
group2 <- as.matrix(COVID19[-c(1:19, 82:87), ]) ## COVID-19 patients
dim(group2)
#> [1]    62 20460
```
