# HDNRA_data corneal

This dataset was acquired during a keratoconus study, a collaborative
project involving Ms.Nancy Tripoli and Dr.Kenneth L.Cohen of Department
of Ophthalmology at the University of North Carolina, Chapel Hill. The
fitted feature vectors for the complete corneal surface dataset
collectively into a feature matrix with dimensions of 150 × 2000.

## Usage

``` r
data(corneal)
```

## Format

### 'corneal'

A data frame with 150 observations on the following 4 groups.

- normal group1:

  row 1 to row 43 in total 43 rows of the feature matrix correspond to
  observations from the normal group

- unilateral suspect group2:

  row 44 to row 57 in total 14 rows of the feature matrix correspond to
  observations from the unilateral suspect group

- suspect map group3:

  row 58 to row 78 in total 21 of the feature matrix correspond to
  observations from the suspect map group

- clinical keratoconus group4:

  row 79 to row 150 in total 72 of the feature matrix correspond to
  observations from the clinical keratoconus group

## References

Smaga Ł, Zhang J (2019). “Linear hypothesis testing with functional
data.” *Technometrics*, **61**(1), 99–110.
[doi:10.1080/00401706.2018.1456976](https://doi.org/10.1080/00401706.2018.1456976)
.

## Examples

``` r
library(HDNRA)
data(corneal)
dim(corneal)
#> [1]  150 2000
group1 <- as.matrix(corneal[1:43, ]) ## normal group
dim(group1)
#> [1]   43 2000
group2 <- as.matrix(corneal[44:57, ]) ## unilateral suspect group
dim(group2)
#> [1]   14 2000
group3 <- as.matrix(corneal[58:78, ]) ## suspect map group
dim(group3)
#> [1]   21 2000
group4 <- as.matrix(corneal[79:150, ]) ## clinical keratoconus group
dim(group4)
#> [1]   72 2000
```
