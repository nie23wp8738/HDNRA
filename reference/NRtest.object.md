# S3 Class "NRtest"

The `"NRtest"` objects provide a comprehensive summary of hypothesis
test outcomes, including test statistics, p-values, parameter estimates,
and confidence intervals, if applicable.

## Usage

``` r
NRtest.object(
  statistic,
  p.value,
  method,
  null.value,
  alternative,
  parameter = NULL,
  sample.size = NULL,
  sample.dimension = NULL,
  estimation.method = NULL,
  data.name = NULL,
  ...
)
```

## Arguments

- statistic:

  Numeric scalar containing the value of the test statistic, with a
  `names` attribute indicating the name of the test statistic.

- p.value:

  Numeric scalar containing the p-value for the test.

- method:

  Character string giving the name of the test.

- null.value:

  Character string indicating the null hypothesis.

- alternative:

  Character string indicating the alternative hypothesis.

- parameter:

  Numeric vector containing the estimated approximation parameter(s)
  associated with the approximation method. This vector has a `names`
  attribute describing its element(s).

- sample.size:

  Numeric vector containing the number of observations in each group
  used for the hypothesis test.

- sample.dimension:

  Numeric scalar containing the dimension of the dataset used for the
  hypothesis test.

- estimation.method:

  Character string giving the name of the approximation approach used to
  approximate the null distribution of the test statistic.

- data.name:

  Character string describing the data set used in the hypothesis test.

- ...:

  Additional optional arguments.

## Value

An object of class `"NRtest"` containing both required and optional
components depending on the specifics of the hypothesis test, shown as
follows:

## Details

A class of objects returned by high-dimensional hypothesis testing
functions in the HDNRA package, designed to encapsulate detailed results
from statistical hypothesis tests. These objects are structured
similarly to htest objects in the package EnvStats but are tailored to
the needs of the HDNRA package.

## Required Components

These components must be present in every `"NRtest"` object:

- `statistic`:

  Must e present.

- `p.value`:

  Must e present.

- `null.value`:

  Must e present.

- `alternative`:

  Must e present.

- `method`:

  Must e present.

## Optional Components

These components are included depending on the specifics of the
hypothesis test performed:

- `parameter`:

  May be present.

- `sample.size`:

  May be present.

- `sample.dimension`:

  May be present.

- `estimation.method`:

  May be present.

- `data.name`:

  May be present.

## Methods

The class has the following methods:

- [`print.NRtest`](print.NRtest.md):

  Printing the contents of the NRtest object in a human-readable form.

## Examples

``` r
# Example 1: Using Bai and Saranadasa (1996)'s test (two-sample problem)
NRtest.obj1 <- NRtest.object(
  statistic = c("T[BS]" = 2.208),
  p.value = 0.0136,
  method = "Bai and Saranadasa (1996)'s test",
  data.name = "group1 and group2",
  null.value = c("Difference between two mean vectors is o"),
  alternative = "Difference between two mean vectors is not 0",
  parameter = NULL,
  sample.size = c(n1 = 24, n2 = 26),
  sample.dimension = 20460,
  estimation.method = "Normal approximation"
)
print(NRtest.obj1)
#> 
#> Results of Hypothesis Test
#> --------------------------
#> 
#> Test name:                       Bai and Saranadasa (1996)'s test
#> 
#> Null Hypothesis:                 Difference between two mean vectors is o
#> 
#> Alternative Hypothesis:          Difference between two mean vectors is not 0
#> 
#> Data:                            group1 and group2
#> 
#> Sample Sizes:                    n1 = 24
#>                                  n2 = 26
#> 
#> Sample Dimension:                20460
#> 
#> Test Statistic:                  T[BS] = 2.208
#> 
#> Approximation method to the      Normal approximation
#> null distribution of T[BS]: 
#> 
#> P-value:                         0.0136
#> 

# Example 2: Using Fujikoshi et al. (2004)'s test (GLHT problem)
NRtest.obj2 <- NRtest.object(
  statistic = c("T[FHW]" = 6.4015),
  p.value = 0,
  method = "Fujikoshi et al. (2004)'s test",
  data.name = "Y",
  null.value  = "The general linear hypothesis is true",
  alternative = "The general linear hypothesis is not true",
  parameter = NULL,
  sample.size = c(n1 = 43, n2 = 14, n3 = 21, n4 = 72),
  sample.dimension = 2000,
  estimation.method = "Normal approximation"
)
print(NRtest.obj2)
#> 
#> Results of Hypothesis Test
#> --------------------------
#> 
#> Test name:                       Fujikoshi et al. (2004)'s test
#> 
#> Null Hypothesis:                 The general linear hypothesis is true
#> 
#> Alternative Hypothesis:          The general linear hypothesis is not true
#> 
#> Data:                            Y
#> 
#> Sample Sizes:                    n1 = 43
#>                                  n2 = 14
#>                                  n3 = 21
#>                                  n4 = 72
#> 
#> Sample Dimension:                2000
#> 
#> Test Statistic:                  T[FHW] = 6.4015
#> 
#> Approximation method to the      Normal approximation
#> null distribution of T[FHW]: 
#> 
#> P-value:                         0
#> 
```
