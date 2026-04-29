# HDNRA 2.1.0

## New features

* Added `WZ2026.GLHTBF.2cNRT()`, an F-type normal-reference test for heteroscedastic high-dimensional general linear hypothesis testing (GLHT) problems.
* `WZ2026.GLHTBF.2cNRT()` supports the grouped-data GLHTBF interface `WZ2026.GLHTBF.2cNRT(Y, G, n, p)`, where `Y` is a list of groupwise data matrices, `G` is the raw full-row-rank contrast matrix, `n` is the vector of group sample sizes, and `p` is the common data dimension.
* The new GLHTBF F-type routine implements a trace-studentized quadratic contrast statistic with Welch--Satterthwaite two-cumulant F-type normal-reference calibration.
* The returned `NRtest` object reports the test statistic, p-value, fitted numerator and denominator degrees of freedom, and the corresponding approximation method.
* The new GLHTBF F-type routine can be used for both omnibus one-way MANOVA-type hypotheses and targeted rank-one contrast hypotheses through the same grouped-data interface.
* Added `CCXH2024.GLHTBF.2cNRT()` for the rank-one scale-invariant GLHTBF normal-reference procedure of Cao et al. (2024).
* Added `LHNB2025.GLHTBF.NABT()` for the rank-one random-integration GLHTBF normal-approximation procedure of Li et al. (2025).

## Documentation and examples

* Updated the GLHTBF documentation to clarify that the software argument `G` is the raw contrast matrix supplied by the user; the normalized contrast matrix and the induced contrast operator are constructed internally.
* Added examples showing how to call `WZ2026.GLHTBF.2cNRT()` for omnibus and rank-one GLHTBF analyses using grouped data.
* Updated the method inventory so that the newly added GLHTBF routines are listed under the correct problem class and calibration family.
* Updated references to the proposed statistic `TNEW` so that its callable routine is consistently recorded as `WZ2026.GLHTBF.2cNRT()`.
* Expanded package references and documentation entries for the newly added GLHTBF procedures.

## Improvements

* Standardized input validation for the new GLHTBF routines, including checks on the groupwise data list `Y`, contrast matrix `G`, sample-size vector `n`, and common dimension `p`.
* Improved consistency of returned approximation fields across GLHTBF normal-reference routines.
* Kept the new F-type implementation inversion-free and compatible with HDLSS settings where the sample covariance matrices may be singular.
* Updated exported functions, manual pages, examples, and package metadata for the new GLHTBF additions.
* Improved consistency between the package description, method inventory, documentation, and simulation workflow.

## Bug fixes

* Fixed minor documentation inconsistencies in GLHTBF examples and method descriptions.
* Fixed possible mismatches between newly exported GLHTBF function names, examples, and test scripts.
* Corrected minor typographical and formatting issues in package documentation.

# HDNRA 2.0.1

* Renamed the function `BS1996.TS.NART` to `BS1996.TS.NABT` for consistency.
* Fixed several typos in the documentation and function names.
* Corrected several typographical errors in the documentation and function names to improve clarity and usability.
* Replaced deprecated `save-always` with `actions/cache@v3` in GitHub Actions workflow.
* Enhanced GitHub Actions performance by implementing caching for R package dependencies, leading to faster build times and improved CI efficiency.

# HDNRA 2.0.0

* The function name has been changed. The title, example, output format of the function have been changed, and some typos have also been corrected.

* Added a helper function to 'HDNRA.cpp' file in order to improve computation speed; updated the code of all functions to facilitate faster computation.

* Added an 'NRtest.object' to output an S3 class 'NRtest' for our package, and also constructed a corresponding print function to output the appropriate format.

* Added a 'zzz.R' file to manage package startup messages and initialization.

# HDNRA 1.0.0

* Initial CRAN submission.
