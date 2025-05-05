# Comparison of AMD to `cs_amd`
## Davis, Chapter 7 "Fill-Reducing Orderings"


### Features of AMD
* AMD finds a symmetric ordering of a matrix on either a symmetric `A`,
or `A + A^T` (`cs::AMDOrder::APlusAT`).
* Diagonal entries are ignored.
* Dense rows are removed.
* Aggressive absorption is optional (on by default).
* AMD takes input parameters to control aggressive absorption and the detection
  of dense rows/columns.
* Statistics are provided on output.


### Features of `cs_amd`
* Can find ordering on `A + A^T` or `A^T A` (with or without dense rows).
* Aggressive absorption is not optional.
* No input parameters for control.
* No statistics are provided.


* both follow minimum degree ordering by a postordering
