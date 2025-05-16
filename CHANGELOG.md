# History of changes

## Version 0.2.0 [in development]

### New features

* [Issue 3](https://github.com/MassimoCimmino/geothermsim/issues/3) - Introduced new initialization options for the `Path` class. The trajectory can now be represented as a spline passing through an array of positions. Functions that evaluate *g*-functions can now be jit-compiled and are compatible with automatic differentiation. This also removes `numpy` as a dependency of `geothermsim` but adds `interpax`.


## Version 0.1.2 (2025-05-06)

### Bug Fixes

* [Issue 1](https://github.com/MassimoCimmino/geothermsim/issues/1) - Fixed an issue where the temperature change evaluated at ground positions was off by a factor `2 * pi * k_s`.
