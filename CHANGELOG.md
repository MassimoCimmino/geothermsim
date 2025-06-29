# History of changes

## Version 0.2.0 (2025-06-29)

### New features

* [Issue 3](https://github.com/MassimoCimmino/geothermsim/issues/3) - Introduced new initialization options for the `Path` class. The trajectory can now be represented as a spline passing through an array of positions. Functions that evaluate *g*-functions can now be jit-compiled and are compatible with automatic differentiation. This adds `interpax` as a dependency of `geothermsim`.
* [Issue 10](https://github.com/MassimoCimmino/geothermsim/issues/10) - Created the `heat_transfer` module. The point heat source solution is moved from the `Path` class to the new module.

### Enhancements

* [Issue 7](https://github.com/MassimoCimmino/geothermsim/issues/7) - Refactored `Simulation` classes to make functions that encapsulate simulations and g-functions evaluation both jittable and differentiable.
* [Issue 9](https://github.com/MassimoCimmino/geothermsim/issues/9) - `GroundHeatExchanger` classes now accept arrays of per-pipe thermal conductivity.
* [Issue 13](https://github.com/MassimoCimmino/geothermsim/issues/13) - Refactored `Borehole` and `Tube` classes to use jitted helper classes and static methods. This decreases the required compilation time when simulating borefields, since now all boreholes rely on the same jitted methods.
* [Issue 14](https://github.com/MassimoCimmino/geothermsim/issues/14) - Moved quadrature methods from the `Basis` class to the `utilities` module.


## Version 0.1.2 (2025-05-06)

### Bug Fixes

* [Issue 1](https://github.com/MassimoCimmino/geothermsim/issues/1) - Fixed an issue where the temperature change evaluated at ground positions was off by a factor `2 * pi * k_s`.
