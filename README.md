# geothermsim : Scalable semi-analytical heat transfer simulation of geothermal borehole fields


## What is geothermsim?

geothermsim is an open-source semi-analytical transient heat transfer modeling tool for geothermal borehole fields. The method uses high-order piecewise polynomial approximations of heat extraction rates along line trajectories representing the boreholes and evaluates ground temperature changes from the spatial and temporal superposition of heat source solutions. Quasi-steady-state variations of heat carrier fluid temperatures inside boreholes are evaluated by leveraging the multipole method to build and solve a system of ordinary differential equations.

geothermsim is written in Python and based on [JAX](https://github.com/jax-ml/jax). Its main goal is to allow large-scale, hardware-accelerated, detailed simulations with support for automatic differentiation.

The paper introducing the modeling approach of geothermsim is currently under review. An arXiv link will follow soon.

## Version 0

geothermsim is in its initial development stage. The API is subject to change as features are added. Version 1 will release when the API is stable and documentation and unit tests are implemented.

## Documentation

Documentation is currently in development. Jupyter notebooks that reproduce the results of the introductory paper can be found [here](notebooks/README.md).

## Installation


### Quick start

**Users** - Install the latest release using [pip](https://pip.pypa.io/en/latest/):

```
pip install geothermsim
```

Alternatively, [download the latest release](https://github.com/MassimoCimmino/geothermsim/releases) and run the installation script:

```
pip install .
```

**Developers** - To get the latest version of the code, you can [download the
repository from github](https://github.com/MassimoCimmino/geothermsim) or clone
the project in a local directory using git:

```
git clone https://github.com/MassimoCimmino/geothermsim.git
```

Install geothermsim in development mode (this requires `pip >= 21.1`):
```
pip install --editable .
```

Once geothermsim is copied to a local directory, you can verify that it is
working properly by running the [notebooks](notebooks/README.md).

### Requirements

geothermsim was developed using Python 3.12, but should be compatible with Python 3.10 and 3.11. In addition, the
following packages are needed to run geothermsim and its examples:
- jax (>= 0.6.0)
- matplotlib (>= 3.10.0),
- numpy (>= 2.2.2)
- scipy (>= 1.15.1)
- SecondaryCoolantProps (>= 1.3)

## License

geothermsim is licensed under the terms of the 3-clause BSD-license.
See [geothermsim license](LICENSE.md).
