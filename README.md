# Blended Chart Surface Interpolation

## Overview

This project aims to extend Blended Chart Surfaces to sequences with
correspondences.

## Setup

This project is developed using [uv]. Python 3.12 is required. Install the
required dependencies with `uv sync`.

[uv]: https://docs.astral.sh/uv/

Optionally, generate open3d stubs using [pybind11-stubgen], which is included as
a development dependency:

```bash
pybind11-stubgen open3d
```

[pybind11-stubgen]: https://pypi.org/project/pybind11-stubgen/

## Usage

Read the help messages.

```bash
# See a list of all the commands.
bcsi --help
# View the help message for a specific command.
bcsi create-submesh --help
```

Any output files will be placed in `output/`.
