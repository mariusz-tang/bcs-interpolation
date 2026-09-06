# Blended Chart Surface Interpolation

This is the code for my master's degree final project, which aims to extend
[Blended Chart Surfaces](https://doi.org/10.48550/arXiv.2606.18069)
to sequences with correspondences.

**Note:** I have tried to keep the code clean and follow good practices, but,
naturally, this became harder as the deadline approached. The CLI code is
particularly bad, and there are quite a few flags which don't do anything.
I do not intend on cleaning up or making significant changes after submission,
although I may revisit the subject matter in future projects.

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
