import pathlib

import pytest

from bcsi import io, mesh

TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"
CUBE_PATH = TEST_DATA_DIR / "cube.obj"


@pytest.fixture
def cube_mesh() -> mesh.TriangleMesh:
    return io.read_mesh(CUBE_PATH)
