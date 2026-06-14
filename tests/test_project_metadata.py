from importlib.metadata import version

import conn2res


def test_conn2res_version_matches_public_distribution():
    assert conn2res.__version__ == version("connectome-reservoir-vkr")
