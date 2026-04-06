"""Smoke tests for Phase 1 scaffold integrity."""


def test_packages_importable() -> None:
    import algorithms  # noqa: F401
    import env  # noqa: F401
    import experiments  # noqa: F401
    import utils  # noqa: F401


def test_dependencies_importable() -> None:
    import gymnasium  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import seaborn  # noqa: F401
    import tqdm  # noqa: F401
