# project root fixtures_dataset.py
import pytest

pytest_plugins = [
    "unit.dataset.fixtures_dataset",
    "unit.twitter.fixtures_twitter",
    "unit.plotting.fixtures_plotting",
    "unit.utils.fixtures_utils",
    "unit.preprocess.fixtures_preprocess",
    "unit.training.fixtures_training",
]


def pytest_addoption(parser):
    parser.addoption("--skip_optional", action="store_true", default=False, help="skip running optional tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "optional: mark test as optional")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip_optional"):
        # --skip_optional given in cli: skip optional tests
        return
    skip_optional = pytest.mark.skip(reason="use --skip_optional option to skip")
    for item in items:
        if "optional" in item.keywords:
            item.add_marker(skip_optional)
