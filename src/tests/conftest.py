# project root fixtures_dataset.py
import pytest


pytest_plugins = [
    "tests.unit.twitter.fixtures_twitter",
    "tests.unit.dataset.fixtures_dataset",
    "tests.unit.plotting.fixtures_plotting",
    "tests.unit.utils.fixtures_utils",
    "tests.unit.preprocess.fixtures_preprocess",
    "tests.unit.training.fixtures_training",
]


@pytest.fixture(scope="module")
def vcr_config():
    return {
        # Replace the Authorization request header with "DUMMY" in cassettes
        "filter_headers": [("Authorization", "DUMMY")],
    }


def pytest_addoption(parser):
    parser.addoption("--skip_optional", action="store_true", default=False, help="skip running optional tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "optional: mark test as optional")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip_optional"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_optional = pytest.mark.skip(reason="use --skip_optional option to skip")
    for item in items:
        if "optional" in item.keywords:
            item.add_marker(skip_optional)
