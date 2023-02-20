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
