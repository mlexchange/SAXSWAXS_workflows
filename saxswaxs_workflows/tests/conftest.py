import pytest
from prefect.testing.utilities import prefect_test_harness

# code for fixtures (datasets, configs, ..)


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield
