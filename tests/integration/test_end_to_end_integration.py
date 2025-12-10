import pytest

from firm_ce.model import Model


@pytest.mark.slow
def test_end_to_end_integration():
    model = Model(
        config_directory="tests/inputs/test_1hr_config_data_diversify/config",
        data_directory="tests/inputs/test_1hr_config_data_diversify/data",
        logging_flag=False,
    )
    model.solve()
