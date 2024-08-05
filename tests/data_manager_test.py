from rl_adn.data_manager.data_manager import GeneralPowerDataManager
import pandas as pd
import numpy as np
import os


def test_GeneralPowerDataManager():
    # Generate sample data
    sample_data = {
        'date_time': pd.date_range(start='2021-01-01', periods=24 * 30, freq='H'),
        'active_power_node_1': np.random.rand(24 * 30),
        'active_power_node_2': np.random.rand(24 * 30),
        'reactive_power_node_1': np.random.rand(24 * 30),
        'price_node_1': np.random.rand(24 * 30)
    }
    df = pd.DataFrame(sample_data)
    datapath = 'sample_data.csv'
    df.to_csv(datapath, index=False)

    # Test initialization
    data_manager = GeneralPowerDataManager(datapath)
    assert isinstance(data_manager.df, pd.DataFrame), "DataFrame not initialized"
    assert len(data_manager.active_power_cols) == 2, "Active power columns not detected correctly"
    assert len(data_manager.reactive_power_cols) == 1, "Reactive power columns not detected correctly"
    assert len(data_manager.price_col) == 1, "Price columns not detected correctly"

    # Test select_timeslot_data
    data = data_manager.select_timeslot_data(2021, 1, 1, 0)
    assert len(data) == 4, "Timeslot data not fetched correctly"

    # Test select_day_data
    day_data = data_manager.select_day_data(2021, 1, 1)
    assert day_data.shape[0] == 24, "Day data not fetched correctly"

    # Test list_dates
    dates = data_manager.list_dates()
    assert len(dates) == 30, "List dates not working correctly"

    # Test random_date
    date = data_manager.random_date()
    assert isinstance(date, tuple) and len(date) == 3, "Random date function not working correctly"

    # Test split_data_set
    train_dates = data_manager.train_dates
    test_dates = data_manager.test_dates
    assert len(train_dates) == 22, "Training dates not split correctly"
    assert len(test_dates) == 8, "Testing dates not split correctly"

    # Cleanup
    os.remove(datapath)
test_GeneralPowerDataManager()
