"""
This module contains the DataManager class for managing and preprocessing
time-series data related to power systems. It includes functionalities for
data loading, cleaning, and basic manipulations.
"""




from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from numpy import array
import random
import re


class GeneralPowerDataManager:
    """
    A class to manage and preprocess time series data for power systems.

    Attributes:
        df (pd.DataFrame): The original data.
        data_array (np.ndarray): Array representation of the data.
        active_power_cols (List[str]): List of columns related to active power.
        reactive_power_cols (List[str]): List of columns related to reactive power.
        renewable_active_power_cols (List[str]): List of columns related to renewable active power.
        renewable_reactive_power_cols (List[str]): List of columns related to renewable reactive power.
        price_col (List[str]): List of columns related to price.
        train_dates (List[Tuple[int, int, int]]): List of training dates.
        test_dates (List[Tuple[int, int, int]]): List of testing dates.
        time_interval (int): Time interval of the data in minutes.
    """

    def __init__(self, datapath: str) -> None:
        """
        Initialize the GeneralPowerDataManager object.

        Parameters:
            datapath (str): Path to the CSV file containing the data.
        """
        if datapath is None:
            raise ValueError("Please input the correct datapath")

        data = pd.read_csv(datapath)

        # Check if 'date_time' column exists
        if 'date_time' in data.columns:
            data.set_index('date_time', inplace=True)
        else:
            first_col = data.columns[0]
            data.set_index(first_col, inplace=True)

        data.index = pd.to_datetime(data.index)

        # Print data scale and initialize time interval
        min_date = data.index.min()
        max_date = data.index.max()
        print(f"Data scale: from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        self.time_interval = int((data.index[1] - data.index[0]).seconds / 60)
        print(f"Data time interval: {self.time_interval} minutes")

        # Initialize other attributes
        self.df = data
        self.data_array = data.values

        self.active_power_cols = [col for col in self.df.columns if re.fullmatch(r'active_power(_\w+)?', col)]
        self.reactive_power_cols = [col for col in self.df.columns if re.fullmatch(r'reactive_power(_\w+)?', col)]
        self.renewable_active_power_cols = [col for col in self.df.columns if re.fullmatch(r'renewable_active_power(_\w+)?', col)]
        self.renewable_reactive_power_cols = [col for col in self.df.columns if re.fullmatch(r'renewable_reactive_power(_\w+)?', col)]
        self.price_col = [col for col in self.df.columns if re.fullmatch(r'price(_\w+)?', col)]
        # Display dataset information
        print(f"Dataset loaded from {datapath}")
        print(f"Dataset dimensions: {self.df.shape}")
        print(f"Dataset contains the following types of data:")
        print(
            f"Active power columns: {self.active_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.active_power_cols]})")
        print(
            f"Reactive power columns: {self.reactive_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.reactive_power_cols]})")
        print(
            f"Renewable active power columns: {self.renewable_active_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.renewable_active_power_cols]})")
        print(
            f"Renewable reactive power columns: {self.renewable_reactive_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.renewable_reactive_power_cols]})")
        print(f"Price columns: {self.price_col} (Indices: {[self.df.columns.get_loc(col) for col in self.price_col]})")
        # Calculate max and min for each type of power
        self.active_power_max = self.df[self.active_power_cols].max().max()
        self.active_power_min = self.df[self.active_power_cols].min().min()

        self.reactive_power_max = self.df[self.reactive_power_cols].max().max() if self.reactive_power_cols else None
        self.reactive_power_min = self.df[self.reactive_power_cols].min().min() if self.reactive_power_cols else None

        self.renewable_active_power_max = self.df[
            self.renewable_active_power_cols].max().max() if self.renewable_active_power_cols else None
        self.renewable_active_power_min = self.df[
            self.renewable_active_power_cols].min().min() if self.renewable_active_power_cols else None

        self.renewable_reactive_power_max = self.df[
            self.renewable_reactive_power_cols].max().max() if self.renewable_reactive_power_cols else None
        self.renewable_reactive_power_min = self.df[
            self.renewable_reactive_power_cols].min().min() if self.renewable_reactive_power_cols else None
        self.price_min = self.df[self.price_col].min().values[0] if self.price_col else None
        self.price_max = self.df[self.price_col].max().values[0] if self.price_col else None
        # split the train and test dates
        self.train_dates = []
        self.test_dates = []
        self.split_data_set()
        self._replace_nan()
        self._check_for_nan()


    def _replace_nan(self) -> None:
        """
        Replace NaN values in the data with interpolated values or the average of the surrounding values.
        """
        self.df.interpolate(inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)

    def _check_for_nan(self) -> None:
        """
        Check if any of the arrays contain NaN values and raise an error if they do
        """
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("Data still contains NaN values after preprocessing")

    def select_timeslot_data(self, year: int, month: int, day: int, timeslot: int) -> np.ndarray:
        """
               Select data for a specific timeslot on a specific day.

               Parameters:
                   year (int): The year of the date.
                   month (int): The month of the date.
                   day (int): The day of the date.
                   timeslot (int): The timeslot index.

               Returns:
                   np.ndarray: The data for the specified timeslot.
               """
        dt = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0, second=0, tz='UTC') + pd.Timedelta(
            minutes=self.time_interval * timeslot)
        row = self.df.loc[dt]
        return row.values

    def select_day_data(self, year: int, month: int, day: int) -> np.ndarray:
        """
        Select data for a specific day.

        Parameters:
            year (int): The year of the date.
            month (int): The month of the date.
            day (int): The day of the date.

        Returns:
            np.ndarray: The data for the specified day.
        """
        start_dt = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0, second=0, tz='UTC')
        end_dt = start_dt + pd.Timedelta(days=1)
        day_data = self.df.loc[start_dt:end_dt - pd.Timedelta(minutes=1), :]
        return day_data.values

    def list_dates(self) -> List[Tuple[int, int, int]]:
        """
               List all available dates in the data.

               Returns:
                   List[Tuple[int, int, int]]: A list of available dates as (year, month, day).
               """
        dates = self.df.index.strftime('%Y-%m-%d').unique()
        year_month_day = [(int(date[:4]), int(date[5:7]), int(date[8:10])) for date in dates]
        return year_month_day

    def random_date(self) -> Tuple[int, int, int]:
        """
                Randomly select a date from the available dates in the data.

                Returns:
                    Tuple[int, int, int]: The year, month, and day of the selected date.
                """
        dates = self.list_dates()
        year, month, day = random.choice(dates)
        return year, month, day
    def split_data_set(self):
        """
        Split the data into training and testing sets based on the date.

        The first three weeks of each month are used for training and the last week for testing.
        """
        all_dates = self.list_dates()
        all_dates.sort(key=lambda x: (x[0], x[1], x[2]))  # Sort dates

        train_dates = []
        test_dates = []

        current_month = all_dates[0][1]
        current_year = all_dates[0][0]
        monthly_dates = []

        for date in all_dates:
            year, month, day = date
            if month != current_month or year != current_year:
                # Sort monthly dates and split into train and test
                monthly_dates.sort()
                train_len = int(len(monthly_dates) * (3 / 4))  # First three weeks for training
                train_dates += monthly_dates[:train_len]
                test_dates += monthly_dates[train_len:]

                # Reset for the new month
                monthly_dates = []
                current_month = month
                current_year = year

            monthly_dates.append(date)

        # Handle the last month
        if len(monthly_dates) > 0:
            monthly_dates.sort()
            train_len = int(len(monthly_dates) * (3 / 4))
            train_dates += monthly_dates[:train_len]
            test_dates += monthly_dates[train_len:]

        self.train_dates = train_dates
        self.test_dates = test_dates


