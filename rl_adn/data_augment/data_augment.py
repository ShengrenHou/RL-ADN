import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq
from copulas.multivariate import GaussianMultivariate
import re
from datetime import datetime, timedelta
from rl_adn.data_manager.data_manager import  GeneralPowerDataManager
from multicopula import EllipticalCopula

class ActivePowerDataManager(GeneralPowerDataManager):
    """
    A subclass of GeneralPowerDataManager that adds a method for retrieving
    active power data specifically.
    """

    def __init__(self, datapath: str) -> None:
        """
        Initialize the ActivePowerDataManager object with the path to the data file.
        """
        if not datapath:
            raise ValueError("Please input the correct datapath")

        self.df = pd.read_csv(datapath, index_col='date_time')
        self.df.index = pd.to_datetime(self.df.index)

        # Assuming the data interval is consistent throughout the dataset
        self.time_interval = int((self.df.index[1] - self.df.index[0]).seconds / 60)
        print(f"Data time interval: {self.time_interval} minutes")

    def get_active_power_data(self) -> np.ndarray:
        """
        Retrieve and preprocess active power data from the dataset.
        """
        self.df.interpolate(method='linear', inplace=True)
        self.df['day'] = self.df.index.date
        self.df['time'] = self.df.index.time

        count_per_day = self.df.groupby('day').size()
        expected_time_steps = 24 * 60 / self.time_interval
        days_with_extra_steps = count_per_day[count_per_day > expected_time_steps]

        if not days_with_extra_steps.empty:
            self.df = self.df[~self.df['day'].isin(days_with_extra_steps.index)]

        active_power_columns = [col for col in self.df.columns if re.fullmatch(r'active_power(_\w+)?', col)]
        active_power_df = self.df[active_power_columns].copy()
        active_power_df['day'] = self.df['day']
        active_power_df['time'] = self.df['time']

        reshaped_active_power_df = active_power_df.set_index(['day', 'time']).stack().reset_index().rename(
            columns={'level_2': 'node', 0: 'value'})

        grouped_active_power_df = reshaped_active_power_df.groupby('time')['value'].apply(list).reset_index()

        reshaped_df = pd.DataFrame(grouped_active_power_df['value'].tolist(), index=grouped_active_power_df['time'])

        active_power_array = reshaped_df.to_numpy().T
        active_power_array = active_power_array[~np.isnan(active_power_array).any(axis=1)]

        return active_power_array


class TimeSeriesDataAugmentor:
    def __init__(self, data_manager, augmentation_model_name='GMC'):
        """
        Initialize the data augmentor with a data manager instance and the selected augmentation model.
        Additional parameters can be set here if required.
        """
        self.data_manager = data_manager
        self.augmentation_model_name = augmentation_model_name
        self.augmentation_model = None

        self._create_augmentation_model()  # Create the augmentation model upon initialization

    def _create_augmentation_model(self):
        """
        Private method to create the augmentation model based on the chosen method, e.g., GMC.
        GMC: Data augmentation using Gaussian Mixture Copulas
        GMM: Data augmentaiton using Gaussian Mixture models
        TC: Data augmentaiton using T Copulas
        """
        if self.augmentation_model_name == 'GMC':
            # Extract data from the data manager
            active_power_array = self.data_manager.get_active_power_data()

            # Determine the best number of components for each GMM model, for one day now it is 96 time steps
            self.n_models = int(24.0 * 60.0 / self.data_manager.time_interval)

            # print(f'data manager time interval {self.data_manager.time_interval}')
            best_components = [self._bic_value(active_power_array[:, i].reshape(-1, 1), 20) for i in
                               range(self.n_models)]

            # Fit the GMM models
            self.gmm_models = [GaussianMixture(n_components=bc).fit(active_power_array[:, i].reshape(-1, 1))
                               for i, bc in enumerate(best_components)]

            # Transform the data to standard format for copula fitting
            std_input_data = np.empty((active_power_array.shape[0], self.n_models))
            for i in range(self.n_models):
                std_input_data[:, i] = np.array(
                    [self._gmm_cdf(self.gmm_models[i], x) for x in active_power_array[:, i]]).reshape(1, -1)
            self.copula = GaussianMultivariate()
            self.copula.fit(std_input_data)

            # Assign the copula as the augmentation model
            self.augmentation_model = self.copula

        if self.augmentation_model_name == 'GMM':
            active_power_array = self.data_manager.get_active_power_data()

            # Determine the best number of components for each GMM model, for one day now it is 96 time steps
            self.n_models = int(24.0 * 60.0 / self.data_manager.time_interval)

            # print(f'data manager time interval {self.data_manager.time_interval}')
            best_components = [self._bic_value(active_power_array[:, i].reshape(-1, 1), 20) for i in
                               range(self.n_models)]

            # Fit the GMM models
            self.gmm_models = [GaussianMixture(n_components=bc).fit(active_power_array[:, i].reshape(-1, 1))
                               for i, bc in enumerate(best_components)]

            self.augmentation_model = self.gmm_models

        if self.augmentation_model_name == 'TC':
            active_power_array = self.data_manager.get_active_power_data()
            self.n_models = int(24.0 * 60.0 / self.data_manager.time_interval)

            self.tc_model = EllipticalCopula(active_power_array.T)
            self.tc_model.fit()
            self.augmentation_model = self.tc_model

    def _gmm_cdf(self, gmm, x):
        """
        Convert CDF to pseudo-observations in the interval [0, 1].
        """
        cdf = 0
        for n in range(gmm.n_components):
            cdf += gmm.weights_[n] * norm.cdf(x, gmm.means_[n, 0], np.sqrt(gmm.covariances_[n, 0]))
        return cdf

    def _inverse_gmm_cdf(self, gmm, percentile):
        """
        Find the inverse of the CDF for a given percentile using a GMM model.
        """

        def f(x):
            return self._gmm_cdf(gmm, x) - percentile

        return brentq(f, -3000, 3000)

    def _bic_value(self, data, n_components_range):
        """
        Compute BIC value for different numbers of components to determine the best model.
        """
        bic_values = []
        for n_components in range(1, n_components_range):
            gmm = GaussianMixture(n_components=n_components).fit(data)
            bic_values.append(gmm.bic(data))
        best_component = np.argmin(bic_values) + 1
        return best_component

    def check_data_format(self):
        """
        Verify that the data matches the expected format for augmentation.
        """
        # Implementation details to verify data format

    def augment_data(self, num_nodes, num_days, start_date):
        """
        Perform data augmentation using the specified model and parameters.
        """
        if self.augmentation_model_name == 'GMC':
            num_samples = num_days * num_nodes
            print('The number of samples is', num_samples)

            generated_pesudo_obs = np.empty((0, self.n_models))
            count = 0
            while True:
                # print('days is generated in sample', count)
                gen_one_sample = np.array(self.copula.sample(1))
                if gen_one_sample.min() > 0 and gen_one_sample.max() < 1:
                    count += 1
                    generated_pesudo_obs = np.vstack((gen_one_sample, generated_pesudo_obs))
                    if count == num_samples:
                        break
            # print(' the pesudo data is now sampled and next process is to transfer it to the realistic data')
            tran_samples = np.empty((generated_pesudo_obs.shape[0], generated_pesudo_obs.shape[1]))
            for i in range(self.n_models):
                tran_samples[:, i] = np.array(
                    [self._inverse_gmm_cdf(self.gmm_models[i], u) for u in generated_pesudo_obs[:, i]])
                print(f'the {i} model columns now is calculated')
            tran_samples = tran_samples.flatten()

        if self.augmentation_model_name == 'GMM':
            num_samples = num_days * num_nodes
            print('The number of samples is', num_samples)

            # generating the data
            gmm_samples = np.empty((num_samples, self.n_models))
            for i in range(self.n_models):
                gmm_samples[:, i] = self.gmm_models[i].sample(num_samples)[0].reshape(-1)

            tran_samples = gmm_samples.flatten()

        if self.augmentation_model_name == 'TC':
            num_samples = num_days * num_nodes
            print('The number of samples is', num_samples)

            # generating the data
            TC_samples = np.empty((0, self.n_models))
            count = 0
            while True:
                gen_one_sample = np.array(self.tc_model.sample(1)).reshape(1, -1)

                # cancel inf
                if np.isinf(gen_one_sample).any() == False:
                    count += 1
                    # print(count,num_samples)
                    TC_samples = np.vstack((gen_one_sample, TC_samples))
                    if count == num_samples:
                        break
            tran_samples = TC_samples.flatten()

        # Initialize lists to hold timestamps and node indices
        timestamps = []
        node_index = []

        # Populate timestamps and node_index for each day and each node

        for day in range(num_days):
            for node in range(1, num_nodes + 1):
                time_step = timedelta(minutes=self.data_manager.time_interval)
                timestamps.extend([start_date + timedelta(days=day) + i * time_step for i in range(self.n_models)])
                node_index.extend([f'active_power_node_{node}' for _ in range(self.n_models)])

        # Create DataFrame
        synthetic_data_df = pd.DataFrame({
            'date_time': timestamps,
            'node': node_index,
            'value': tran_samples
        })

        # Pivot the DataFrame to get it into the desired format
        augmented_df = synthetic_data_df.pivot(index='date_time', columns='node', values='value').reset_index()
        # Reorder the columns based on we need
        active_power_cols = self.sort_columns(augmented_df.columns, r'active_power(_\w+)?')
        reactive_power_cols = self.sort_columns(augmented_df.columns, r'reactive_power(_\w+)?')
        renewable_active_power_cols = self.sort_columns(augmented_df.columns, r'renewable_active_power(_\w+)?')
        renewable_reactive_power_cols = self.sort_columns(augmented_df.columns, r'renewable_reactive_power(_\w+)?')
        price_cols = self.sort_columns(augmented_df.columns, r'price(_\w+)?')
        # Combine columns in the specified order
        ordered_columns = (
                ['date_time'] +
                active_power_cols +
                reactive_power_cols +
                renewable_active_power_cols +
                renewable_reactive_power_cols +
                price_cols
        )
        ordered_augmented_df = augmented_df[ordered_columns]

        return ordered_augmented_df

    def save_augmented_data(self, augmented_df, file_name):
        augmented_df.to_csv(file_name, index=False)
        print('The data file is stored:', file_name)

    def sort_columns(self, columns, pattern):
        def sort_key(col_name):
            parts = col_name.split('_')
            if parts[-1].isdigit():
                return int(parts[-1])
            return 0  # Default sort value for non-numeric endings

        filtered_cols = [col for col in columns if re.fullmatch(pattern, col)]
        return sorted(filtered_cols, key=sort_key)


if __name__ == "__main__":
    input_data_file = 'test_original_data.csv'  # Replace with your actual file path
    augmentation_model_name = 'GMM'
    num_nodes = 3  # For examples, if you have 34 nodes
    num_days = 3  # Assuming you want to generate data for a full year

    # Initialize the data manager with the input CSV file
    data_manager = ActivePowerDataManager(input_data_file)

    # Initialize the TimeSeriesDataAugmentor with the data manager and model name
    augmentor = TimeSeriesDataAugmentor(data_manager, augmentation_model_name)

    # Generate augmented data
    augmented_df = augmentor.augment_data(num_nodes, num_days,start_date=datetime(2021, 1, 1, 0, 0))

    # Define the file name where to save the augmented data

    # Save the augmented data to a CSV file
    augmentor.save_augmented_data(augmented_df, 'test_generated_data.csv')

    print("Data augmentation completed and saved to file.")