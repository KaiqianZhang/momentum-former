from torch_geometric.data import Dataset, Data
import pandas as pd
import torch
import numpy as np

# Dataset class for handling futures time series data
class FuturesTimeSeries(Dataset):
    """
    A custom PyTorch Geometric dataset class for loading and processing futures time series data.
    
    Attributes:
    -----------
    window : int
        The number of previous days to consider for each sample.
    start_date : str
        The start date of the time series data to be included.
    end_date : str
        The end date of the time series data to be included.
    datapath : str
        Path to the CSV file containing futures data.
    add_corr : bool
        Whether to include correlation data as features.
    corr_pos : float
        Positive correlation threshold.
    corr_neg : float
        Negative correlation threshold.
    corr_window : int
        Window size for calculating rolling correlations.
    """

    def __init__(
        self,
        window: int = 60,
        start_date: str = '1999-01-04',
        end_date: str = '2023-04-19',
        datapath: str = './input/futures-clean/futures_clean.csv',
        add_corr: bool = True,
        corr_pos: float = 0.3,
        corr_neg: float = -0.3,
        corr_window: int = 252,
    ):
        super().__init__()
        self.corr_pos = corr_pos
        self.corr_neg = corr_neg
        
        # Load the futures data from a CSV file, ensuring proper date formatting
        raw_data = pd.read_csv(datapath, index_col=0)
        raw_data.index = pd.to_datetime(raw_data.index, format="%m/%d/%y")
        raw_data = raw_data.apply(pd.to_numeric)
        
        # Calculate daily returns as percentage change between consecutive days
        self.returns_data = (raw_data / raw_data.shift(1) - 1).dropna()
        
        # Remove zero return days, which typically occur during contract rollovers
        self.returns_data = self.returns_data[self.returns_data.sum(1) != 0].replace(to_replace=0, method='ffill')
        raw_data = raw_data.loc[self.returns_data.index]
       
        # Calculate volatility metrics using Exponential Weighted Moving Average (EWMA)
        std60 = self.returns_data.ewm(span=60, min_periods=60).std()
        std126 = self.returns_data.ewm(span=126, min_periods=126).std()
        std252 = self.returns_data.ewm(span=252, min_periods=252).std()
        
        # Normalize returns by volatility and calculate monthly returns over different periods
        self.returns_normalized = self.returns_data / std60
        self.returns_data_1month = (raw_data / raw_data.shift(21) - 1) / std60 / np.sqrt(21)
        self.returns_data_3month = (raw_data / raw_data.shift(63) - 1) / std60 / np.sqrt(63)
        self.returns_data_6month = (raw_data / raw_data.shift(126) - 1) / std126 / np.sqrt(126)
        self.returns_data_12month = (raw_data / raw_data.shift(252) - 1) / std252 / np.sqrt(252)
        
        # Calculate MACD (Moving Average Convergence Divergence) indicators with different halflives
        self.MACD1 = raw_data.ewm(halflife=8).mean() - raw_data.ewm(halflife=24).mean() 
        self.MACD2 = raw_data.ewm(halflife=16).mean() - raw_data.ewm(halflife=48).mean()
        self.MACD3 = raw_data.ewm(halflife=32).mean() - raw_data.ewm(halflife=96).mean()
        
        # Filter data to ensure all metrics have the same date range
        self.returns_data_12month = self.returns_data_12month.dropna() 
        filter_dates = self.returns_data_12month.index
        self.returns_data = self.returns_data.loc[filter_dates]
        self.returns_normalized = self.returns_normalized.loc[filter_dates]
        self.returns_data_1month = self.returns_data_1month.loc[filter_dates]
        self.returns_data_3month = self.returns_data_3month.loc[filter_dates]
        self.returns_data_6month = self.returns_data_6month.loc[filter_dates]
        self.MACD1 = self.MACD1.loc[filter_dates]
        self.MACD2 = self.MACD2.loc[filter_dates]
        self.MACD3 = self.MACD3.loc[filter_dates]
        
        # Subset the data based on start and end dates provided
        indices = np.nonzero((self.returns_data.index >= start_date) & (self.returns_data.index <= end_date))[0]
        start_index = max(indices[0] - window, 0)
        end_index = indices[-1]
        
        # Ensure correlation calculations have sufficient historical data
        assert (indices[0] - corr_window) > 0
        corr_start = indices[0] - corr_window
        self.returns_corr = self.returns_data.iloc[corr_start:end_index + 1].copy()
        
        # Filter data by the start and end dates
        self.returns_data = self.returns_data.iloc[start_index:end_index + 1]
        self.returns_normalized = self.returns_normalized.iloc[start_index:end_index + 1]
        self.returns_data_1month = self.returns_data_1month.iloc[start_index:end_index + 1]
        self.returns_data_3month = self.returns_data_3month.iloc[start_index:end_index + 1]
        self.returns_data_6month = self.returns_data_6month.iloc[start_index:end_index + 1]
        self.MACD1 = self.MACD1.iloc[start_index:end_index + 1]
        self.MACD2 = self.MACD2.iloc[start_index:end_index + 1]
        self.MACD3 = self.MACD3.iloc[start_index:end_index + 1]
        
        self.window = window
        self.corr_window = corr_window
        self.add_corr = add_corr

    def len(self) -> int:
        """
        Return the total number of samples in the dataset.
        The length is calculated as the total number of data points minus the window size.
        """
        return len(self.returns_data) - self.window

    def get(self, idx: int) -> Data:
        """
        Generate a sample from the dataset at the specified index.
        
        Parameters:
        -----------
        idx : int
            Index of the sample to generate.
        
        Returns:
        --------
        data : torch_geometric.data.Data
            A Data object containing the features and target variable.
        """
        # Initialize the tensor to hold features across multiple windows
        x = torch.zeros(len(self.returns_data.columns), self.window, 8)  # 8: number of different feature sets
        y = torch.tensor(np.array([self.returns_normalized.iloc[idx + self.window].values]), dtype=torch.float32)
        y_actual = torch.tensor(np.array([self.returns_data.iloc[idx + self.window].values]), dtype=torch.float32)
        
        # Fill in the tensor with features
        data = Data(x=x, y=y, y_actual=y_actual)
        data.x[:, :, 0] = torch.tensor(np.transpose(self.returns_normalized.iloc[idx:idx + self.window].values), dtype=torch.float32)
        data.x[:, :, 1] = torch.tensor(np.transpose(self.returns_data_1month.iloc[idx:idx + self.window].values), dtype=torch.float32)
        data.x[:, :, 2] = torch.tensor(np.transpose(self.returns_data_3month.iloc[idx:idx + self.window].values), dtype=torch.float32)
        data.x[:, :, 3] = torch.tensor(np.transpose(self.returns_data_6month.iloc[idx:idx + self.window].values), dtype=torch.float32)
        data.x[:, :, 4] = torch.tensor(np.transpose(self.returns_data_12month.iloc[idx:idx + self.window].values), dtype=torch.float32)
        data.x[:, :, 5] = torch.tensor(np.transpose(self.MACD1.iloc[idx:idx + self.window].values), dtype=torch.float32)
        data.x[:, :, 6] = torch.tensor(np.transpose(self.MACD2.iloc[idx:idx + self.window].values), dtype=torch.float32)
        data.x[:, :, 7] = torch.tensor(np.transpose(self.MACD3.iloc[idx:idx + self.window].values), dtype=torch.float32)
        
        return data
