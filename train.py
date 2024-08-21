import gc
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from models.transformer import FuturesTransformer
from datasets.futures_time_series import FuturesTimeSeries
from utils.sharpe_ratio import Sharpe_ratio

# Global variable for versioning model checkpoints
v_num = 0

def train_and_evaluate_model(years, model_class, dataset_class, sharpe_ratio_fn):
    """
    Train and evaluate the model over multiple years, compute and return cumulative return, volatility, and Sharpe ratio.

    Parameters:
    - years: List of years to iterate through for validation.
    - model_class: The class of the model to be trained (e.g., Futures_Transformer).
    - dataset_class: The class used to load the dataset (e.g., Futures_time_series).
    - sharpe_ratio_fn: The function to compute the Sharpe ratio.

    Returns:
    - cumulative_return: Cumulative return over rolling years.
    - overall_volatility: Combined volatility over rolling years.
    - sharpe_ratio: Overall Sharpe ratio.
    """

    results, returns, positions, asset_returns = [], [], [], []
    
    for year in years:
        print(f"Validation year: {year}")
        
        # Define hyperparameters for the model
        window, features = 60, 8
        d_model, nhead, layers, dropout = 8, 2, 1, 0.5
        learning_rate, weight_decay = 0.01, 100
        batch_size, alpha = 512 * 2, 0
        
        # Initialize the model
        model = model_class(
            features=features,
            window=window,
            d_model=d_model,
            nhead=nhead,
            layers=layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            alpha=alpha,
            save=False,
        )

        print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Prepare data loaders for training, validation, and testing
        train_set = dataset_class(window=window, start_date='2001-01-01', end_date=f'{year-1}-12-31')
        drop_last = len(train_set) % batch_size < 60
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=drop_last)

        val_set = dataset_class(window=window, start_date=f'{year}-01-01', end_date=f'{year}-12-31')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        test_set = dataset_class(window=window, start_date='2023-01-01', end_date='2023-04-01') if year == 2022 else dataset_class(window=window, start_date=f'{year+1}-01-01', end_date=f'{year+1}-12-31')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # Clean up GPU memory before training
        gc.collect()
        torch.cuda.empty_cache()

        # Training loop with learning rate decay
        for epoch_limit in [20, 10, 10]:
            print('Current learning rate:', model.hparams.learning_rate)
            trainer = pl.Trainer(log_every_n_steps=1000, accelerator="gpu", max_epochs=epoch_limit, num_sanity_val_steps=0)
            trainer.fit(model, train_loader, val_loader)

            if trainer.interrupted:
                raise RuntimeError("Training was interrupted.")
            
            # Decay the learning rate
            model.hparams.learning_rate *= 0.1

        # Evaluate model performance and calculate Sharpe ratio
        port = sharpe_ratio_fn(model, test_loader)
        results.append(port[0])
        returns.append(port[1])
        positions.append(port[2])
        asset_returns.append(port[3])

        # Clean up GPU memory after training
        gc.collect()
        torch.cuda.empty_cache()

  if __name__ == "__main__":
    cumulative_return, overall_volatility, sharpe_ratio = train_and_evaluate_model(
        years=[2018, 2019, 2020, 2021, 2022],
        model_class=FuturesTransformer,
        dataset_class=FuturesTimeSeries,
        sharpe_ratio_fn=Sharpe_ratio
    )
