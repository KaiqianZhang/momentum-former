import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Function to calculate the Sharpe ratio of a portfolio strategy
def Sharpe_ratio(model: pl.LightningModule, loader: DataLoader, return_results: bool = True) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame], None]:
    """
    Calculate the Sharpe ratio of a trading strategy based on model predictions.
    
    Parameters:
    -----------
    model : pl.LightningModule
        The trained model to evaluate.
    loader : DataLoader
        DataLoader for the validation set.
    return_results : bool
        Whether to return the results or just print them.
    
    Returns:
    --------
    Union[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame], None]
        Returns various metrics and results if return_results is True, otherwise None.
    """
    # Run predictions using the trained model
    trainer = pl.Trainer(log_every_n_steps=1000, accelerator="gpu", max_epochs=50)
    prediction = trainer.predict(model, loader)
    
    # Extract predictions and actual returns from the validation set
    predict, actual = zip(*prediction)
    predict = np.concatenate([i for i in predict])
    actual = np.concatenate([i for i in actual])
    l = len(predict)
    
    # Calculate return, standard deviation, and accuracy metrics
    ret = ((predict * actual).mean(-1) + 1).prod() - 1
    std = np.std((predict * actual).mean(-1)) * np.sqrt(l)
    accuracy_asset = np.sum((predict * actual) > 0) / l / 24
    accuracy_port = np.sum((predict * actual).mean(-1) > 0) / l
    
    # Print calculated metrics
    print("Portfolio return: ", ret)
    print("Portfolio std: ", std)
    print("Portfolio Sharpe: ", ret / std)
    print("Asset accuracy: ", accuracy_asset)
    print("Portfolio accuracy: ", accuracy_port)
    
    # Optionally return detailed results
    if return_results:
        return (np.array([ret, std, ret / std, accuracy_asset, accuracy_port]), 
                (predict * val_set.returns_normalized.iloc[-l:].values).mean(-1), 
                predict,
                val_set.returns_normalized.iloc[-l:])
