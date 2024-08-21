import numpy as np

# Trading strategy function to evaluate predictions
def trading_strategy(predict: np.ndarray, actual: np.ndarray, contracts: int = 5) -> float:
    """
    Simulate a simple trading strategy based on model predictions and compute the Sharpe ratio.
    
    Parameters:
    -----------
    predict : np.ndarray
        Array of predicted returns.
    actual : np.ndarray
        Array of actual returns.
    contracts : int
        Number of contracts to long and short.
    
    Returns:
    --------
    sharpe_ratio : float
        The Sharpe ratio of the simulated trading strategy.
    """
    long = []
    short = []
    
    # Calculate long and short positions based on predicted rankings
    for i in range(len(actual)):
        ranks = predict[i].argsort()
        long.append(np.mean(actual[i][ranks[-contracts:]]))
        short.append(np.mean(actual[i][ranks[:contracts]]))
    
    long = np.array(long)
    short = np.array(short)
    portfolio = long - short
    
    # Calculate the Sharpe ratio for the strategy
    return ((long - short + 1).prod() - 1) / (np.std(long - short) * np.sqrt(len(long)))
