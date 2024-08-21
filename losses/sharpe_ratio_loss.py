import torch
import pytorch_lightning as pl

# Define a custom loss function based on the Sharpe Ratio
class SharpeRatioloss(pl.LightningModule):
    """
    Custom loss function for optimizing the Sharpe ratio of the portfolio.
    This is used to encourage the model to make predictions that maximize the Sharpe ratio.
    
    Attributes:
    -----------
    eps : float
        Small value to prevent division by zero in loss calculations.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute the Sharpe Ratio loss.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Model predictions for asset returns.
        tgt : torch.Tensor
            Actual asset returns.
        
        Returns:
        --------
        loss : torch.Tensor
            The computed Sharpe Ratio loss.
        """
        # Calculate daily returns based on predictions and actual values
        # Use predictions as positions of the portfolio
        daily_return = (pred * tgt)
        mean_return = daily_return.mean()
        MSR = (daily_return * daily_return).mean()
        
        # Compute Sharpe ratio and return negative to minimize during training
        return -(mean_return * np.sqrt(252) / (torch.sqrt(MSR - mean_return * mean_return)))
