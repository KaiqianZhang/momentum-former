import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.utils import unbatch
from models.transformer_layer import TransformerLayer2
from losses.sharpe_ratio_loss import SharpeRatioloss
from utils.trading_strategy import trading_strategy
from utils.weight_decay import add_weight_decay
from train import v_num

# Transformer-based model for futures prediction
class FuturesTransformer(pl.LightningModule):
    """
    A PyTorch Lightning module implementing a transformer-based model for futures prediction.
    
    Attributes:
    -----------
    features : int
        Number of input features per time step.
    window : int
        Length of the input time window.
    d_model : int
        Dimensionality of the transformer model.
    nhead : int
        Number of attention heads.
    layers : int
        Number of transformer layers in the model.
    assets : int
        Number of assets being predicted.
    dropout : float
        Dropout rate for regularization.
    model_name : str
        Name of the model, used for saving checkpoints.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) applied to the optimizer.
    alpha : float
        Coefficient for regularization or custom loss scaling.
    layer_norm_eps : float
        Epsilon value for layer normalization.
    save : bool
        Whether to save model checkpoints after each epoch.
    print_training : bool
        Whether to print training progress after each epoch.
    """
    def __init__(self,
                 features: int,
                 window: int = 60,
                 d_model: int = 16, 
                 nhead: int = 4, 
                 layers: int = 2,
                 assets: int = 24, 
                 dropout: float = 0.1,
                 model_name: str = "Futures_Transformer",
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 alpha: float = 1,
                 layer_norm_eps: float = 1e-5,
                 save: bool = True,
                 print_training: bool = True,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.assets = assets
        self.window = window
        self.features = features
        
        # LSTM encoder to extract temporal features from the input
        self.LSTM_encoder = nn.LSTM(features, d_model, num_layers=1, batch_first=True, bias=False)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Stack of transformer layers to model interactions between assets
        self.layers = nn.ModuleList(
            [TransformerLayer2(d_model, nhead, d_model, batch_first=True, dropout=dropout, layer_norm_eps=layer_norm_eps)
             for _ in range(layers)]
        )
        
        # Second stack of transformer layers (can be used for further refinement)
        self.layers2 = nn.ModuleList(
            [TransformerLayer2(d_model, nhead, d_model, batch_first=True, dropout=dropout, layer_norm_eps=layer_norm_eps)
             for _ in range(layers)]
        )
        
        # Output layers to predict returns for each asset
        self.prediction_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(d_model, 1),  # Predict ranking scores of each contract
                nn.Tanh()
            ) for _ in range(assets)]
        )
        
        # Loss function based on Sharpe Ratio
        self.loss_direction = SharpeRatioloss()
    
    def forward(self, batch: Data) -> torch.Tensor:
        """
        Forward pass through the entire model, from input features to predictions.
        
        Parameters:
        -----------
        batch : torch_geometric.data.Data
            A batch of input data.
        
        Returns:
        --------
        torch.Tensor
            Predictions for each asset in the batch.
        """
        x = batch.x
        
        # Apply LSTM encoder
        encoder = self.LSTM_encoder(x)[0]
        encoder = self.norm1(encoder)
        
        # Pass through the first stack of transformer layers
        for mod in self.layers:
            encoder = mod(encoder)
        
        # Pool the encoded features across the time dimension
        encoder = encoder.mean(-2)
        
        # Apply layer normalization
        encoder = torch.stack(unbatch(encoder, batch.batch)) 
        encoder = self.norm2(encoder) 
        
        # Pass through the second stack of transformer layers
        for mod in self.layers2:
            encoder = mod(encoder)
        
        # Predict returns for each asset
        predictions = torch.concat([self.prediction_layers[i](encoder[:, i, :]) for i in range(self.assets)], dim=1)
        
        return predictions
    
    def training_step(self, batch: Data, batch_idx: int) -> dict:
        """
        Perform a single training step, including forward pass and loss calculation.
        
        Parameters:
        -----------
        batch : torch_geometric.data.Data
            A batch of input data.
        batch_idx : int
            Index of the batch (used for logging or debugging).
        
        Returns:
        --------
        dict
            A dictionary containing the loss and other relevant metrics.
        """
        pred = self.forward(batch)
        target = batch.y 
        loss = self.loss_direction(pred, target)
        return {"loss": loss,       
                'pred': pred,
                'target': batch.y_actual}

    def training_epoch_end(self, training_step_outputs: List[dict]):
        """
        Called at the end of the training epoch to log metrics and save checkpoints.
        
        Parameters:
        -----------
        training_step_outputs : List[dict]
            List of outputs from each training step in the epoch.
        """
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/entire_batch", avg_loss, sync_dist=True, prog_bar=True)
        
        # Concatenate predictions and actual values across all batches
        predict = torch.concat([x["pred"] for x in training_step_outputs]).detach().cpu().numpy()
        actual = torch.concat([x["target"] for x in training_step_outputs]).detach().cpu().numpy()
        
        # Calculate Sharpe ratio for the entire epoch
        SR = trading_strategy(predict, actual, contracts=5)
        global v_num
        
        # Print and save model if required
        if self.hparams.print_training:
            print("Loss over entire batch: ", avg_loss.item(), f" Version: {v_num}", f" Sharpe: {SR}")
        if self.hparams.save:
            torch.save(self.state_dict(), f'version{v_num}.pt')
        v_num += 1
        
    def validation_step(self, batch: Data, batch_idx: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Perform a single validation step, including forward pass and loss calculation.
        
        Parameters:
        -----------
        batch : torch_geometric.data.Data
            A batch of input data.
        batch_idx : int
            Index of the batch (used for logging or debugging).
        
        Returns:
        --------
        Tuple[float, np.ndarray, np.ndarray]
            A tuple containing the loss, predictions, and actual values for the batch.
        """
        pred = self.forward(batch)
        target = batch.y
        loss = self.loss_direction(pred, target)
        
        return (loss.item(), pred.detach().cpu().numpy(), batch.y.detach().cpu().numpy())
    
    def validation_epoch_end(self, outs: List[Tuple[float, np.ndarray, np.ndarray]]):
        """
        Called at the end of the validation epoch to log metrics and evaluate the model.
        
        Parameters:
        -----------
        outs : List[Tuple[float, np.ndarray, np.ndarray]]
            List of outputs from each validation step in the epoch.
        """
        global v_num
        
        # Extract and concatenate losses, predictions, and actual values
        loss, predict, actual = zip(*outs)
        loss = np.mean(loss)
        predict = np.concatenate([i for i in predict])
        actual = np.concatenate([i for i in actual])
        
        # Calculate Sharpe ratio for the entire validation set
        SR = trading_strategy(predict, actual, contracts=5)
        
        # Log validation loss and print details
        self.log("val_loss", loss, prog_bar=True)
        if self.hparams.print_training:
            print("Validation loss: ", loss, f" Version: {v_num}", " Sharpe ratio: ", SR)
    
    def predict_step(self, batch: Data, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a prediction step on the input data.
        
        Parameters:
        -----------
        batch : torch_geometric.data.Data
            A batch of input data.
        batch_idx : int
            Index of the batch (used for logging or debugging).
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the predictions and actual values for the batch.
        """
        return (self.forward(batch).detach().cpu().numpy(), batch.y.detach().cpu().numpy())
        
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for training.
        
        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the optimizer and optional learning rate scheduler.
        """
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias"],
        )
        opt = torch.optim.AdamW(parameters, lr=self.hparams.learning_rate)
        return {"optimizer": opt}
