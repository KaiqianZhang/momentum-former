import torch
import torch.nn as nn

# Custom Transformer layer with modifications for this model
class TransformerLayer2(nn.Module):
    """
    A modified transformer encoder layer for the model.
    
    Attributes:
    -----------
    d_model : int
        Dimensionality of the input and output embeddings.
    nhead : int
        Number of heads in the multi-head attention mechanism.
    dim_feedforward : int
        Size of the feedforward network.
    dropout : float
        Dropout rate applied to various components.
    layer_norm_eps : float
        Epsilon value for layer normalization.
    batch_first : bool
        Whether the input and output tensors are provided as (batch, seq, feature).
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Initialize multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        
        # Feedforward network layers
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(dim_feedforward, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU activation function for non-linearity

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the transformer layer.
        
        Parameters:
        -----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
        --------
        torch.Tensor
            Output tensor after applying the transformer layer.
        """
        x = src
        
        # Apply self-attention and add residual connection
        x = self.norm1(x + self._sa_block(x))
        
        # Apply feedforward network and add residual connection
        x = self.norm2(x + self._ff_block(x))
        
        return x

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self-attention block within the transformer layer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.
        
        Returns:
        --------
        torch.Tensor
            Output tensor after self-attention and dropout.
        """
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feedforward block within the transformer layer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.
        
        Returns:
        --------
        torch.Tensor
            Output tensor after feedforward layers and dropout.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
