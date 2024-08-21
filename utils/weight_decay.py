import torch.nn as nn
from typing import List, Dict, Any, Tuple

# Function to apply weight decay to selected model parameters
def add_weight_decay(
    model: nn.Module,
    weight_decay: float = 1e-5,
    skip_list: Tuple[str] = ("bias", "bn", "LayerNorm.bias", "LayerNorm.weight"),
) -> List[Dict[str, Any]]:
    """
    Add weight decay to the model's optimizer, excluding certain parameters.
    
    Parameters:
    -----------
    model : nn.Module
        The PyTorch model containing the parameters.
    weight_decay : float
        The weight decay factor.
    skip_list : Tuple[str]
        List of parameter names to exclude from weight decay.
    
    Returns:
    --------
    List[Dict[str, Any]]
        A list of dictionaries specifying which parameters should or should not have weight decay applied.
    """
    decay = []
    no_decay = []
    
    # Iterate through all model parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters
        
        # Apply weight decay only to specific types of parameters
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    
    # Return the parameter groups for the optimizer
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
