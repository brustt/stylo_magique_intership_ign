from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Losses(nn.Module):
    def __init__(self, losses: Dict[str, Dict]):
        """Init Losses wrapper for mix losses

        losses : {name : {params: xxx, weight: xxx}}

        Args:
            losses (Dict): losses dict 
        """
        super().__init__()
        self.dict_losses = self.init_losses(losses)

    def init_losses(self, losses) -> Dict[str, Tuple]:
        
        dict_losses = {}

        for name, loss in losses.items():
            ls_func = _register_losses[name]
            if loss["params"]:
                dict_losses[name] = (ls_func(**loss["params"]), loss["weight"])
            else:
                dict_losses[name] = (ls_func(), loss["weight"])

        return dict_losses
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Compute loss

        Args:
            x (Tensor): pred
            y (Tensor): label
        """
        if len(self.dict_losses) == 1:
            # Return the single loss directly without applying weighting
            return next(iter(self.dict_losses.values()))[0](x, y)
        else:
            # Combine losses if there are multiple
            combined_loss = 0.0
            for name, (loss, weight) in self.dict_losses.items():
                loss = loss(x, y)
                combined_loss += weight * loss
            return combined_loss



class FocalLoss(nn.Module):
    '''
    Adapted from https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html#sigmoid_focal_loss
    '''
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        """
            Args:
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target) -> torch.Tensor:
        """

        Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

        Raises:
            ValueError: _description_

        Returns:
            torch.Tensor: _description_
        """
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        
        return loss

class DiceLoss(nn.Module):
    """
    https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    """
    def __init__(self, smooth=1, to_probs=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.to_probs = to_probs 

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.to_probs:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

_register_losses = {
    "BCE": nn.BCEWithLogitsLoss,
    "FOCAL": FocalLoss,
    "DICE": DiceLoss

}