from .dice_loss import DiceLoss
from .rot_loss import BinRotLoss
from .UltraFast_loss import OhemCELoss, SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis


__all__ = [
	'DiceLoss','BinRotLoss', 'OhemCELoss', 'SoftmaxFocalLoss', 'ParsingRelationLoss', 'ParsingRelationDis'
]
