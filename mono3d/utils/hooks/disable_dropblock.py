from dropblock import DropBlock2D
from mmcv.utils import Registry
from mmcv.runner.hooks import Hook, HOOKS
from mmcv.runner import EpochBasedRunner
from tenacity import after
import torch.nn as nn 
import torch.distributed as dist 


@HOOKS.register_module()
class DisableDropBlock(Hook):
    def __init__(self, epoch: int) -> None:
        super().__init__()
        self.epoch = epoch 

    def after_train_epoch(self, runner):
        assert isinstance(runner, EpochBasedRunner), type(runner)
        if runner.epoch + 1 != self.epoch: return 
        model:nn.Module = runner.model 
        for k, v in model.named_modules():
            if isinstance(v, DropBlock2D):
                print('disable dropblock module. rank: {}; key: {}'.format(dist.get_rank(), k))
                assert hasattr(v, 'drop_prob'), type(v)
                v.drop_prob = 0. 


    