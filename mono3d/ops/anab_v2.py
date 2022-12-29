
import torch 
import torch.nn as nn

from mono3d.ops.anab import ANAB 


class ANAB_v2(ANAB):
    def __init__(self, ch, psp_size=[1, 4, 8, 16], with_atten=True):
        super().__init__(ch, None, psp_size, with_atten)

    def forward(self, x, y):
        assert x.shape == y.shape 
        B, C, H, W = x.size() 

        query = self.query_conv(x) 
        query = query.view(B, self.key_ch, H * W).permute(0, 2, 1)
        if self.with_atten:
            psp_atten = self.spatial_conv(y)
            psp_atten = self.sigmoid(psp_atten) 
        else:
            psp_atten = None 
        
        key = self.key_conv(y)
        key = self.key_papa(key, psp_atten) 

        value = self.value_conv(y)
        value = self.value_papa(value, psp_atten).permute(0, 2, 1)

        atten = torch.bmm(query, key) 
        atten = self.softmax(atten) 

        new_value = torch.bmm(atten, value) 
        new_value = new_value.permute(0, 2, 1).view(B, self.outch, H, W)

        out = new_value + x 

        return out.contiguous()

