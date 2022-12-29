import torch.nn as nn 
import torch 
from mmdet3d.models import NECKS


@NECKS.register_module()
class PV2BEV(nn.Module):
    def __init__(self, voxel_size, ranges, in_channels, n_z):
        super().__init__()
        self.voxel_size = voxel_size 
        self.ranges = ranges 
        self.bev_map_zs = int(round((ranges[3] - ranges[0]) / voxel_size[0]))
        self.bev_map_xs = int(round((ranges[4] - ranges[1]) / voxel_size[1])) 
        self.bev_map_ys = int(round((ranges[5] - ranges[2]) / voxel_size[2])) 
        self.n_z = n_z
        self.implicit_depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, n_z, 1),
            nn.Sigmoid()
        )

    def forward(self, x, frustum_xyz):
        # x.shape == (bs, ch, h, w)
        # weights = self.implicit_depth_conv(x)
        # weights = weights / torch.norm(weights, 1, dim=1, keepdim=True)

        # weights.shape == (bs, n_z, h, w)
        # x = x.permute(0, 2, 3, 1).unsqueeze(1) * weights.unsqueeze(-1)
        bs, ch, h, w = x.shape 
        n_z = self.n_z 
        x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(1).expand(bs, n_z, h, w, ch)
        
        # x.shape == (bs, n_z, h, w, ch)
        Nprime = bs * n_z * h * w
        # x = x.view(Nprime, ch)
        frustum_xyz = frustum_xyz.reshape(Nprime, 3)
        batch_idx = torch.cat([torch.full([Nprime//bs, 1], ix, device=x.device, dtype=torch.long) for ix in range(bs)])
        frustum_xyzb = torch.cat((frustum_xyz, batch_idx), -1)
        del frustum_xyz

        kept = (frustum_xyzb[:, 0] >= 0) & (frustum_xyzb[:, 0] < self.bev_map_xs) \
            & (frustum_xyzb[:, 1] >= 0) & (frustum_xyzb[:, 1] < self.bev_map_ys) \
            & (frustum_xyzb[:, 2] >= 0) & (frustum_xyzb[:, 2] < self.bev_map_zs)

        # x = x[kept]
        x = x[kept.view(bs, n_z, h, w,)]
        frustum_xyzb = frustum_xyzb[kept]
        ranks = frustum_xyzb[:, 0] * (self.bev_map_ys * self.bev_map_zs * bs) \
            + frustum_xyzb[:, 1] * (self.bev_map_zs * bs) \
            + frustum_xyzb[:, 2] * bs \
            + frustum_xyzb[:, 3]
        sorts = ranks.argsort() 
        x, frustum_xyzb, ranks = x[sorts], frustum_xyzb[sorts], ranks[sorts]

        x, frustum_xyzb = QuickCumsum.apply(x, frustum_xyzb, ranks) 

        final = torch.zeros((bs, ch, self.bev_map_zs, self.bev_map_xs, self.bev_map_ys), device=x.device)
        final[frustum_xyzb[:, 3], :, frustum_xyzb[:, 2], frustum_xyzb[:, 0], frustum_xyzb[:, 1]] = x

        # collapse y 
        # final = torch.cat(final.unbind(dim=-1), 1)
        final = final.sum(-1)
        
        return final 
        
    
class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None