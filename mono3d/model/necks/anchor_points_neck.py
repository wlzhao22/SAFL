from mmdet3d.models.builder import NECKS
import torch 
import torch.nn as nn

from mono3d.model.backbones.dla_dcn import DLAUp 


@NECKS.register_module()
class AnchorPointsBasedViewTransformation(nn.Module):
    def __init__(self,  voxel_size, ranges,  in_channels, down_ratio, groups=1, levels=3) -> None:
        super().__init__()
        self.bev_map_xs = int(round((ranges[3] - ranges[0]) / voxel_size[0])) 
        self.bev_map_ys = int(round((ranges[4] - ranges[1]) / voxel_size[1])) 
        self.bev_map_zs = int(round((ranges[5] - ranges[2]) / voxel_size[2]))
        self.pos_embed_mlp = nn.Sequential(
            nn.Linear(3, 16, ),
            nn.ReLU(True), 
            nn.Linear(16, 32), 
            nn.ReLU(True), 
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        self.in_channels = in_channels 
        self.down_ratio = down_ratio
        self.groups = groups

        self.list_conv_adaption = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 * 2**i, 64, 3, 1, 1, bias=False), 
                nn.BatchNorm2d(64, ), 
                nn.ReLU(True),
            ) for i in range(3)
        ])

        self.levels = levels 
        self.conv_in = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.dla_up = DLAUp(0, channels=[64 * 2**i for i in range(levels)], scales=[2**i for i in range(levels)])
        self.list_conv_down = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(64 * 2**i, 64 * 2**(i + 1), 4, 2, 1, bias=False), 
                nn.BatchNorm2d(64 * 2**(i + 1)),
                nn.ReLU(True),
            ) for i in range(levels - 1)]
        )

        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2, ),
            nn.ReLU(True), 
            nn.Linear(in_channels * 2, in_channels * 2), 
            nn.ReLU(True), 
            nn.Linear(in_channels * 2, in_channels),
        )

        # self.pos_embed = nn.Parameter(torch.zeros(self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, self.in_channels))

    def pos_embed(self, device, eps=1e-7):
        yy, xx, zz = torch.meshgrid(torch.arange(self.bev_map_ys), torch.arange(self.bev_map_xs), torch.arange(self.bev_map_zs))
        pos = torch.stack((yy, xx, zz), axis=-1).float().to(device)
        pos = pos.div(pos.new_tensor([self.bev_map_ys, self.bev_map_xs, self.bev_map_zs])) 
        assert pos.shape == (self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, 3)
        f = self.pos_embed_mlp(pos.view(-1, 3))
        f = f.view(self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, self.in_channels)
        return f 
        # z = torch.arange(self.bev_map_xs).float()
        # z = z.expand(self.bev_map_ys, self.bev_map_zs,  self.bev_map_xs,)
        # z = z.permute(0, 2, 1).unsqueeze(-1)
        # return z.to(device)
    
    # def attention(self, features, pos_embed, eps=1e-7):
    #     bs = features.shape[0]
    #     ch = features.shape[-1]
    #     assert features.shape == (bs, self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, ch)
    #     query = features.view(-1, ch)
    #     query = self.attention_mlp(query)
    #     query = query.view(bs, self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, ch)
    #     # query, temp = query[..., [0]], query[..., 1].sigmoid() * 0.05
    #     # query = (1. / query.add(eps) - 1)
    #     # w = torch.exp(-(query - pos_embed).pow(2.).sum(-1).sqrt() * temp)
    #     query = query.div(torch.norm(query, p=2, dim=-1, keepdim=True).add(eps))
    #     pos_embed = pos_embed.div(torch.norm(pos_embed, p=2, dim=-1, keepdim=True).add(eps))
    #     w = query * pos_embed 
    #     w = w.sum(-1, keepdim=True)
    #     # w = w.max(-1)[0].unsqueeze(-1)
    #     # assert 0 <= w.min() and w.max() <= 1
    #     assert w.shape == (bs, self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, 1)
    #     return features * w, w.squeeze(-1).max(-1)[0]

    def attention(self, features, pos_embed):
        return features + pos_embed, None

    def bev_conv(self, bev_features:torch.Tensor, ):
        levels = [self.conv_in(bev_features.contiguous())]
        for i in range(self.levels - 1):
            levels.append(self.list_conv_down[i](levels[-1]))
        out = self.dla_up(levels)
        return out[0]

    def forward(self, features, anchor_points_xyz, anchor_points_uv, anchor_points_mask):
        bs, ch, h, w = features[0].shape
        device = features[0].device
        features = [self.list_conv_adaption[i](f) for i, f in enumerate(features)]
        
        assert anchor_points_xyz.shape == (bs, self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, 3)
        batch_indices = torch.arange(bs).to(device)[:, None, None, None,].expand(bs, self.bev_map_ys, self.bev_map_xs, self.bev_map_zs,)

        anchor_points_uv_d = anchor_points_uv.div(self.down_ratio).floor().long()
        # u_max = anchor_points_uv_d[..., 0].max()
        # u_min = anchor_points_uv_d[..., 0].min() 
        # v_max = anchor_points_uv_d[..., 1].max() 
        # v_min = anchor_points_uv_d[..., 1].min() 
        anchor_point_features = [f[
            batch_indices, 
            :, 
            anchor_points_uv_d[..., 1].div(2**i), 
            anchor_points_uv_d[..., 0].div(2**i)
        ].contiguous() for i, f in enumerate(features)]
        assert anchor_point_features[0].shape == (bs, self.bev_map_ys, self.bev_map_xs, self.bev_map_zs, ch)

        pos_embed = self.pos_embed(device)
        # pos_embed = self.pos_embed
        # anchor_point_features, activation = self.attention(anchor_point_features, pos_embed)
        anchor_point_features = [f + pos_embed for f in anchor_point_features]
        anchor_point_features = [f * anchor_points_mask.unsqueeze(-1) for f in anchor_point_features]

        # collapse 
        bev_features = [f.mean(-2).permute(0, 3, 1, 2) for f in anchor_point_features]
        assert bev_features[0].shape == (bs, ch, self.bev_map_ys, self.bev_map_xs)
        bev_features = torch.cat(bev_features, axis=1)

        bev_features = self.bev_conv(bev_features)

        return features[0], bev_features, None
