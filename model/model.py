'''
lfh 24/12/2
CNN +transform 配准+warp差分每个特征层+decoder
配准局部配准
'''
import numpy as np
from .resnet34 import Res34backbone
from .transformer import Feature_self_Attention
from .Decoder import Res_Decoder, Res_block
import torch
from .utils import *
import torch.nn as nn
import torch.nn.functional as F
def feature_warp(feature, flow, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode)

def local_correlation_softmax(feature0, feature1, local_radius,
                              padding_mode='zeros',
                              ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]
    sample_coords_softmax = sample_coords
    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax
    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]
    # mask invalid locations
    corr[~valid] = -1e9
    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]
    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
    v = correspondence - coords_init
    match_prob = prob
    return v, match_prob
class MFE(nn.Module):
    def __init__(self, num_scales=1,
                 upsample_factor= [8, 4, 2, 1, 1 / 2] ,
                 feature_channels=256,
                 ):
        super(MFE, self).__init__()
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor #1/8
        # self.upsample_factor = [4, 2, 1, 1 / 2,1/4] #1/4
        # self.upsample_factor = [16, 8, 4, 2, 1]
        self.backbone = Res34backbone()
        pretrained_dict = torch.load('./weights/resnet34-333f7ec4.pth')
        model_dict = self.backbone.state_dict()  # 获取自定义网络的权重
        print('Loadinng pre weight')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        self.decoder = Res_Decoder(num_scales, block=Res_block)
        self.feature_flow_attn = Feature_self_Attention(in_channels=feature_channels)
    def extract_feature(self, img0, img1, img2):
        concat = torch.cat((img0, img1, img2), dim=0)  # [2B, C, H, W]
        extract_out = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low
        features = extract_out[-2]
        feature0, feature1, feature2 = [], [], []
        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 3, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])
            feature2.append(chunks[2])
        return feature0, feature1, feature2, extract_out
    def upsample_vector(self, vector, bilinear=False, upsample_factor=8,
                      ):
        up_vector = F.interpolate(vector, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor
        return  up_vector
    def vector_up(self, v1, v2):
        vector1_list = []
        vector2_list = []  ###不同尺度上采样配准
        for i in range(len(self.upsample_factor)):
            vi_1 = self.upsample_vector(v1,  bilinear=True, upsample_factor=self.upsample_factor[i])
            vi_2 = self.upsample_vector(v2,  bilinear=True, upsample_factor=self.upsample_factor[i])
            vector1_list.append(vi_1)
            vector2_list.append(vi_2)
        return vector1_list, vector2_list

    def compute_vector(self, feature0, feature1, feature2, attn_window_list, Local_Window_list, prop_radius_list):
        upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1))
        attn_window = attn_window_list[0]
        Local_Window = Local_Window_list[0]
        prop_radius = prop_radius_list[0]
        # add position to features
        feature0, feature1, feature2 = feature_add_position(feature0, feature1, feature2, attn_window,
                                                            self.feature_channels)
        v_0_2 = local_correlation_softmax(feature0, feature2, Local_Window)[0]
        v_1_2 = local_correlation_softmax(feature1, feature2, Local_Window)[0]
        v_0_2s = self.feature_flow_attn(feature0, v_0_2.detach(),
                                       local_window_attn=prop_radius > 0,
                                       local_window_radius=prop_radius)
        v_1_2s = self.feature_flow_attn(feature1,  v_1_2.detach(),
                                       local_window_attn=prop_radius > 0,
                                       local_window_radius=prop_radius)

        return v_0_2s, v_1_2s

    def warp_feature(self, v0_2, v1_2, Features):
        x_list = []
        warp_img1 = []
        warp_img2 = []
        for i in range(len(Features)):
            temp = torch.chunk(Features[i][0], 3, 0)
            f1, f2, f3 = temp[0], temp[1], temp[2]
            f1_warp = feature_warp(f1, v0_2[i])
            f2_warp = feature_warp(f2, v1_2[i])
            diff1 = (f3 - f1_warp).abs()
            diff2 = (f3 - f2_warp).abs()
            warp_img1.append(f1_warp)
            warp_img2.append(f2_warp)
            # x_list.append(diff2+diff1)
            x_list.append(torch.cat([diff2, diff1], dim=1))
        return x_list, warp_img1[0], warp_img2[0]
    def forward(self, img0, img1, img2,
                attn_window_list,
                Local_Window_list,
                prop_radius_list,
                **kwargs,
                ):
        feature0_list, feature1_list, feature2_list, Extract_out = self.extract_feature(img0, img1,
                                                                                        img2)  # list of features
        # Extract_out=self.extract_feature_nodcbm(img0,img1,img2)
        v0_2, v1_2 = self.compute_vector(feature0_list[0], feature1_list[0], feature2_list[0], attn_window_list,
                                         Local_Window_list, prop_radius_list)
        v0_2_list, v1_2_list = self.vector_up(v0_2, v1_2 )
        warp_Feature, f1_warp, f2_warp = self.warp_feature( v0_2_list, v1_2_list, Extract_out)
        predict = self.decoder(warp_Feature)
        return predict, [f1_warp, f2_warp], [img2, img2]
