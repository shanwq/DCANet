import torch
import numpy as np
import torch.nn.functional
import torch.nn.functional as F
import math, copy
from copy import deepcopy
from torch import nn, Tensor
from typing import Optional
from torch.cuda.amp import autocast
import os
from .unet_my_3d_3scale import Unet_Mip3D
# from .mask2former_modeling.transformer_decoder.mask2former_transformer_decoder3d import MultiScaleMaskedTransformerDecoder3d
from scipy.ndimage import rotate, zoom

from typing import Any, Optional, Tuple, Type

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Mask_to_image_Attention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_mask = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_image = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_image = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_embed, mask_embed):
        img_embed = img_embed.permute(0, 2, 3, 4, 1)
        mask_embed = mask_embed.permute(0, 2, 3, 4, 1)
        img_embed = img_embed.reshape(1, -1, self.dim)
        mask_embed = mask_embed.reshape(1, -1, self.dim)
        
        b, n, c = img_embed.shape
        # print('bnc',b,n,c, img_embed.shape, mask_embed.shape)
        q_mask = self.q_mask(mask_embed).reshape(b, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k_image = self.k_image(img_embed).reshape(b, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        v_image = self.v_image(img_embed).reshape(b, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q_mask = q_mask * self.scale
        attn = q_mask @ k_image.transpose(-2, -1)

        # print('relative_position_bias is', relative_position_bias.shape)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v_image.dtype)
        x = (attn @ v_image).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(1,320,16,4,4 )
        return x


class DCANet_Concat(nn.Module):

    def __init__(self, input_channels=1, base_num_features=32, num_classes=2, num_conv_per_stage=2,

                #  nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                #  final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                #  conv_kernel_sizes=None,
                #  upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False, # TODO default False
                #  max_num_features=None, basic_block=ConvDropoutNormNonlin,
                #  seg_output_use_bias=False,
                #  patch_size=None, is_vit_pretrain=False, 
                 vit_depth=12, vit_hidden_size=768, vit_mlp_dim=3072, vit_num_heads=12,
                 max_msda='', is_max_ms=True, is_max_ms_fpn=False, max_n_fpn=4, max_ms_idxs=[-4,-3,-2], max_ss_idx=0,
                 is_max_bottleneck_transformer=False, max_seg_weight=1.0, max_hidden_dim=192, max_dec_layers=10,
                 mw = 0.5,):
                #  is_max=True, is_masked_attn=False, is_max_ds=False, is_masking=False, is_masking_argmax=False,
                #  is_fam=False, fam_k=5, fam_reduct_ratio=8,
                #  is_max_hungarian=False, num_queries=None, is_max_cls=False,
                #  point_rend=False, num_point_rend=None, no_object_weight=None, is_mhsa_float32=False, no_max_hw_pe=False,
                #  max_infer=None, cost_weight=[2.0, 5.0, 5.0], vit_layer_scale=False, decoder_layer_scale=False):

        super(DCANet_Concat, self).__init__()

        self.mip_3D = Unet_Mip3D()
        
        self.input_channels = input_channels # 1 
        self.base_num_features = base_num_features # 32
        self.num_classes = num_classes # 1
        self.num_conv_per_stage = num_conv_per_stage # 2 
        output_features = base_num_features
        input_features = input_channels
        
        # self.Mip_Prompt_bottle = nn.Sequential(
        #     ConvBlock(input_channels=32, output_channels=64, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
        #     ConvBlock(input_channels=64, output_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
        #     ConvBlock(input_channels=128, output_channels=256, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
        #     ConvBlock(input_channels=256, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
        #     ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
        # )
        # self.Mip_Prompt_0 = nn.Sequential(
        #     ConvBlock(input_channels=1, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
        #     ConvBlock(input_channels=32, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
        # )
        
        # self.conv_blocks_context_0 = nn.ModuleList()
        self.conv_blocks_context_0 = nn.Sequential(
            ConvBlock(input_channels=1, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
            ConvBlock(input_channels=32, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
        )
        self.conv_blocks_context_1= nn.Sequential(
            ConvBlock(input_channels=64, output_channels=64, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=64, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_2= nn.Sequential(
            ConvBlock(input_channels=128, output_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=128, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_3= nn.Sequential(
            ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_4= nn.Sequential(
            # ConvBlock(input_channels=512, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=256, output_channels=320, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_5= nn.Sequential(
            # ConvBlock(input_channels=640, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        # self.tu_0 = nn.ConvTranspose3d(in_channels=640, out_channels=320, kernel_size=(1,2,2), 
        self.tu_0 = nn.ConvTranspose3d(in_channels=320, out_channels=320, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)
        self.conv_blocks_localization_0= nn.Sequential(
            ConvBlock(input_channels=640, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        # self.tu_1 = nn.ConvTranspose3d(in_channels=640, out_channels=256, kernel_size=(1,2,2), 
        self.tu_1 = nn.ConvTranspose3d(in_channels=320, out_channels=256, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)
        self.conv_blocks_localization_1 = nn.Sequential(
            ConvBlock(input_channels=512, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            # ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.tu_2 = nn.ConvTranspose3d(in_channels=512, out_channels=128, kernel_size=(2,2,2), 
        # self.tu_2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)
        self.conv_blocks_localization_2 = nn.Sequential(
            ConvBlock(input_channels=256, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=128, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_3 = nn.ConvTranspose3d(in_channels=256, out_channels=64, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)

        self.conv_blocks_localization_3 = nn.Sequential(
            ConvBlock(input_channels=128, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=64, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_4 = nn.ConvTranspose3d(in_channels=128, out_channels=32, kernel_size=(1,2,2), 
                                       stride=(1,2,2), bias=False)

        self.conv_blocks_localization_4 = nn.Sequential(
            ConvBlock(input_channels=64, output_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=32, output_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        # self.input_proj_0 = nn.ModuleList()
        self.input_proj_0 = nn.Sequential(
                nn.Conv3d(in_channels=256, out_channels=192, kernel_size=(1,1,1), stride=(1,1,1)),
                nn.GroupNorm(32, num_channels=192, eps=1e-05),
            )
        # self.input_proj_1 = nn.ModuleList()
        self.input_proj_1 = nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=192, kernel_size=(1,1,1), stride=(1,1,1)),
                nn.GroupNorm(32, num_channels=192, eps=1e-05),
            )
        # self.input_proj_2 = nn.ModuleList()
        self.input_proj_2 = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=192, kernel_size=(1,1,1), stride=(1,1,1)),
                nn.GroupNorm(32, num_channels=192, eps=1e-05),
            )
        
        self.linear_mask_features = nn.Conv3d(in_channels=32, out_channels=192, kernel_size=(1,1,1), stride=(1,1,1))
        self.linear_mask_features_mip = nn.Conv3d(in_channels=32, out_channels=192, kernel_size=(1,1,1), stride=(1,1,1))
        from .mask2former_modeling.transformer_decoder.mask2former_transformer_decoder3d_Mask import MultiScaleMaskedTransformerDecoder3d
        cfg = {
                    "num_classes": 2,
                    "hidden_dim": 192,
                    "num_queries": 20,  # N=K if 'fixed matching', else default=100,
                    "nheads": 8,
                    "dim_feedforward": 1536, # 2048,
                    "dec_layers": 3, # max_dec_layers, # 9 decoder layers, add one for the loss on learnable query?
                    "pre_norm": False,
                    "enforce_input_project": False,
                    "mask_dim": 192, #max_hidden_dim, # input feat of segm head?
                    "non_object": True,
                    "use_layer_scale": False,
                }
        cfg['num_feature_levels'] = 3 # 1 if not self.is_max_ms or self.is_max_ms_fpn else 3
        cfg["is_masking"] = True #if is_masking else False
        cfg["is_masking_argmax"] = False #True if is_masking_argmax else False
        cfg["is_mhsa_float32"] = True #if is_mhsa_float32 else False
        cfg["no_max_hw_pe"] = False# True if no_max_hw_pe else False
        self.predictor = MultiScaleMaskedTransformerDecoder3d(in_channels=max_hidden_dim, mask_classification=True, **cfg)
        
        
        
    def forward(self, x, mip):
        skips = []
        seg_outputs = [] # deepsupervison
        # print('input_x.shape is', x.shape)
        x = self.conv_blocks_context_0(x)
        
        pred_soft, feature_list = self.mip_3D(mip)
        # for i in range(len(feature_list)):
        #     print(i, feature_list[i].shape)
        # # x= x + fea_mip
        # quit()
        # print('conv_blocks_context_0.shape is', x.shape)
        # print('feature_list[0].shape is', feature_list[0].shape)
        skips.append(x)
        x = torch.concat([x, feature_list[0]], dim=1)
        x = self.conv_blocks_context_1(x)
        # print('conv_blocks_context_1.shape is', x.shape)
        # print('feature_list[1].shape is', feature_list[1].shape)
        # quit()
        skips.append(x)
        x = torch.concat([x, feature_list[1]], dim=1)
        x = self.conv_blocks_context_2(x)
        # print('conv_blocks_context_2.shape is', x.shape)
        # print('feature_list[2].shape is', feature_list[2].shape)
        skips.append(x)
        x = torch.concat([x, feature_list[2]], dim=1)
        x = self.conv_blocks_context_3(x)
        # print('conv_blocks_context_3.shape is', x.shape)
        # print('feature_list[3].shape is', feature_list[3].shape)
        skips.append(x)
        # x = torch.concat([x, feature_list[3]], dim=1)
        x = self.conv_blocks_context_4(x)
        # print('conv_blocks_context_4.shape is', x.shape)
        # print('feature_list[4].shape is', feature_list[4].shape)
        skips.append(x)
        
        # x = torch.concat([x, feature_list[4]], dim=1)
        x = self.conv_blocks_context_5(x)
        # print('conv_blocks_context_5.shape is', x.shape)
        # print('feature_list[5].shape is', feature_list[5].shape)
        
        ds_feats = [] # obtain multi-scale feature
        ds_feats.append(x)
        # x = torch.concat([x, feature_list[5]], dim=1)
        
        # Mip_bn_fea = self.Mip_Prompt_bottle(fea_mip)
        # print('Mip_bn_fea',Mip_bn_fea.shape)
        # cross_fea_mip = self.mask_to_image_attn(x, Mip_bn_fea)
        
        # x= x + cross_fea_mip
        
        
        x = self.tu_0(x)
        # print('tu_0.shape is', x.shape)
        x = torch.cat((x, skips[-1]), dim=1)
        # print('tu_0_cat.shape is', x.shape)
        x = self.conv_blocks_localization_0(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_0.shape is', x.shape)
        # print('feature_list[6].shape is', feature_list[6].shape)
        
        # x = torch.concat([x, feature_list[6]], dim=1)
        x = self.tu_1(x)
        # print('tu_1.shape is', x.shape)
        x = torch.cat((x, skips[-2]), dim=1)
        # print('tu_1_cat.shape is', x.shape)
        x = self.conv_blocks_localization_1(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_1.shape is', x.shape)
        
        x = torch.concat([x, feature_list[7]], dim=1)
        # print('feature_list[7].shape is', feature_list[7].shape)
        x = self.tu_2(x)
        x = torch.cat((x, skips[-3]), dim=1)
        x = self.conv_blocks_localization_2(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_2.shape is', x.shape)
        
        x = torch.concat([x, feature_list[8]], dim=1)
        # print('feature_list[8].shape is', feature_list[8].shape)
        x = self.tu_3(x)
        x = torch.cat((x, skips[-4]), dim=1)
        x = self.conv_blocks_localization_3(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_3.shape is', x.shape)
        
        x = torch.concat([x, feature_list[9]], dim=1)
        # print('feature_list[9].shape is', feature_list[9].shape)
        x = self.tu_4(x)
        x = torch.cat((x, skips[-5]), dim=1)
        x = self.conv_blocks_localization_4(x)
        # print('conv_blocks_localization_4.shape is', x.shape)
        # print('feature_list[10].shape is', feature_list[10].shape)
        # print(x.shape)
        # quit()
        ds_feats.append(x)
        
        
        multi_scale_features = []
        ms_pixel_feats = ds_feats[-4:-1]
        
        f_0 = self.input_proj_0(ms_pixel_feats[0])
        # print('input_proj_0.shape is', ms_pixel_feats[0].shape)
        multi_scale_features.append(f_0)
        f_1 = self.input_proj_1(ms_pixel_feats[1])
        # print('input_proj_1.shape is', ms_pixel_feats[1].shape)
        multi_scale_features.append(f_1)
        f_2 = self.input_proj_2(ms_pixel_feats[2])
        multi_scale_features.append(f_2)
        # print('input_proj_2.shape is', ms_pixel_feats[2].shape)
        
        transformer_decoder_in_feature =  multi_scale_features  # feature pyramid
        
        mask_features = self.linear_mask_features(ds_feats[-1]) # following SingleScale
        # print('self.mask_features :', mask_features.shape)
        mask_features_mip = self.linear_mask_features_mip(feature_list[-1]) # following SingleScale
        
        predictions = self.predictor(transformer_decoder_in_feature, mask_features, mask_features_mip, mask=None)
        # print('self.predictions :', len(predictions),predictions.keys())
        return pred_soft, predictions
        


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(192,192), patch_size=(16,16), in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H % self.img_size[0] == 0 and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # assert C == self.in_chans, \
        #     f"Input image chanel ({C}) doesn't match model ({self.in_chans})"
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
            
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if len(x.shape) == 4: # 2d
            if mask is None:
                mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    
        elif len(x.shape) == 5: # 3d
            if mask is None:
                mask = torch.zeros((x.size(0), x.size(2), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
            not_mask = ~mask
            z_embed = not_mask.cumsum(1, dtype=torch.float32)
            y_embed = not_mask.cumsum(2, dtype=torch.float32)
            x_embed = not_mask.cumsum(3, dtype=torch.float32)
            
            if self.normalize:
                eps = 1e-6
                z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
                y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            # dim_t = self.temperature ** (3 * (dim_t // 3) / self.num_pos_feats)
            dim_t = self.temperature ** torch.div(3 * torch.div(dim_t, 3), self.num_pos_feats, rounding_mode='trunc')
            pos_x = x_embed[:, :, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, :, None] / dim_t
            pos_z = z_embed[:, :, :, :, None] / dim_t

            pos_x = torch.stack(
                (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5
            ).flatten(4)
            pos_y = torch.stack(
                (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5
            ).flatten(4)
            pos_z = torch.stack(
                (pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5
            ).flatten(4)

            pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
            # pos = (pos_z + pos_y + pos_x).permute(0, 4, 1, 2, 3)

        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
        
        
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, kernel_size=None, stride=None, padding = None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvBlock, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        # if conv_kwargs is None:
        #     conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        # self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = self.conv_op(input_channels, output_channels, kernel_size, stride, padding = padding)
        
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        # print('conv.shape is', x.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0] 
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(self.ls1(tgt2))
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)
        

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.ls1 = nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]  
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]

        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(self.ls1(tgt2))

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False, use_layer_scale=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(self.ls1(tgt))
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(self.ls1(tgt2))
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class StackedConvLayers(nn.Module):
    def __init__(self, input_features, output_features, num_conv_per_stage,
                 conv_op=nn.Conv3d, kernel_size=None, stride=None, padding = None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_features
        self.output_channels = output_features

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        # if conv_kwargs is None:
        #     conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        # self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        # if first_stride is not None:
        #     self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
        #     self.conv_kwargs_first_conv['stride'] = first_stride
        # else:
        #     self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            # *([ConvDropoutNormNonlin(input_features, output_features, self.conv_op,
            #             #    self.conv_kwargs_first_conv,    
            #                kernel_size, stride[0], padding, 
            #                self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
            #                self.nonlin, self.nonlin_kwargs)] +
            #   [ConvDropoutNormNonlin(input_features, output_features, self.conv_op,
            #             #    self.conv_kwargs,
            #                kernel_size, stride[1], padding, 
            #                self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
            #                self.nonlin, self.nonlin_kwargs) for _ in range(num_conv_per_stage - 1)]))
            ConvDropoutNormNonlin(input_features, output_features, self.conv_op,
                        #    self.conv_kwargs_first_conv,    
                           kernel_size, stride[0], padding, 
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs),
            ConvDropoutNormNonlin(input_features, output_features, self.conv_op,
                        #    self.conv_kwargs,
                           kernel_size, stride[1], padding, 
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs))

    def forward(self, x):
        return self.blocks(x)
    
    
class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, kernel_size=None, stride=None, padding = None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        # if conv_kwargs is None:
        #     conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        # self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = self.conv_op(input_channels, output_channels, kernel_size, stride, padding = padding)
        
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        # print('conv.shape is', x.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))
