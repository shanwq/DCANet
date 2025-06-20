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
# from unet_my import Unet_Mip
# from .mask2former_modeling.transformer_decoder.mask2former_transformer_decoder3d import MultiScaleMaskedTransformerDecoder3d
# quit()
'''TransUnet3D'''
from scipy.ndimage import rotate, zoom

class TransUnet3d(nn.Module):

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

        super().__init__()

        self.input_channels = input_channels # 1 
        self.base_num_features = base_num_features # 32
        self.num_classes = num_classes # 1
        self.num_conv_per_stage = num_conv_per_stage # 2 
        output_features = base_num_features
        input_features = input_channels
        
        
        # self.conv_blocks_context_0 = nn.ModuleList()
        self.conv_blocks_context_0 = nn.Sequential(
            ConvBlock(input_channels=1, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
            ConvBlock(input_channels=32, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
        )
        self.conv_blocks_context_1= nn.Sequential(
            ConvBlock(input_channels=32, output_channels=64, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=64, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_2= nn.Sequential(
            ConvBlock(input_channels=64, output_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=128, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_3= nn.Sequential(
            ConvBlock(input_channels=128, output_channels=256, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_4= nn.Sequential(
            ConvBlock(input_channels=256, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_5= nn.Sequential(
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_0 = nn.ConvTranspose3d(in_channels=320, out_channels=320, kernel_size=(1,2,2), 
                                       stride=(1,2,2), bias=False)
        self.conv_blocks_localization_0= nn.Sequential(
            ConvBlock(input_channels=640, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_1 = nn.ConvTranspose3d(in_channels=320, out_channels=256, kernel_size=(1,2,2), 
                                       stride=(1,2,2), bias=False)
        self.conv_blocks_localization_1 = nn.Sequential(
            ConvBlock(input_channels=512, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.tu_2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)
        self.conv_blocks_localization_2 = nn.Sequential(
            ConvBlock(input_channels=256, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=128, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)

        self.conv_blocks_localization_3 = nn.Sequential(
            ConvBlock(input_channels=128, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=64, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_4 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(1,2,2), 
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
        from .mask2former_modeling.transformer_decoder.mask2former_transformer_decoder3d import MultiScaleMaskedTransformerDecoder3d
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
        
        # self.predictor = nn.ModuleList()
        
        # self.pe_layer = PositionEmbeddingSine(num_pos_feats= 64, temperature= 10000, normalize=True, scale = 6.283185307179586)

        # self.transformer_self_attention_layers = nn.ModuleList()
        # self.transformer_self_attention_layers.append(
        #     SelfAttentionLayer(d_model=192, nhead=8),
        #     SelfAttentionLayer(d_model=192, nhead=8),
        #     SelfAttentionLayer(d_model=192, nhead=8),
        # )
        # # self.transformer_self_attention_layers_0 = nn.ModuleList()
        # # self.transformer_self_attention_layers_0.append(
        # #     SelfAttentionLayer(d_model=192, nhead=8),
        # # )
        # # self.transformer_self_attention_layers_1 = nn.ModuleList()
        # # self.transformer_self_attention_layers_1.append(
        # #     SelfAttentionLayer(d_model=192, nhead=8),
        # # )
        # # self.transformer_self_attention_layers_2 = nn.ModuleList()
        # # self.transformer_self_attention_layers_2.append(
        # #     SelfAttentionLayer(d_model=192, nhead=8),
        # # )
        
        # self.transformer_cross_attention_layers = nn.ModuleList()
        # self.transformer_cross_attention_layers.append(
        #     CrossAttentionLayer(d_model=192, nhead=8),
        #     CrossAttentionLayer(d_model=192, nhead=8),
        #     CrossAttentionLayer(d_model=192, nhead=8),
        # )
        
        # # self.transformer_cross_attention_layers_0 = nn.ModuleList()
        # # self.transformer_cross_attention_layers_0.append(
        # #     CrossAttentionLayer(d_model=192, nhead=8)
        # # )
        
        # # self.transformer_cross_attention_layers_1 = nn.ModuleList()
        # # self.transformer_cross_attention_layers_1.append(
        # #     CrossAttentionLayer(d_model=192, nhead=8)
        # # )
        # # self.transformer_cross_attention_layers_2 = nn.ModuleList()
        # # self.transformer_cross_attention_layers_2.append(
        # #     CrossAttentionLayer(d_model=192, nhead=8)
        # # )
        
        # self.transformer_ffn_layers = nn.ModuleList()
        # self.transformer_ffn_layers.append(
        #     FFNLayer(d_model=192, dim_feedforward=1536),
        #     FFNLayer(d_model=192, dim_feedforward=1536),
        #     FFNLayer(d_model=192, dim_feedforward=1536),
        # )
        # # self.transformer_ffn_layers_0 = nn.ModuleList()
        # # self.transformer_ffn_layers_0.append(
        # #     FFNLayer(d_model=192, dim_feedforward=1536)
        # # )
        
        # # self.transformer_ffn_layers_1 = nn.ModuleList()
        # # self.transformer_ffn_layers_1.append(
        # #     FFNLayer(d_model=192, dim_feedforward=1536)
        # # )
        
        # # self.transformer_ffn_layers_2 = nn.ModuleList()
        # # self.transformer_ffn_layers_2.append(
        # #     FFNLayer(d_model=192, dim_feedforward=1536)
        # # )
        
        # self.decoder_norm = nn.LayerNorm(192, eps=1e-05, elementwise_affine=True)
        
        # self.num_queries = 20
        # # learnable query features
        # self.query_feat = nn.Embedding(self.num_queries, 192)
        # # learnable query p.e.
        # self.query_embed = nn.Embedding(self.num_queries, 192)

        # self.level_embed = nn.Embedding(3, 192)
        
        # # self.input_proj
        # self.class_embed = nn.Linear(in_features=192, out_features=3, bias=True)
        
        # self.mask_embed = MLP(input_dim=192, hidden_dim=192, output_dim=192, num_layers=3)
        

        
    def forward(self, x):
        skips = []
        seg_outputs = [] # 当深监督时会用到
        # print('input_x.shape is', x.shape)
        x = self.conv_blocks_context_0(x)
        # print('conv_blocks_context_0.shape is', x.shape)
        skips.append(x)
        x = self.conv_blocks_context_1(x)
        # print('conv_blocks_context_1.shape is', x.shape)
        # quit()
        skips.append(x)
        x = self.conv_blocks_context_2(x)
        # print('conv_blocks_context_2.shape is', x.shape)
        skips.append(x)
        x = self.conv_blocks_context_3(x)
        # print('conv_blocks_context_3.shape is', x.shape)
        skips.append(x)
        x = self.conv_blocks_context_4(x)
        # print('conv_blocks_context_4.shape is', x.shape)
        skips.append(x)
        
        x = self.conv_blocks_context_5(x)
        # print('conv_blocks_context_5.shape is', x.shape)
        
        ds_feats = [] # obtain multi-scale feature
        ds_feats.append(x)
        
        x = self.tu_0(x)
        x = torch.cat((x, skips[-1]), dim=1)
        x = self.conv_blocks_localization_0(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_0.shape is', x.shape)
        
        x = self.tu_1(x)
        x = torch.cat((x, skips[-2]), dim=1)
        x = self.conv_blocks_localization_1(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_1.shape is', x.shape)
        
        x = self.tu_2(x)
        x = torch.cat((x, skips[-3]), dim=1)
        x = self.conv_blocks_localization_2(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_2.shape is', x.shape)
        
        x = self.tu_3(x)
        x = torch.cat((x, skips[-4]), dim=1)
        x = self.conv_blocks_localization_3(x)
        ds_feats.append(x)
        # print('conv_blocks_localization_3.shape is', x.shape)
        
        x = self.tu_4(x)
        x = torch.cat((x, skips[-5]), dim=1)
        x = self.conv_blocks_localization_4(x)
        # print('conv_blocks_localization_4.shape is', x.shape)
        # print(x.shape)
        # quit()
        ds_feats.append(x)
        
        
        multi_scale_features = []
        ms_pixel_feats = ds_feats[-4:-1]
        
        f_0 = self.input_proj_0(ms_pixel_feats[0])
        # print('input_proj_0.shape is', f_0.shape)
        multi_scale_features.append(f_0)
        f_1 = self.input_proj_1(ms_pixel_feats[1])
        # print('input_proj_1.shape is', f_1.shape)
        multi_scale_features.append(f_1)
        f_2 = self.input_proj_2(ms_pixel_feats[2])
        multi_scale_features.append(f_2)
        # print('input_proj_2.shape is', f_2.shape)
        
        transformer_decoder_in_feature =  multi_scale_features  # feature pyramid
        
        mask_features = self.linear_mask_features(ds_feats[-1]) # following SingleScale
        # print('self.mask_features :', mask_features.shape)
        
        predictions = self.predictor(transformer_decoder_in_feature, mask_features, mask=None)
        # print('self.predictions :', len(predictions),predictions.keys())
        return predictions
        
            
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
        print('conv.shape is', x.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class TransUnet3D_Mip(nn.Module):
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

        super(TransUnet3D_Mip, self).__init__()
        # self.device = device
        self.seg3d = TransUnet3d()
        # self.mip_num = mip_num
        self.mip_model0 = Unet_Mip(1, 2)
        
    def forward(self, img3d, mip0, mip_index):
        
        pred_soft, feat_list = self.mip_model0(mip0)
        
        print('feat_list', len(feat_list))
        feature = feat_list
        
        depth = 64 
        argmax_bool = np.zeros((192,192,64))
        
        mip_index_all = np.load('/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_path/IXI035-IOP-0873-MRA_0.npy')
        mip_index = mip_index_all[192:384, 192:384]
        print('mip_index', mip_index.shape, np.unique(mip_index))
        
        # mip_index = np.expand_dims(mip_index, axis = 0)
        # mip_index = np.expand_dims(mip_index, axis = 0)
        # print('mip_index', mip_index.shape)
        
        argmax_bool = (np.arange(depth) == mip_index[...,None])
        print('argmax_bool', argmax_bool.shape, np.unique(argmax_bool))
        
        index_1 = np.where(mip_index==5)
        print(index_1, len(index_1[0]))
        
        argmax_bool_4 = argmax_bool[:,:,5]
        index_4 = np.where(argmax_bool_4==True)
        print(index_4, len(index_4[0]))
        
        feature_num = [64, 64, 32, 16, 16, 16, 16, 16, 32, 64, 64]
        ''' conv0, conv1, conv2, conv3, conv4, conv5, deconv5, deconv4, deconv3, deconv2, deconv1 
        feature = {'0':[conv0, deconv_1], '1':[conv1, deconv_2], '2':[conv2, deconv_3],
            '3':[conv3, deconv_4], '4':[conv4, deconv_5], '5':[conv5]}
        '''
        k = 0
        index_list = []
        twod_2_threed_feat = []
        for num in feature_num:
            if k <= 5:
                ga = math.pow(2, k )
                new_mip_index = zoom(copy.deepcopy(mip_index), (1 / ga, 1 / ga), order=1).astype(np.int64)
                print('!!!!!!!aa, new_mip_index are', new_mip_index.shape, np.unique(new_mip_index))
                new_mip_index = (new_mip_index // ga).astype(np.int64)
            if k>5:
                ga = math.pow(2, 10-k )
                new_mip_index = zoom(copy.deepcopy(mip_index), (1 / ga, 1 / ga), order=1).astype(np.int64)
                print('!!!!!!!aa, new_mip_index are', new_mip_index.shape, np.unique(new_mip_index))
                new_mip_index = (new_mip_index // ga).astype(np.int64)
            index_torch = torch.zeros((num, *new_mip_index.shape))
            for i in range(0, num):
                index_torch[i, :, :] = torch.from_numpy(copy.deepcopy(new_mip_index))
            index_torch = index_torch.long().unsqueeze(-3)
            print('#####################k, num, index_torch are', k, num, index_torch.shape)
            index_list.append(index_torch)
            
            sizea = feature[k].shape[-2:]
            print('sizea', sizea, sizea[0],sizea[1])
            argmax_bool_feat = np.zeros((sizea[0], sizea[1], num),)
            print('argmax_bool_feat1', argmax_bool_feat.shape)
            argmax_bool_feat = (np.arange(num) == new_mip_index[...,None])
            print('argmax_bool_feat2', argmax_bool_feat.shape)
            argmax_bool_feat = np.swapaxes(argmax_bool_feat, 0, 2)
            argmax_bool_feat = torch.from_numpy(argmax_bool_feat)
            argmax_bool_feat = argmax_bool_feat.unsqueeze(0).unsqueeze(0).cuda()
            print('argmax_bool_feat3', argmax_bool_feat.shape, feature[k].shape)
            feature_k = feature[k].unsqueeze(2)
            feature_k = feature_k.repeat(1, 1, num, 1, 1) 
            print('feature_k', feature_k.shape,)
            new_fea = feature_k* argmax_bool_feat
            print('new_fea', new_fea.shape, torch.unique(new_fea))
            print()
            twod_2_threed_feat.append(new_fea)
            k += 1
            
        # new_dip_index = zoom(copy.deepcopy(mip_index), (1 / ga, 1 / ga), order=1).astype(np.int64)
        result3d = self.seg3d(img3d, twod_2_threed_feat)
        
        
        quit()
     
        
        # mip_list = [mip0, mip1, mip2]
        feature_list = []
        feature_one = {}
        print('mip_indexall_shape',len(mip_index))
        print('feature.shape', feature['0'][0].shape)
        for jj in range(0, 6):
            now_index = mip_index[dim][jj]
            print('mip_index[0][0]:',dim,jj,mip_index[dim][jj].shape)
            now_index = now_index.unsqueeze(-1) 
            if jj == 0:
                print('now_index:',now_index.shape)
                print('feature[0][0]:',feature['0'][0].shape)
                feature_one['0'] = feature['0'][0].gather(dim-3, now_index).squeeze(-1)
                print('output_fea:',feature_one['0'].shape)
            elif jj == 5:
                print('now_index:',now_index.shape)
                print('feature[0][0]:',feature['5'][0].shape)
                feature_one['5'] = feature['5'][0].gather(dim-3, now_index).squeeze(-1)
                print('output_fea:',feature_one['5'].shape)
            else:
                print('now_index:',now_index.shape)
                print('feature[str(jj)[0]:',feature[str(jj)][0].shape)
                one = feature[str(jj)][0].gather(dim - 3, now_index).squeeze(-1)
                two = feature[str(jj)][1].gather(dim - 3, now_index).squeeze(-1)
                print('output_fea:',one.shape, two.shape)
                feature_one[str(jj)] = [one, two]
        feature_list.append(feature_one)
        mip_result_list, feature_list_result = [], []
        for dim in range(0, self.mip_num):
            mip_result = self.mip_model0(mip0[dim], feature_list[dim])
            mip_result_list.append(mip_result)
            feature_list_result.append(feature_list[dim]['0'])
        return result3d, feature_list_result, mip_result_list



# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES']='1'
#     trans_unet_3d_mip = TransUnet3D_Mip()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     trans_unet_3d_mip = trans_unet_3d_mip.cuda()
#     # TransU3d = trans_unet_3d.cuda()
#     input_tensor = torch.randn(1, 1, 64, 192, 192).cuda()
#     # TransU3d = trans_unet_3d.cuda()
#     mip0 = torch.randn(1, 1, 192, 192).cuda()
#     mip_index0 = torch.randn(1, 1, 192, 192).cuda()

#     pred_soft, feat = trans_unet_3d_mip(input_tensor, mip0, mip_index0)
#     print(pred_soft.shape)


    # out = trans_unet_3d(input_tensor)
    # print('!!', out['pred_logits'].shape)
    # print('!!', out['pred_masks'].shape)
    # print('!!', len(out['aux_outputs']))
    # for iii in range(len(out['aux_outputs'])):
    #     print(iii, out['aux_outputs'][iii]['pred_logits'].shape)
    #     print(iii, out['aux_outputs'][iii]['pred_masks'].shape)
    # print('out.shape is',out.shape)
    



# class StackedConvLayers(nn.Module):
#     def __init__(self, input_feature_channels, output_feature_channels, num_convs,
#                  conv_op=nn.Conv2d, conv_kwargs=None,
#                  norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
#                  dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
#                  nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
#         '''
#         stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
#         :param input_feature_channels:
#         :param output_feature_channels:
#         :param num_convs:
#         :param dilation:
#         :param kernel_size:
#         :param padding:
#         :param dropout:
#         :param initial_stride:
#         :param conv_op:
#         :param norm_op:
#         :param dropout_op:
#         :param inplace:
#         :param neg_slope:
#         :param norm_affine:
#         :param conv_bias:
#         '''
#         self.input_channels = input_feature_channels
#         self.output_channels = output_feature_channels

#         if nonlin_kwargs is None:
#             nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
#         if dropout_op_kwargs is None:
#             dropout_op_kwargs = {'p': 0.5, 'inplace': True}
#         if norm_op_kwargs is None:
#             norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
#         if conv_kwargs is None:
#             conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

#         self.nonlin_kwargs = nonlin_kwargs
#         self.nonlin = nonlin
#         self.dropout_op = dropout_op
#         self.dropout_op_kwargs = dropout_op_kwargs
#         self.norm_op_kwargs = norm_op_kwargs
#         self.conv_kwargs = conv_kwargs
#         self.conv_op = conv_op
#         self.norm_op = norm_op

#         if first_stride is not None:
#             self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
#             self.conv_kwargs_first_conv['stride'] = first_stride
#         else:
#             self.conv_kwargs_first_conv = conv_kwargs

#         super(StackedConvLayers, self).__init__()
#         self.blocks = nn.Sequential(
#             *([ConvDropoutNormNonlin(input_feature_channels, output_feature_channels, self.conv_op,
#                            self.conv_kwargs_first_conv,
#                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
#                            self.nonlin, self.nonlin_kwargs)] +
#               [ConvDropoutNormNonlin(output_feature_channels, output_feature_channels, self.conv_op,
#                            self.conv_kwargs,
#                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
#                            self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

#     def forward(self, x):
#         return self.blocks(x)
    

    
    
    