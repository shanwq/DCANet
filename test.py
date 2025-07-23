import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import math
import torchio
import torch.nn.functional as F
import json
from copy import deepcopy
from torch import nn
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from glob import glob
import SimpleITK as sitk
from typing import Tuple, List

from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
# from medpy.io import load,save
from tqdm import tqdm
from torchvision import utils
# from hparam import hparams as hp
from hparam_ours import hparams as hp
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init_normal(m):
    classname = m.__class__.__name__
    gain = 0.02
    init_type = hp.init_type

    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


# MRA paths
source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir
# MIP paths
source_test_dir = hp.source_test_dir
mip_img_test_dir = hp.mip_img_test_dir
# save path for predictions
output_int_dir = hp.output_int_dir
patch_size = 128, 128, 128

hpparams_dict = {key: value for key, value in hp.__dict__.items() if not key.startswith('__') and not callable(key)}

from models.three_d.DCANet_3scale_MIFB_DSGFormer import DCANet
model = DCANet()#input_channels=hp.in_class, out_channels=hp.out_class, init_features=8)#, base_n_filter=2)  #2

print(hpparams_dict)
os.makedirs(hp.output_dir, exist_ok=True)
# Put the hpparams_dict in the .json file
with open(hp.output_dir+'/experimental_settings_test.json', 'w', encoding='utf-8') as file:
    json.dump(hpparams_dict, file, ensure_ascii=False, indent=4)

def parse_training_args(parser):
    """	
    """
    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest_checkpoint_file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')
    # training
    training = parser.add_argument_group('training setup') 
    training.add_argument('--epochs_per_checkpoint', type=int, default=1, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=1, help='batch-size')     #12
    training.add_argument('--sample', type=int, default=12, help='number of samples during training')    #12

    parser.add_argument(
        '-k',
        "--ckpt_dir",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMPa')
    training.add_argument('--cudnn-enabled', default=True, help='En	able cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')


    return parser

def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...],
                                      image_size: Tuple[int, ...],
                                      step_size: float) -> List[List[int]]:
    # compute how many windows needed for testing the volume
                                          
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)

    return steps

def predict(arr, mip_arr):

    prob_map = torch.zeros((1, 2,) + arr.shape[-3:]).half().cuda()
    cnt_map = torch.zeros_like(prob_map)
    arr_clip = np.clip(arr, -9999, 9999)
    mip_arr_clip = np.clip(mip_arr, -9999, 9999)
    # Normalization norm and mean
    raw_norm = (arr_clip - arr_clip.mean()) / (arr_clip.std()+ 1e-8)
    raw_mip_norm = (mip_arr_clip - mip_arr_clip.mean()) / (mip_arr_clip.std()+ 1e-8)

    step_size = 0.6
    steps = _compute_steps_for_sliding_window(patch_size, raw_norm.shape[-3:], step_size) 
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    kk=0
    # patch-based prediction
    for x in steps[0]:
        lb_x = x
        ub_x = x + patch_size[0]
        for y in steps[1]:
            lb_y = y
            ub_y = y + patch_size[1]
            for z in steps[2]:
                lb_z = z
                ub_z = z + patch_size[2]
                with torch.no_grad():
                    numpy_arr = raw_norm[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis] if len(raw_norm.shape)==4 else raw_norm[lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis, np.newaxis]
                    numpy_mip_arr = raw_mip_norm[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis] if len(raw_mip_norm.shape)==4 else raw_mip_norm[lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis, np.newaxis]
                    
                    tensor_arr = torch.from_numpy(numpy_arr).cuda()
                    mip_img = torch.from_numpy(numpy_mip_arr).cuda()
                    pred_mip, seg_pro = model(tensor_arr, mip_img) # (1, c, d, h, w)
                    
                    mask_cls, mask_pred = seg_pro["pred_logits"], seg_pro["pred_masks"]
                    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # filter out non-object class
                    mask_pred = mask_pred.sigmoid()
                    seg_pro = torch.einsum("bqc,bqdhw->bcdhw", mask_cls, mask_pred)
                    _pred = seg_pro

                    prob_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += _pred
                    # NOTE: should also smooth cnt_map if apply gaussian_mask before |  neural_network.py -> network.predict_3D -> _internal_predict_3D_3Dconv_tiled
                    cnt_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += 1
                    
                kk=kk+1

    torch.cuda.empty_cache()
    return prob_map.detach().cpu()


def Inference3D(input_img_path, mip_path, save_path=None):
    '''
    read, inference and save the prediction results for the test case.
    '''
    sitk_raw = sitk.ReadImage(input_img_path)
    arr_raw = sitk.GetArrayFromImage(sitk_raw)
    sitk_mip = sitk.ReadImage(mip_path)
    arr_mip = sitk.GetArrayFromImage(sitk_mip)
    
    origin_spacing = sitk_raw.GetSpacing()
    origin = sitk_raw.GetOrigin()
    direction = sitk_raw.GetDirection()
    rai_size = sitk_raw.GetSize()
    
    pad_flag = 0
    
    with torch.no_grad():
        prob_map = predict(arr_raw, arr_mip) #torch.Size([1, 2, 92, 1024, 1024])


    prob_map_interp = np.zeros(list(prob_map.size()[:2]) + list(sitk_raw.GetSize()[::-1]), dtype=np.float16)
    for i in range(prob_map.size(1)):
        prob_map_interp[:, i] = F.interpolate(prob_map[:, i:i + 1].cuda().float(),
                                                size=sitk_raw.GetSize()[::-1],
                                                mode="trilinear").detach().half().cpu().numpy()
    del prob_map

    segmentation = np.argmax(prob_map_interp.squeeze(0), axis=0)
    
    save_dir = save_path +input_img_path.split('/')[-1]
    os.makedirs(save_path, exist_ok = True)
    del prob_map_interp
    # segmentation = np.swapaxes(segmentation, 0,2)
    pred_sitk = sitk.GetImageFromArray(segmentation.astype(np.int8))
    pred_sitk.SetDirection(direction)
    pred_sitk.SetOrigin(origin)
    pred_sitk.SetSpacing(origin_spacing)
    
    print(save_dir)
    sitk.WriteImage(pred_sitk, save_dir)



def test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    
    
    ckpt = torch.load(os.path.join(args.output_dir, args.ckpt_dir), map_location=lambda storage, loc: storage)
    model.cuda()
    model.eval()

    model.load_state_dict(ckpt["model"])

    raw_data_dir = source_test_dir
    raw_data_dir_list = sorted(glob(raw_data_dir+"/*.nii.gz"))
    raw_mip_dir_list = sorted(glob(mip_img_test_dir+"/*.nii.gz"))

    for i, (raw_img_path, mip_path) in enumerate(tqdm(zip(raw_data_dir_list, raw_mip_dir_list))):
        print(output_int_dir + raw_img_path.split('/')[-1])
        # quit()
        if os.path.exists(output_int_dir + raw_img_path.split('/')[-1]):
            continue
        Inference3D(raw_img_path, mip_path, output_int_dir_new)
    print('inference done!')   

if __name__ == '__main__':
    test()
