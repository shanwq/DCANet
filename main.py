import os

import random
import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
import torch.nn.functional as F
import SimpleITK as sitk

from copy import deepcopy
from torch import nn
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment

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
import json
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

random.seed(3047)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir
mip_img_train_dir = hp.mip_img_train_dir
mip_lbl_train_dir = hp.mip_lbl_train_dir


source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

output_int_dir = hp.output_int_dir
output_float_dir = hp.output_float_dir
patch_size = hp.patch_size

hpparams_dict = {key: value for key, value in hp.__dict__.items() if not key.startswith('__') and not callable(key)}

print(hpparams_dict)
os.makedirs(hp.output_dir, exist_ok=True)

with open(hp.output_dir+'/experimental_settings_train.json', 'w', encoding='utf-8') as file:
    json.dump(hpparams_dict, file, ensure_ascii=False, indent=4)


def parse_training_args(parser):
    """	
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest_checkpoint_file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup') 

    training.add_argument('--epochs', type=int, default=300, help='Number of total epochs to run')   
    training.add_argument('--epochs_per_checkpoint', type=int, default=1, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=1, help='batch-size')     #12
    training.add_argument('--sample', type=int, default=12, help='number of samples during training')    #12
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:12359", type=str, help="distributed url")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--distributed", default=False,action="store_true", help="start distributed training")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--init-lr", type=float, default=0.005, help="learning rate")   #0.001

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



def compute_loss(output, target, is_max=True, is_c2f=False, is_sigmoid=True, is_max_hungarian=True, is_max_ds=True, point_rend=False, num_point_rend=None, no_object_weight=None):
    total_loss, smooth, do_fg = None, 1e-5, False
    
    if isinstance(output, (tuple, list, dict)):
        len_ds = 1+len(output['aux_outputs']) if isinstance(output, dict) else len(output)
        max_ds_loss_weights = [1] * (len_ds) # previous had a bug with exp weight for 'v0' ..

    if is_max and is_max_hungarian:
        # output: a dict of ['pred_logits', 'pred_masks', 'aux_outputs']
        aux_outputs = output['aux_outputs'] # a list of dicts of ['pred_logits', 'pred_masks'], length is 3
        
        num_classes = 2
        target_onehot = torch.zeros_like(target.repeat(1, num_classes, 1, 1, 1), device=target.device)
        target_onehot.scatter_(1, target.long(), 1)
        target_sum = target_onehot.flatten(2).sum(dim=2) # (b, 3)
        targets = []
        for b in range(len(target_onehot)):
            target_mask = target_onehot[b][target_sum[b] > 0] # (K, D, H, W)
            target_label = torch.nonzero(target_sum[b] > 0).squeeze(1) # (K)
            targets.append({'labels':target_label, 'masks':target_mask})
        from loss_function import HungarianMatcher3D, compute_loss_hungarian
        cost_weight = [2.0, 5.0, 5.0]
        matcher = HungarianMatcher3D(
                cost_class=cost_weight[0], # 2.0
                cost_mask=cost_weight[1],
                cost_dice=cost_weight[2],
            )
        outputs_without_aux = {k: v for k, v in output.items() if k != "aux_outputs"}
        loss_list = []
        loss_final = compute_loss_hungarian(outputs_without_aux, targets, 0, matcher, 2, point_rend, num_point_rend, no_object_weight=no_object_weight, cost_weight=cost_weight)
        loss_list.append(max_ds_loss_weights[0] * loss_final)
        if is_max_ds and "aux_outputs" in output:
            for i, aux_outputs in enumerate(output["aux_outputs"][::-1]): # reverse order
                loss_aux = compute_loss_hungarian(aux_outputs, targets, i+1, matcher, 2, point_rend, num_point_rend, no_object_weight=no_object_weight, cost_weight=cost_weight)
                loss_list.append(max_ds_loss_weights[i+1] *loss_aux)
        total_loss = (loss_list[0] + sum(loss_list[1:])/len(loss_list[1:])) / 2

        # quit()
        return total_loss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    # print('intersect', torch.sum(intersect))
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    # print('y_sum, z_sum', torch.sum(y_sum), torch.sum(z_sum))
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # from weights_init.weight_init_normal import 

    from data_function_mip import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)
    if hp.mode == '3d':
        from models.three_d.DCANet_3scale_MIFB_DSGFormer import DCANet
        model = DCANet()#input_channels=hp.in_class, out_channels=hp.out_class, init_features=8)#, base_n_filter=2)  #2

    model.apply(weights_init_normal)

    para = list(model.parameters())#+list(model_inv.parameters())
    optimizer = torch.optim.AdamW(para, lr=0.0003, weight_decay=3e-05) # 0.0003

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs= 50, max_epochs= hp.total_epochs, warmup_start_lr=0.000003, eta_min= 0)
    
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(hp.ckpt_dir)
        # print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(hp.ckpt_dir, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"], strict=False)
        # optimizer.load_state_dict(ckpt["optim"])

        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cuda()

        scheduler.load_state_dict(ckpt["scheduler"])
        # elapsed_epochs = ckpt["epoch"]
        elapsed_epochs = 0
    else:
        elapsed_epochs = 0

    model.cuda()

    writer = SummaryWriter(args.output_dir)
    print(source_train_dir, label_train_dir)
    
    # quit()
    train_dataset = MedData_train(source_train_dir,label_train_dir, mip_img_train_dir, mip_lbl_train_dir, patch_size)
    train_loader = DataLoader(train_dataset.queue_dataset, 
                            batch_size=1,
                            num_workers=8, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    
    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)


    print(args.output_dir)
    torch.cuda.empty_cache()
    # quit()
    for epoch in range(1, epochs + 1):
        epoch += elapsed_epochs

        for i, batch in enumerate(train_loader):
            t_start=time.time()
            # quit()
            optimizer.zero_grad()

            if (hp.in_class == 1) and (hp.out_class == 1) :
                x = batch['source']['data']
                y = batch['label']['data']
                mip_img = batch['mip_img']['data']
                mip_lbl = batch['mip_lbl']['data']
                
                x = torch.transpose(x, 2, 4)
                y = torch.transpose(y, 2, 4)
                mip_img = torch.transpose(mip_img, 2, 4)
                mip_lbl = torch.transpose(mip_lbl, 2, 4)

                y = y.type(torch.FloatTensor).cuda()
                x = x.type(torch.FloatTensor).cuda()
                mip_img = mip_img.type(torch.FloatTensor).cuda()
                mip_lbl = mip_lbl.type(torch.FloatTensor).cuda()


            pred_mip, outputs = model(x, mip_img)
            
            is_max, is_c2f, is_sigmoid, is_max_hungarian = True, False, True, True
            is_max_ds, point_rend, num_point_rend, no_object_weight = True, False, None, None
            
            loss_3d = compute_loss(outputs, y, is_max, is_c2f, is_sigmoid, is_max_hungarian, is_max_ds, point_rend, num_point_rend, no_object_weight)
            loss_mip = dice_loss(pred_mip, mip_lbl)
            loss = 0.2*loss_mip + loss_3d
            print(f'Batch: {i}/{len(train_loader)} epoch {epoch}, loss:{loss:06f}, loss_3d:{loss_3d:06f}, miploss:{loss_mip:06f}, lr:{scheduler._last_lr[0]}, time:{time.time()-t_start}')

            # ########################################################################################################################
            
            ##########################################################################################################################
            # num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1

            ## log
            writer.add_scalar('Training/Loss', loss.item(),iteration)
            writer.add_scalar('Training/Loss_3d', loss_3d.item(),iteration)
            writer.add_scalar('Training/Loss_mip', loss_mip.item(),iteration)
            writer.add_scalar('Training/lr', scheduler._last_lr[0],iteration)
            
            # writer.add_scalar('Training/false_positive_rate', false_positive_rate,iteration)
            # writer.add_scalar('Training/false_negtive_rate', false_negtive_rate,iteration)
            # writer.add_scalar('Training/dice', dice,iteration)
            torch.cuda.empty_cache()
            

        scheduler.step()


        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )

        print(os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"))


        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
        
    
    writer.close()


if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    # elif hp.train_or_test == 'test':
        # test()
