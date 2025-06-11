from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import time
import csv
import SimpleITK as sitk
import torch
# from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler,LabelSampler,WeightedSampler
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
from pathlib import Path

from hparam_ours import hparams as hp


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, mip_img_dir, mip_lbl_dir, patch_size):

        patch_size = patch_size
            # patch_size = 512, 512, 46
        queue_length = 10
        samples_per_volume = 10

        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))
            mip_img_dir = Path(mip_img_dir)
            self.mip_img_paths = sorted(mip_img_dir.glob(hp.fold_arch))
            mip_lbl_dir = Path(mip_lbl_dir)
            self.mip_lbl_paths = sorted(mip_lbl_dir.glob(hp.fold_arch))
            

            for (image_path, label_path, mip_img_path, mip_lbl_path) in zip(self.image_paths, self.label_paths, self.mip_img_paths, self.mip_lbl_paths):
                # start_time = time.time()
                # source=tio.ScalarImage(image_path),
                # print(source)
                # source_arr = source[0].data
                # print(source_arr.shape)#, source_arr)
                # source_arr_transpose = torch.transpose(source_arr, 1,3)
                # print(source_arr_transpose.shape)
                # source[0].set_data(source_arr_transpose)
                # print(source)
                # print('耗时',time.time()-start_time)
                
                # aa = sitk.ReadImage(image_path)
                # aa_arr = sitk.GetArrayFromImage(aa)
                # aa_arr = np.expand_dims(aa_arr, axis=0)
                # print(aa_arr.shape,)
                # source[0].set_data(aa_arr)
                # print(source)
                
                # label=tio.LabelMap(label_path),
                # label_arr = label.data[0]
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                # print('耗时',time.time()-start_time)
                # label_arr = label.data[0]
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                subject = tio.Subject(
                    source = tio.ScalarImage(image_path),
                    label = tio.LabelMap(label_path),
                    mip_img = tio.ScalarImage(mip_img_path),
                    mip_lbl = tio.LabelMap(mip_lbl_path),
                )
                self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            #UniformSampler(patch_size),
            # WeightedSampler(patch_size, 'label'),
            LabelSampler(patch_size),
        ) 




    def transform(self):


        training_transform = Compose([
        #ToCanonical(),
        #CropOrPad((hp.crop_or_pad_size, hp.crop_or_pad_size, hp.crop_or_pad_size), padding_mode='reflect'),
        #RandomMotion(),
        #RandomBiasField(),
        ZNormalization(),
        #RandomNoise(),
        # RandomFlip(axes=(0,)),
        #OneOf({
        #    RandomAffine(): 0.8,
        #    RandomElasticDeformation(): 0.2,
        #}),         
        ])

        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):


        self.subjects = []
        self.name_list = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                
                # source=tio.ScalarImage(image_path),
                # source_arr = source.data
                # source_arr_transpose = source_arr.transpose(1,3)
                # source.set_data(source_arr_transpose)
                
                # label=tio.LabelMap(label_path),
                # label_arr = label.data
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
                self.name_list.append(str(image_path).split('/')[-1])
                
        # self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)




