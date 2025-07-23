# DCANet

## Requirements and Datasets
* PyTorch : 1.11.0
* torchio : 0.18.92
* python : 3.7.0
* 
* The IXI-HH dataset is availabel at https://drive.google.com/drive/folders/16pBJAem8zfRcFqLhIxNFSlsZNa0r1UPF or https://www.dropbox.com/scl/fo/6urgp6evks6x1rsg7ck89/ABZA0sfyyGKQr2Y33Qb916Y?rlkey=9txipxphrfm75kusc8oe7s4km&st=szxf713v&dl=0

## DCANet Training:
### 1.Regional-MIP Generation
* Regional_MIP_generation code is the file `Regional_Mip_generate.py` in the **utils** folder
* Set the projection depth (such as 15 slices)
* Set the MRA image path in `img_path`; the MRA label path in `lbl_path`
* Set the save path for the generated Regional-MIP volumes in `mip_img_15slice_path`, labels in `mip_lbl_15slice_path`, index matrix in `mip_arg_15slice_path`
* then run the code：
```
cd .../DCANet/utils
python Regional_Mip_generate.py
```

### 2.Training:
The hyper-parameters are defined in the **hparams.py**:
* Set the path of MRA images in: `source_train_dir` and `label_train_dir`;
* Set the path of Regional-MIP volumes in: `mip_img_train_dir` and `mip_lbl_train_dir`;
* Set the save path for trained model in: `output_dir`;
* then run the code：
```
cd .../DCANet
python train.py --epoches 300
```
### 3.Testing:
The hyper-parameters are defined in the **hparams.py**:
* Set the path of MRA images in: `source_test_dir`;
* Set the path of Regional-MIP volumes in: `mip_img_test_dir`;
* Set the save path for prediction results in: `output_int_dir`;
* Set the model path in `ckpt_dir`;
* then run the code：
```
python test.py --ckpt_dir dcanet.pt
```

## By The Way
There are still many problems in this project. If you have questions, start an issue.

## Acknowledgements
This repository is an PyTorch implementation of DCANet for TOF-MRA segmentation.
