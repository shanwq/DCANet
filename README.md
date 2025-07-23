# DCANet

## Requirements and Datasets
* PyTorch : 1.11.0
* torchio : 0.18.92
* python : 3.7.0
* 
* The IXI-HH dataset is availabel at https://drive.google.com/drive/folders/16pBJAem8zfRcFqLhIxNFSlsZNa0r1UPF or https://www.dropbox.com/scl/fo/6urgp6evks6x1rsg7ck89/ABZA0sfyyGKQr2Y33Qb916Y?rlkey=9txipxphrfm75kusc8oe7s4km&st=szxf713v&dl=0

## DCANet Training:
The hyper-parameters are defined in the **hparams.py**:
* Set the path of MRA images in: "source_train_dir" and "label_train_dir";
* Set the path of Regional-MIP volumes in: "source_train_dir" and "label_train_dir";
* Set the save path for trained model in: "output_dir", if you want to load the pre-trained checkpoint, modify the "ckpt_dir"


## Notice
* You can modify **hparam.py** to determine hyper-parameters.
* Regional_MIP_generation is in the **utils** folder


## By The Way
This project is not perfect and there are still many problems. If there is some feedback, start an issue.

## Acknowledgements
This repository is an PyTorch implementation of DCANet for TOF-MRA segmentation.
