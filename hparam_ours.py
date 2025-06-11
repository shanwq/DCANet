class hparams:
    
    fold = 'seed_3047'#0 # defalut = 0
    train_or_test = 'train'
    ckpt_dir = ''
    output_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/ExperiResult/seed_3047/TransU_Mip_submit_flops/ckpt_ck'
    # output_dir = 'logs/batch2'
    aug = False
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 300
    epochs_per_checkpoint = 1
    batch_size = 1
    ckpt = None
    init_lr = 0.0003 # 0.01
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1

    # crop_or_pad_size = 64,64,64 # if 2D: 256,256,1
    # patch_size = 192,192,64 # if 2D: 128,128,1 
    patch_size = 128,128,64 # if 2D: 128,128,1 

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'
    data_description = 'image_align_direction_hdbet_N4_corrected'
    # model_nsetting = 'image_align_direction_hdbet'
    # source_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/fold_%s/no_skull/image'%fold
    
    # label_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/fold_%s/no_skull/label'%fold
    # source_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/fold_%s/no_skull/image'%fold
    # label_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/fold_%s/no_skull/label'%fold
    
    # source_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/noskull_N4_image'
    source_train_dir = '/memory/shanwenqi/Vessel_seg/ITKTubeTK_Healthy/train/img'
    # label_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/noskull_N4_label'
    label_train_dir = '/memory/shanwenqi/Vessel_seg/ITKTubeTK_Healthy/train/lbl'
    mip_img_train_dir = '/memory/shanwenqi/Vessel_seg/ITKTubeTK_Healthy/train/mip_img_9_slice_3D'
    # mip_img_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_img_15_slice_3D'
    # mip_lbl_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_lbl_9_slice_3D'
    # mip_lbl_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_lbl_15_slice_3D'
    mip_lbl_train_dir = '/memory/shanwenqi/Vessel_seg/ITKTubeTK_Healthy/train/mip_lbl_9_slice_3D'
    
    
    source_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_image'
    label_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_label'
    mip_img_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_img_15_slice_3D'
    mip_lbl_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_15_slice_3D'
 
 
    grad_clip = 0.3
    r1_lambda = 0.5

    # output_int_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/pred/int_pred'%fold
    # output_float_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/pred/float_pred'%fold 
    output_int_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/ExperiResult/seed_3047/TransU_Mip_15slice_3scale_Mask_coef5v1/pred_from0'
    output_float_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/ExperiResult/seed_3047/TransU_Mip_15slice_3scale_coef4v1/pred'

    init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]
    cost_weight = [2.0, 5.0, 5.0]
            # matcher = HungarianMatcher3D(
            #     cost_class=cost_weight[0], # 2.0
            #     cost_mask=cost_weight[1],
            #     cost_dice=cost_weight[2],
            # )
