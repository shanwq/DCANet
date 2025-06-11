class hparams:
    
    fold = 'seed_3047'#0 # defalut = 0
    train_or_test = 'train'
    ckpt_dir = ''
    output_dir = ''
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
    # patch_size = 128,128,128 # if 2D: 128,128,1 
    patch_size = 128,128,64 # if 2D: 128,128,1 

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'
    data_description = ''
    
    source_train_dir = './train/img'
    label_train_dir = './train/lbl'
    mip_img_train_dir = './train/mip_img'
    mip_lbl_train_dir = './train/mip_lbl'
    
    source_test_dir = './test/img'
    label_test_dir = './test/lbl'
    mip_img_test_dir = './test/mip_img'
    mip_lbl_test_dir = './test/mip_lbl'
 
 
    grad_clip = 0.3
    r1_lambda = 0.5

    output_int_dir = './pred'
    # output_float_dir = '/pred'

    init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]
    cost_weight = [2.0, 5.0, 5.0]

