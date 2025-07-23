import numpy as np
import SimpleITK as sitk
import os



def mip_generate(n_slice):
    '''Slide Window MIP Generation with padding'''
    '''stride = 1'''
    n_slice = n_slice
    img_path = ''
    lbl_path = ''
    mip_img_15slice_path = ''
    mip_lbl_15slice_path = ''
    mip_arg_15slice_path = ''

    os.makedirs(mip_img_15slice_path, exist_ok=True)
    os.makedirs(mip_lbl_15slice_path, exist_ok=True)
    os.makedirs(mip_arg_15slice_path, exist_ok=True)
    
    for img_name in os.listdir(img_path):
        
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)

        mip_img_all = np.zeros_like(img_arr)
        mip_lbl_all = np.zeros_like(img_arr)
        mip_arg_all = np.zeros_like(img_arr)
        
        for kk in range(0, img_arr.shape[0]):
            if kk < int((n_slice-1)/2) :
                # print(kk)
                img_patch_arr = img_arr[:n_slice,:,:]
                lbl_patch_arr = lbl_arr[:n_slice,:,:]
                mip_img_n = np.max(img_patch_arr, 0)
                arg_arr = np.argmax(img_patch_arr, 0)
                # print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
                right_mip_label_arr = np.zeros_like(arg_arr)
                for i in range(0, arg_arr.shape[0]):
                    for j in range(0, arg_arr.shape[1]):
                        right_mip_label_arr[i,j] = lbl_patch_arr[arg_arr[i][j], i, j]
                right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            elif kk > ((img_arr.shape[0] - int((n_slice-1)/2))-1):
                # print(kk)
                img_patch_arr = img_arr[-1*n_slice:,:,:]
                lbl_patch_arr = lbl_arr[-1*n_slice:,:,:]
                mip_img_n = np.max(img_patch_arr, 0)
                arg_arr = np.argmax(img_patch_arr, 0)
                # print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
                right_mip_label_arr = np.zeros_like(arg_arr)
                for i in range(0, arg_arr.shape[0]):
                    for j in range(0, arg_arr.shape[1]):
                        right_mip_label_arr[i,j] = lbl_patch_arr[arg_arr[i][j], i, j]
                right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            else:
                img_patch_arr = img_arr[kk-int((n_slice-1)/2): kk+int((n_slice-1)/2)+1]
                lbl_patch_arr = lbl_arr[kk-int((n_slice-1)/2): kk+int((n_slice-1)/2)+1]
                mip_img_n = np.max(img_patch_arr, 0)
                arg_arr = np.argmax(img_patch_arr, 0)
                # print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
                right_mip_label_arr = np.zeros_like(arg_arr)
                for i in range(0, arg_arr.shape[0]):
                    for j in range(0, arg_arr.shape[1]):
                        right_mip_label_arr[i,j] = lbl_patch_arr[arg_arr[i][j], i, j]
                right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            
            mip_img_all[kk,:,:] = mip_img_n
            mip_lbl_all[kk,:,:] = right_mip_label_arr
            mip_arg_all[kk,:,:] = arg_arr
            
        mip_img = sitk.GetImageFromArray(mip_img_all)
        mip_lbl = sitk.GetImageFromArray(mip_lbl_all)
        mip_arg = sitk.GetImageFromArray(mip_arg_all)
        mip_img.CopyInformation(img)
        mip_lbl.CopyInformation(img)
        mip_arg.CopyInformation(img)
            
        sitk.WriteImage(mip_img, os.path.join(mip_img_15slice_path, img_name))#'%s'%(img_name[:-7]) + 'img_slice_%s.nii.gz'%n_))
        sitk.WriteImage(mip_lbl, os.path.join(mip_lbl_15slice_path, img_name))#'%s'%(img_name[:-7]) + 'lbl_slice_%s.nii.gz'%kk))
        sitk.WriteImage(mip_arg, os.path.join(mip_arg_15slice_path, img_name))#'%s'%(img_name[:-7]) + 'lbl_slice_%s.nii.gz'%kk))

if __name__ =='__main__':
    n_slice = 15 # projection depth / spacing
    mip_generate(n_slice)
