import torch
import numpy as np
from scipy import ndimage

import tqdm
import os
join = os.path.join

import skimage 
import skimage.morphology as skmor
from skimage.morphology import square, cube, disk, ball

def get_tsp_map(pred_skel,neigh_r=2):
    """
    All computations can be reimplemented using PyTorch.
    """
    # pred_skel = skmor.skeletonize(mask_pred, method='lee')
    
    kernel = square(3)
    kernel[1,1] = 0
    pred_kp = ndimage.convolve(pred_skel, kernel, mode='constant', cval=0.0)

    pred_kp = ( np.maximum(0, pred_kp -2) + np.maximum(0, 2 - pred_kp )) * pred_skel

    negih_kernel = disk(neigh_r)
    pred_tsp = ndimage.convolve(pred_kp, negih_kernel, mode='constant', cval=0.0)
    

    
    return pred_tsp


def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def soft_skeletonize3d(x, thresh_width=14):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    kernel_size = (3, 3, 3)
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool3d(x * -1, kernel_size, 1, 1) * -1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x, kernel_size, 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)

    return x

if __name__ == "__main__":
    mask0 = "xxx"

    # contraction function
    # 0.5^(1/4) = 0.841; 0.5^(1/8) = 0.917;
    # 0.01^(1/64) =0.965; 0.01^(1/32) =0.866 ;0.01^(1/16) =0.75

    # mask0 = np.power(mask0,1/32 )

    mask_t = np.expand_dims(mask0,axis=0) 
    skel02 = soft_skeletonize(torch.from_numpy(mask_t), thresh_width=10)
    skel02 = np.squeeze(skel02.numpy())
    tsp_map = get_tsp_map(skel02)
    