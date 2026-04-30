
import torch
import numpy as np
from scipy import ndimage

import tqdm
import os
join = os.path.join

import skimage 
import skimage.morphology as skmor
from skimage.measure import label, regionprops
from skimage.morphology import square, cube, disk, ball
import SimpleITK as sitk

def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def compute_tsp_dice2d(mask_gt, mask_pred,neigh_r=2):

    gt_skel =  skmor.skeletonize(mask_gt, method='lee')
    pred_skel = skmor.skeletonize(mask_pred, method='lee')
    
    kernel = square(3)
    kernel[1,1] = 0
    gt_kp = ndimage.convolve(gt_skel, kernel, mode='constant', cval=0.0)
    pred_kp = ndimage.convolve(pred_skel, kernel, mode='constant', cval=0.0)
    
    gt_kp = np.logical_or(gt_kp == 1, gt_kp >= 3) * gt_skel
    pred_kp = np.logical_or(pred_kp == 1, pred_kp >= 3) * pred_skel
    
    negih_kernel = disk(neigh_r)
    gt_tsp = ndimage.convolve(gt_kp, negih_kernel, mode='constant', cval=0.0) * mask_gt
    pred_tsp = ndimage.convolve(pred_kp, negih_kernel, mode='constant', cval=0.0) * mask_pred
    
    tsp_dice = compute_dice_coefficient(gt_tsp, pred_tsp) 
    
    return tsp_dice
    

def compute_tsp_dice3d(mask_gt, mask_pred,neigh_r=2):

    gt_skel =  skmor.skeletonize(mask_gt, method='lee')
    pred_skel = skmor.skeletonize(mask_pred, method='lee')
    
    kernel = cube(3)
    kernel[1,1,1] = 0

    gt_kp = ndimage.convolve(gt_skel, kernel, mode='constant', cval=0.0)
    pred_kp = ndimage.convolve(pred_skel, kernel, mode='constant', cval=0.0)
    
    gt_kp = np.logical_or(gt_kp == 1, gt_kp >= 3) * gt_skel
    pred_kp = np.logical_or(pred_kp == 1, pred_kp >= 3) * pred_skel
    
    negih_kernel = ball(neigh_r)
    gt_tsp = ndimage.convolve(gt_kp, negih_kernel, mode='constant', cval=0.0) * mask_gt
    pred_tsp = ndimage.convolve(pred_kp, negih_kernel, mode='constant', cval=0.0) * mask_pred
    
    tsp_dice = compute_dice_coefficient(gt_tsp, pred_tsp)
    
    return tsp_dice

