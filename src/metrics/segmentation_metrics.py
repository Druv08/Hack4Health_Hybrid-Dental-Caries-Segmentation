import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_score(pred, gt):
    smooth = 1e-6
    intersection = (pred * gt).sum()
    return (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

def iou_score(pred, gt):
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / (union + 1e-6)

def pixel_accuracy(pred, gt):
    return (pred == gt).mean()

def sensitivity(pred, gt):
    tp = ((pred == 1) & (gt == 1)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    return tp / (tp + fn + 1e-6)

def specificity(pred, gt):
    tn = ((pred == 0) & (gt == 0)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    return tn / (tn + fp + 1e-6)

def hausdorff_distance(pred, gt):
    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)
    return max(
        directed_hausdorff(pred_pts, gt_pts)[0],
        directed_hausdorff(gt_pts, pred_pts)[0]
    )
