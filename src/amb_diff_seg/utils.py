import torch
import numpy as np
import random
import sys
from omegaconf import OmegaConf
import os

shorthand_dict = {"combined_sensitivity": "S_c",
                    "maximum_dice_matching": "D_max",
                    "diversity_agreement": "D_a",
                    "collective_insight": "CI",
                    "generalized_energy_distance": "GED"}

def variance_ncc_dist(sample_arr, gt_arr):
    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """
    if len(gt_arr.shape) == 3:
        #expand to onehot in 3rd axis
        gt_arr = np.concatenate([1-gt_arr[...,None],gt_arr[...,None]], axis=-1)
    if len(sample_arr.shape) == 3:
        #expand to onehot in 3rd axis
        sample_arr = np.concatenate([1-sample_arr[...,None],sample_arr[...,None]], axis=-1)
    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):
        ncc_list.append(ncc(E_ss, E_sy[j,...]))
    return ncc_list

def pixel_wise_xent(m_samp, m_gt, eps=1e-8):


    log_samples = np.log(m_samp + eps)

    return -1.0*np.sum(m_gt*log_samples, axis=-1)

def ncc(a,v, zero_norm=True, eps=1e-8):
    a = a.flatten()
    v = v.flatten()
    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a)+eps)
        v = (v - np.mean(v)) / (np.std(v)+eps)
    else:
        a = (a) / (np.std(a) * len(a)+eps)
        v = (v) / (np.std(v)+eps)
    return np.correlate(a,v)

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    if seed is not None:
        if seed < 0:
            seed = None
    if seed is None:
        np.random.seed()
        seed = np.random.randint(0, 2**16-1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def generalized_energy_distance(pred,gt,dist=lambda *x: 1-binary_iou(*x),skip_same=False):
    n_gt = gt.shape[-1]
    n_pred = pred.shape[-1]
    dist_pred_gt = np.zeros((n_gt,n_pred))
    cross_mat = []
    for i in range(n_gt):
        cross_mat.append([])
        for j in range(n_pred):
            TP,FP,FN,TN = get_TP_FP_FN_TN(gt[:,:,i],pred[:,:,j])
            cross_mat[-1].append((TP,FP,FN,TN))
            dist_pred_gt[i,j] = dist(TP,FP,FN,TN)
    dist_gt = {}
    for i in range(n_gt):
        for j in range(i*skip_same,n_gt):
            TP,FP,FN,TN = get_TP_FP_FN_TN(gt[:,:,i],gt[:,:,j])
            
            dist_gt[(i,j)] = dist(TP,FP,FN,TN)
    dist_pred = {}
    for i in range(n_pred):
        for j in range(i*skip_same,n_pred):
            TP,FP,FN,TN = get_TP_FP_FN_TN(pred[:,:,i],pred[:,:,j])
            dist_pred[(i,j)] = dist(TP,FP,FN,TN)
    expected_gt = np.array(list(dist_gt.values())).mean()
    expected_pred = np.array(list(dist_pred.values())).mean()
    ged = 2*dist_pred_gt.mean()-expected_gt-expected_pred
    return ged, cross_mat

def get_TP_FP_FN_TN(pred,gt):
    TP = np.sum(gt&pred).astype(int)
    FP = np.sum(~gt&pred).astype(int)
    FN = np.sum(gt&~pred).astype(int)
    TN = np.sum(~gt&~pred).astype(int)
    return TP,FP,FN,TN

def binary_iou(TP,FP,FN,TN):
    if TP+FP+FN==0:
        return 1.0
    else:
        return TP/(TP+FP+FN)

def binary_dice(TP,FP,FN,TN):
    if TP+FP+FN==0:
        return 1.0
    else:
        return 2*TP/(2*TP+FP+FN)
    
def binary_sensitivity(TP,FP,FN,TN):
    if TP+FN==0:
        return 1.0
    else:
        return TP/(TP+FN)

def collective_insight(pred,gt):
    assert gt.max()<=1 and pred.max()<=1
    n_gt = gt.shape[-1]
    n_pred = pred.shape[-1]

    measures = {"combined_sensitivity": float("nan"),
                "maximum_dice_matching": float("nan"),
                "diversity_agreement": float("nan")}
                
    TP,FP,FN,TN = get_TP_FP_FN_TN(gt.any(-1),pred.any(-1))
    measures["combined_sensitivity"] = binary_sensitivity(TP,FP,FN,TN)

    dice_mat = np.zeros((n_gt,n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            TP,FP,FN,TN = get_TP_FP_FN_TN(gt[:,:,i],pred[:,:,j])
            dice_mat[i,j] = binary_dice(TP,FP,FN,TN)
    measures["maximum_dice_matching"] = dice_mat.max(axis=1).mean()

    variance_mat_pred = np.var(pred.reshape(-1,n_pred)[:,:,None].astype(int)-
                               pred.reshape(-1,n_pred)[:,None,:].astype(int),axis=0)
    variance_mat_gt = np.var(gt.reshape(-1,n_gt)[:,:,None].astype(int)-
                             gt.reshape(-1,n_gt)[:,None,:].astype(int),axis=0)
    V_pred_min = variance_mat_pred.min()
    V_pred_max = variance_mat_pred.max()
    V_gt_min = variance_mat_gt.min()
    V_gt_max = variance_mat_gt.max()
    delta_max = abs(V_pred_max-V_gt_max)
    delta_min = abs(V_pred_min-V_gt_min)
    measures["diversity_agreement"] = 1-(delta_max+delta_min)/2
    multiplied = np.prod([v for v in measures.values()])
    added = np.sum([v for v in measures.values()])
    measures["collective_insight"] = 3*multiplied/added
    return measures

def get_ambiguous_metrics(pred,gt,shorthand=True,reduce_to_mean=True,postprocess=False):
    """returns a dictionary of metrics for binary ambiguous segmentation"""
    assert isinstance(gt,np.ndarray), "gt must be a numpy array, found type: "+str(type(gt))
    assert isinstance(pred,np.ndarray), "pred must be a numpy array, found type: "+str(type(pred))
    assert len(gt.shape)==3, "gt must be a 3D numpy array in (H,W,C_gt) format"
    assert len(pred.shape)==3, "pred must be a 3D numpy array in (H,W,C_pred) format"
    assert gt.shape[0]==pred.shape[0], f"gt and pred must have the same height"
    assert gt.shape[1]==pred.shape[1], f"gt and pred must have the same width"
    assert gt.dtype in [bool,np.bool_], f"gt must be a boolean or uint8 array, found type: {gt.dtype}"
    assert pred.dtype in [bool,np.bool_], f"pred must be a boolean or uint8 array, found type: {pred.dtype}"

    if postprocess:
        areas = [pred[:,:,i].astype(float).mean().item() for i in range(pred.shape[2])]
        max_area = max(areas)
        pred = np.stack([pred[:,:,i]*(a>0.5*max_area) for i,a in enumerate(areas)], axis=2)

    measures = collective_insight(pred,gt)
    ged, cross_mat = generalized_energy_distance(pred,gt)
    measures["generalized_energy_distance"] = ged
    if shorthand:
        measures = {shorthand_dict[k]:v for k,v in measures.items()}
    measures["iou"] = [binary_iou(*cm_ij) for cm_ij in sum(cross_mat,[])]
    measures["dice"] = [binary_dice(*cm_ij) for cm_ij in sum(cross_mat,[])]
    measures["ncc"] = variance_ncc_dist(pred.transpose((2,0,1)),gt.transpose((2,0,1))) 
    if reduce_to_mean:
        for k,v in measures.items():
            if isinstance(v,list):
                measures[k] = np.mean(v).item()
            #convert float64 to float
            if isinstance(measures[k],np.float64):
                measures[k] = measures[k].item()
            assert isinstance(measures[k],float), f"value for k={k} must be a float, found type: {type(measures[k])}"
    return measures

def get_train_metrics(pred, gt, reduce_to_mean=True):
    metrics = {"iou": [],
               "dice": [],
               "mse": []}
    for i in range(pred.shape[0]):
        metrics["mse"].append(((pred[i]-gt[i])**2).mean().item())
        pred_i = pred[i].detach().cpu().numpy()>0.0
        gt_i = gt[i].detach().cpu().numpy()>0.0
        TP,FP,FN,TN = get_TP_FP_FN_TN(gt_i,pred_i)
        metrics["iou"].append(binary_iou(TP,FP,FN,TN))
        metrics["dice"].append(binary_dice(TP,FP,FN,TN))
    if reduce_to_mean:
        metrics = {k: np.mean(v).item() for k,v in metrics.items()}
    return metrics

def load_config():
    args = sys.argv[1:]

    if not args:
        raise ValueError("Missing config file path as the first argument.")

    config_path = args[0]

    if '=' in config_path:
        raise ValueError(f"Invalid config path '{config_path}'. The first argument must be a file path, not a key=value pair.")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(args[1:])
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg