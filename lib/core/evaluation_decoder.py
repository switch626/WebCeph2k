
import cv2
import math
import os

import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from ..utils.transforms import transform_preds, crop


def get_preds(scores):

    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds_1 = idx.repeat(1, 1, 2).float()

    preds_1[:, :, 0] = (preds_1[:, :, 0] - 1) % scores.size(3) + 1
    preds_1[:, :, 1] = torch.floor((preds_1[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds_1 *= pred_mask

    if True:
        preds = np.zeros((scores.shape[0], scores.shape[1], 2))
        for batch in range(scores.shape[0]):
            features = scores[batch, ...]
            for landmarks in range(features.shape[0]):
                x_local = int(preds_1[batch, landmarks, 0].numpy())
                y_local = int(preds_1[batch, landmarks, 1].numpy())
                delta = 2  # training = 4
                opti = 3  # training = 1

                if y_local-delta < 0:
                    y_local = delta
                if x_local-delta < 0:
                    x_local = delta
                heatmap = features[landmarks, y_local-delta:y_local+delta+opti, x_local-delta:x_local+delta+opti]

                grid_y, grid_x = torch.meshgrid(torch.arange(0, heatmap.shape[0]), torch.arange(0, heatmap.shape[1]))
                grid_y = grid_y.float()
                grid_x = grid_x.float()

                mean_y = torch.sum(grid_y * heatmap) / torch.sum(heatmap) + y_local - delta + 1
                mean_x = torch.sum(grid_x * heatmap) / torch.sum(heatmap) + x_local - delta + 1
                gt_mean_coords = torch.stack([mean_x, mean_y])               
                preds[batch, landmarks, :] = gt_mean_coords.numpy()

        return torch.from_numpy(preds)
    return preds_1

def compute_nme(preds, meta, SDR_List, Each_Point, PCK_Curve, isTest=False):

    target = meta['pts'].cpu().numpy()
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)
    comp = np.linspace(0, 5, 200) * 100

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        pts_pred_copy, pts_gt_copy = preds[i, ], target[i, ]
        Lsub_count = 0
        for cl in range(pts_pred.shape[0]):
            if meta['vis'][i][cl] == 0:
                pts_pred_copy[cl] = torch.from_numpy(np.zeros((1,2)))
                pts_gt_copy[cl] = torch.from_numpy(np.zeros((1,2)))
                Lsub_count += 1
        rmse[i] = np.sum(np.linalg.norm(pts_pred_copy - pts_gt_copy, axis=1)) / (L-Lsub_count) * meta['pixels'][i]

        for j in range(L):
            if meta['vis'][i][j] > 0:
                # calculate the every point error
                d_x = pts_pred[j,][0] - pts_gt[j,][0]
                d_y = pts_pred[j,][1] - pts_gt[j,][1]

                tmp = math.sqrt(d_x ** 2 + d_y ** 2) * meta['pixels'][i]
                Each_Point[j, 0] += tmp
                Each_Point[j, 1] += 1

                if isTest:
                    cnt = 0
                    for c_ in comp:
                        if tmp * 100.0 <= c_:
                            PCK_Curve[cnt] += 1
                        cnt += 1

                if tmp * 100. > 500:
                    SDR_List[6][0] += 1
                elif tmp * 100. > 400:
                    SDR_List[5][0] += 1
                elif tmp * 100. > 300:
                    SDR_List[4][0] += 1
                elif tmp * 100. > 250:
                    SDR_List[3][0] += 1
                elif tmp * 100. > 200:
                    SDR_List[2][0] += 1
                elif tmp * 100. > 100:
                    SDR_List[1][0] += 1
                else:
                    SDR_List[0][0] += 1

    return rmse, SDR_List, Each_Point, PCK_Curve


def decode_preds(output, center, scale, res):
    coords = get_preds(output)  # float type
    preds = coords.cpu().clone()

    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
