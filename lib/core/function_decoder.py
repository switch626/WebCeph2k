
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np

from .evaluation_decoder import decode_preds, compute_nme

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn import model_selection
import numpy as np

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    num_classes = config.MODEL.NUM_JOINTS

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    SDR_List = np.zeros([7, 2])
    Each_Point = np.zeros([num_classes, 2])
    PCK_Curve = np.zeros([200])

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):

        data_time.update(time.time()-end)

        output = model(inp)

        target = target.cuda(non_blocking=True)

        loss = 0

        for s0 in range(output.shape[0]):  
            for s1 in range(output.shape[1]):  
                if meta['vis'][s0][s1] > 0:  
                    loss += critertion(output[s0][s1], target[s0][s1])

        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)

        nme_batch, SDR_List, Each_Point, PCK_Curve = compute_nme(preds, meta, SDR_List, Each_Point, PCK_Curve, False)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)

def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    SDR_List = np.zeros([7, 2])
    Each_Point = np.zeros([num_classes, 2])
    PCK_Curve = np.zeros([200])

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)
            score_map = output.data.cpu()

            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)
   
            nme_temp, SDR_List, Each_Point, PCK_Curve = compute_nme(preds, meta, SDR_List, Each_Point, PCK_Curve, True)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.shape[0]
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Valid Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    sum_total = 0
    for idx in range(SDR_List.shape[0]):
        sum_total += SDR_List[idx][0]

    thr_100 = round(SDR_List[0][0] / sum_total * 100.0, 2)
    thr_200 = round((SDR_List[0][0] + SDR_List[1][0]) / sum_total * 100.0, 2)
    thr_250 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0]) / sum_total * 100.0, 2)
    thr_300 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0] + SDR_List[3][0]) / sum_total * 100.0, 2)
    thr_400 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0] + SDR_List[3][0] + SDR_List[4][0]) / sum_total * 100.0, 2)
    thr_500 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0] + SDR_List[3][0] + SDR_List[4][0] + SDR_List[5][0]) / sum_total * 100.0, 2)

    fold = '{} {} {} {} {} {}'.format(thr_100, thr_200, thr_250, thr_300, thr_400,thr_500)
    logger.info(str(fold))

    each_point_str = ' '
    avg_ = 0
    avg_count = 0
    for idx in range(Each_Point.shape[0]):  
        each_point_str += '{:.2f} '.format(Each_Point[idx, 0] / Each_Point[idx, 1])  
        avg_ += Each_Point[idx, 0] # * Each_Point[idx, 1] 
        avg_count += Each_Point[idx, 1] 

    nme = avg_/avg_count

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    SDR_List = np.zeros([7, 2])
    Each_Point = np.zeros([num_classes, 2])
    PCK_Curve = np.zeros([200])

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)

            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)

            nme_temp, SDR_List, Each_Point, PCK_Curve = compute_nme(preds, meta, SDR_List, Each_Point, PCK_Curve, True)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.shape[0]
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]  

            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    sum_total = 0
    for idx in range(SDR_List.shape[0]):
        sum_total += SDR_List[idx][0]

    thr_100 = round(SDR_List[0][0] / sum_total * 100.0, 2)
    thr_200 = round((SDR_List[0][0] + SDR_List[1][0]) / sum_total * 100.0, 2)
    thr_250 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0]) / sum_total * 100.0, 2)
    thr_300 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0] + SDR_List[3][0]) / sum_total * 100.0, 2)
    thr_400 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0] + SDR_List[3][0] + SDR_List[4][0]) / sum_total * 100.0, 2)
    thr_500 = round((SDR_List[0][0] + SDR_List[1][0] + SDR_List[2][0] + SDR_List[3][0] + SDR_List[4][0] + SDR_List[5][0]) / sum_total * 100.0, 2)

    fold = '{} {} {} {} {} {}'.format(thr_100, thr_200, thr_250, thr_300, thr_400,thr_500)
    logger.info(str(fold))

    each_point_str = ' '
    avg_ = 0
    avg_count = 0
    for idx in range(Each_Point.shape[0]):  
        each_point_str += '{:.2f} '.format(Each_Point[idx, 0] / Each_Point[idx, 1])  
        avg_ += Each_Point[idx, 0]  # * Each_Point[idx, 1]  
        avg_count += Each_Point[idx, 1]

    nme = avg_/avg_count

    return nme, predictions
