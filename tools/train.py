
import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset

from lib.core import function_decoder
from lib.models.HRSpacetoDepth import get_HRSpacetoDepth_net
import time
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cephalometric Landmarks Localization Network...')
    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = get_HRSpacetoDepth_net(config)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
  
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 1e2
    best_nme_t = 1e2
    nme_t = 1e2
    last_epoch = config.TRAIN.BEGIN_EPOCH
  
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'BestModel.pth')

        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train='train'),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train='valid'),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train='test'),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
  
    best_epoch = -1
  
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function_decoder.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)

        lr_scheduler.step()

        nme, predictions = function_decoder.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)
        
        is_best = nme < best_nme

        if is_best:
            nme_t, predictions_t = function_decoder.inference(config, test_loader, model)
            best_epoch = epoch

        best_nme = min(nme, best_nme)
        best_nme_t = nme_t if is_best else best_nme_t

        logger.info("best: {},  ===NowValidNME: {:.3f},  ===BestValidNME: {:.3f},  ===NowTestNME:{:.3f}, ===BestTestNME:{:.3f}, ===BestEpoch:{}".format(is_best,nme,best_nme,nme_t,best_nme_t,best_epoch)) 
        print()
        if is_best:
            final_model_state_file = os.path.join(final_output_dir, 'BestModel.pth')
            torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
    print('done!')


