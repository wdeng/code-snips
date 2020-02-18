import os
import argparse
import json
import time
import shutil
import numpy as np

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.abspath('./'))
from prostate_seg.data_prostate import DataProstate, json2obj
import prostate_seg.transform_2d as trans
from prostate_seg.trainer import ModelTrain as Trainer

# from libs.net_models.unet_dilated import UNetDilated as UNet
from prostate_seg.res_fpn import ResFPN50 as UNet

# from libs.gan_models.generators import Symmetric as Generator
# from libs.gan_models.discriminators import Discriminator
# from libs.gan_models.pixel_discriminator import PixelD as Discriminator

from prostate_seg.dice_loss import DiceLoss

def train(net, hps, dts):
    writer = SummaryWriter(hps.checkpoint_path)

    transforms_t = []
    transforms_t.append(Compose([
        trans.RandomRotation(),
        trans.RandomRescale((0.6, 1.8)),
        trans.RandomCrop(hps.crop_size),
        trans.RandomHorizontalFlip(),
    ]))
    transforms_t[0].target = 'both'

    transforms_t.append(Compose([
        trans.RandomGamma((0.6, 1.2)),
        trans.Normalize((-1.0, 1.0)), 
        trans.RandomContrast((0.2, 1.0)),
        trans.RandomLocalContrast(sigma=16, constrast_range=(0.8, 1.0))
    ]))
    transforms_t[-1].target = 'data'

    transforms_t.append(Compose([
        trans.RandomElastic(alpha=(0, 16), sigma=18),
    ]))
    transforms_t[-1].target = 'both'
    
    transforms_t.append(Compose([
        trans.RandomBlur(sigmas=(0.0, 5.0)), 
    ]))
    transforms_t[-1].target = 'data'

    # transforms_t.append(Compose([ trans.ThresholdToBinary(0.3), ]))
    # transforms_t[-1].target = 'labels'

    transforms_e = []
    transforms_e.append(Compose([trans.Normalize((-1.0, 1.0))]))
    transforms_e[0].target = 'data'

    trainer = Trainer(hps, net)

    trainset = DataProstate(dts.train_path, transforms_t)
    trainer.trainloader = DataLoader(trainset, batch_size=hps.batch_size, shuffle=True, num_workers=8)

    evalset = DataProstate(dts.val_path, transforms_e)
    trainer.evaloader = DataLoader(evalset, batch_size=2, shuffle=False, num_workers=2)

    trainer.loss_func = DiceLoss()

    for epoch in range(trainer.epoch_start, hps.total_iterations):
        loss = trainer.train()
        writer.add_scalar('data/train_loss', loss, epoch)
        current_epoch = epoch + 1
        if current_epoch >= 100 and not current_epoch % hps.save_iterations:
            p = os.path.join(hps.checkpoint_path, str(current_epoch))
            running_loss = trainer.validate(write_path=p, epoch=current_epoch, writer=writer)
            if running_loss < trainer.best_loss:
                writer.add_scalar('val/soft_dice', running_loss, current_epoch)
                trainer.save_checkpoint(current_epoch, 'model_'+str(current_epoch), hps.checkpoint_path)

    writer.close()
    del trainer, trainset
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-hp', '--hyperparams', default='./hparams.json',
                        type=str, metavar='FILE.PATH',
                        help='path to hyperparameters setting file (default: ./hparams.json)')
    parser.add_argument('-d', '--datasets', default='./data_setting.json',
                        type=str, metavar='FILE.PATH',
                        help='path to dataset setting file (default: ./data_setting.json)')
    args = parser.parse_args()

    try:
        hp_data = open(args.hyperparams).read()
    except IOError:
        print('Couldn\'t read hyperparameter setting file')

    try:
        path_data = open(args.datasets).read()
    except IOError:
        print('Couldn\'t read dataset setting file')

    hps = json2obj(hp_data)
    dts = json2obj(path_data)

    net = UNet(hps.input_channels, hps.class_num, output_func='Sigmoid', pretrained=True)

    try:
        train(net, hps, dts)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            torch.cuda.empty_cache()
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    main()
