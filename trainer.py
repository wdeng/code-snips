'''
    The related to training functions
'''

from random import shuffle
import shutil
import os

import numpy as np
import math

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc

from torchvision.transforms import Compose

from skimage import io

from skimage.morphology import remove_small_objects, remove_small_holes, label
from skimage.measure import find_contours
from scipy.ndimage import distance_transform_edt
from skimage.morphology import watershed

EPS = 1E-6
def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * (intersection.sum()+EPS) / (im1.sum() + im2.sum() + EPS)

def post_watershed_slow(seeds, mask):
    mask = remove_small_objects(remove_small_holes(mask > 0.5, 8000), 200)
    seeds = remove_small_objects(remove_small_holes(seeds > 0.8, 5000), 20)
    seeds = seeds & mask

    distance = -1 * distance_transform_edt(mask)
    labels = label(seeds)
    segs = watershed(distance, labels, mask=mask)
    return segs

def normalize(im):
    im_min, im_max = np.min(im), np.max(im)
    return (im - im_min) / (im_max - im_min)

def write_output(ims, name, path='./', norm=True):
    if not os.path.exists(path):
        os.mkdir(path)
    path_ = os.path.join(path, name)
    for idx, im in enumerate(ims):
        if type(im) is torch.Tensor:
            im = im.cpu().numpy()
        fname = path_ + str(idx) + '.png'
        im = normalize(im) if norm else im
        io.imsave(fname, im)


def load_checkpoint(resume_path, model, optimizer):
    if os.path.isfile(resume_path):
        print('loading checkpoint: {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        epoch_start = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))
    return best_loss, epoch_start


def get_roc(predicts, truths):
    '''
    must be (c, n, w, d)
    '''
    class_n = truths.shape[0]
    if truths.dtype != 'bool':
        truths = truths > 0.1

    fpr = dict()
    tpr = dict()
    thre = dict()
    roc_auc = dict()
    for i in range(class_n):
        fpr[i], tpr[i], thre[i] = roc_curve(
            truths[i].ravel(), predicts[i].ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc, thre


def save_checkpoint(states, path, filename='model_latest'):
    if not os.path.exists(path):
        os.makedirs(path)
    # os.path.join(path, prefix) + '_' + filename
    checkpoint_name = os.path.join(path, filename+'.pth.tar')
    torch.save(states, checkpoint_name)


class ModelTrain(object):
    def __init__(self, hps, model):
        self.resume_path = hps.resume_path
        self.best_loss = 1e8
        self.epoch_start = 0
        self.checkpoint_path = hps.checkpoint_path

        self.device = torch.device("cuda" if hps.use_gpu else "cpu")
        self.model_setup(model, self.device)

        self.training_setup(hps)

        if os.path.isfile(self.resume_path):
            self.load_checkpoint()

    def model_setup(self, model, device):
        self.model = model.to(device)
        # self.weight_init

    def weight_init(self, model, init_type=None):
        if init_type == 'kaiming':
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif init_type == 'normal':
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

    # param use **args passing will be easier
    def loss_setup(self, loss_class, weights=None):
        if weights is None:
            self.loss_func = loss_class()
        else:
            self.loss_func = loss_class(weight=weights.to(self.device))

    def training_setup(self, hps):
        self.optimizer = getattr(optim, hps.optimizer)(
            self.model.parameters(), lr=hps.learning_rate.lr, amsgrad=True)
        params = hps.learning_rate.scheduler_params.__dict__
        self.scheduler = getattr(optim.lr_scheduler, hps.learning_rate.scheduler)(
            self.optimizer, **params)

    def data_setup(self, *input):
        r"""Defines the datasets for the training and evaluation
        Should be overridden by all subclasses.
        """
        self.evaloader = None
        self.trainloader = None

        raise NotImplementedError

    def load_checkpoint(self):
        self.best_loss, self.epoch_start = load_checkpoint(
            self.resume_path, self.model, self.optimizer)

    def save_checkpoint(self, epoch, filename='model_latest', path=None, ):
        if not path: path = self.checkpoint_path
        os.makedirs(path, exist_ok=True)
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'optimizer': self.optimizer.state_dict()
            },
            path, filename)

    def train(self, device=None):  # torch.device("cpu")
        assert hasattr(self, 'trainloader')

        if device is None:
            device = self.device
        running_loss, iters = 0.0, 0
        self.model.train()

        for _, sample in enumerate(self.trainloader):
            inputs, labels = sample['data'].to(
                device), sample['labels'].to(device)
            if 'weights' in sample:
                self.loss_func.weight = sample['weights'].to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            iters += 1
        running_loss /= iters
        return running_loss

    def validate_with_auc(self, device=None, output_func='none', write_path=None):
        assert hasattr(self, 'evaloader')
        if device is None:
            device = self.device
        running_loss, iters = 0.0, 0
        self.model.eval()
        results = []
        refs = []
        im_names = []
        with torch.no_grad():
            for _, sample in enumerate(self.evaloader):
                inputs, labels = sample['data'].to(
                    device), sample['labels'].to(device)
                if 'img_name' in sample:
                    im_names += sample['img_name']
                if 'weights' in sample:
                    self.loss_func.weight = sample['weights'].to(device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                running_loss += loss.item()
                if hasattr(F, output_func):
                    outputs = getattr(F, output_func)(outputs)
                results += list(outputs.cpu().numpy())
                refs += list(labels.cpu().numpy())
                iters += 1

        running_loss /= iters

        if len(im_names) > 0 and write_path is not None:
            for i, r in enumerate(results):
                write_output(r, im_names[i] + '.out', write_path)

        results = np.swapaxes(np.array(results), 0, 1)
        refs = np.swapaxes(np.array(refs), 0, 1)

        _, _, aucs, _ = get_roc(results, refs)
        return running_loss, aucs

    def validate(self, device=None, output_func='none'):
        assert hasattr(self, 'evaloader')
        if device is None:
            device = self.device
        running_loss, iters = 0.0, 0
        self.model.eval()
        refs = []
        results = []
        im_names = []
        with torch.no_grad():
            for _, sample in enumerate(self.evaloader):
                inputs, labels = sample['data'].to(
                    device), sample['labels'].to(device)
                if 'weights' in sample: self.loss_func.weight = sample['weights'].to(device)
                if 'img_name' in sample:
                    im_names += sample['img_name']
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                running_loss += loss.item()
                if hasattr(F, output_func):
                    outputs = getattr(F, output_func)(outputs)
                iters += 1
                results += list(outputs.cpu().numpy())
                refs += list(labels.cpu().numpy())

        running_loss /= iters

        images = {
            'ref': refs,
            'result': results,
            'names': im_names
        }
        return running_loss, images
