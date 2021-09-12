#!/usr/bin/env python
"""
Created by zhenlinx on 1/20/19
"""
import os
import sys
import shutil

import torch


def initialize_model(model, optimizer=None, ckpoint_path=None):
    """
    Initilaize a reg_model with saved checkpoins, or random values
    :param model: a pytorch reg_model to be initialized
    :param optimizer: optional, optimizer whose parameters can be restored from saved checkpoints
    :param ckpoint_path: The path of saved checkpoint
    :return: currect epoch and best validation score
    """
    finished_epoch = 0
    best_score = 0
    if ckpoint_path:
        if os.path.isfile(ckpoint_path):
            print("=> loading checkpoint '{}'".format(ckpoint_path))
            checkpoint = torch.load(ckpoint_path, map_location='cpu')
            if 'best_score' in checkpoint:
                best_score = checkpoint['best_score']
            elif 'reg_best_score' in checkpoint:
                best_score = checkpoint['reg_best_score']
            elif 'seg_best_score' in checkpoint:
                best_score = checkpoint['seg_best_score']
            else:
                ValueError('no best score key')

            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if optimizer and checkpoint.__contains__('optimizer_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            finished_epoch = finished_epoch + checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpoint_path, checkpoint['epoch']))
            del checkpoint
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(ckpoint_path))
    else:
        # reg_model.apply(weights_init)
        model.weights_init()
    return finished_epoch, best_score


def save_checkpoint(state, is_best, path, prefix=None, name='checkpoint.pth.tar', max_keep=1):
    if not os.path.exists(path):
        os.makedirs(path)
    # name = '_'.join([str(state['epoch']), filename])
    name = '_'.join([prefix, name]) if prefix else name
    best_name = '_'.join([prefix, 'model_best.pth.tar']) if prefix else 'model_best.pth.tar'
    torch.save(state, os.path.join(path, name))
    if is_best:
        state["optimizer_state_dict"] = None
        torch.save(state, os.path.join(path, best_name))


def weight_from_truth(truths, n_classes):
    ratio_inv = torch.zeros(n_classes)
    for i_class in range(n_classes):
        try:
            ratio_inv[i_class] = len(truths.view(-1)) / torch.sum(truths == i_class)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            ratio_inv[i_class] = 0
            pass
    loss_weight = ratio_inv / torch.sum(ratio_inv)

    return loss_weight