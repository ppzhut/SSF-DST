from __future__ import print_function

import random

import numpy as np

import data_loader

np.set_printoptions(suppress=True)
import os
import time
import torch
current_path = os.path.dirname(os.path.abspath(__file__))


def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def monitor(process, multiple, second):
    while True:
        sum = 0
        for ps in process:
            if ps.is_alive():
                sum += 1
        if sum < multiple:
            break
        else:
            time.sleep(second)


def save_load_name(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    return name


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, os.path.join(current_path, f'pre_trained_models/{name}.pt'))


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(os.path.join(current_path, f'pre_trained_models/{name}.pt'))
    return model


def getData(name="S1", time_len=1, dataset="DTU"):
    DTU_document_path = "/root/autodl-tmp/DTU"
    KUL_document_path = "/root/autodl-tmp/KUL"

    if dataset == 'DTU':
        return data_loader.get_DTU_data(name, time_len, DTU_document_path)
    elif dataset == 'KUL':
        return data_loader.get_KUL_data(name, time_len, KUL_document_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)





