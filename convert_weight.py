#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_weight.py
#   Author      : YunYang1994
#   Created date: 2019-07-27 18:07:20
#   Description :
#
#================================================================

import torch
import numpy as np
from yolact import Yolact


with torch.no_grad():
    model = Yolact()
    model.load_weights("./yolact_darknet53_54_800000.pth")

modules = model.children()

# Darknet53
darknet53 = next(modules)

i = 0
def parse_sequential(sequential, weights):
    global i
    for layer in sequential:
        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            print("=> Parsing ", layer)
            weights.append([layer.weight, layer.bias])
        elif isinstance(layer, torch.nn.BatchNorm2d):
            i += 1
            print("=> Parsing ", layer)
            weights.append([layer.weight, layer.bias,
                layer.running_mean, layer.running_var])
        else:
            continue
    return True

def parse_module(module, weights):
    children = module.children()
    while True:
        try:
            child = next(children)
            if isinstance(child, torch.nn.Module):
                parse_module(child, weights)
            if isinstance(child, torch.nn.Sequential):
                parse_sequential(child, weights)
        except StopIteration:
            break
    return True

darknet53_weights = []
parse_module(darknet53, darknet53_weights)




# weights['preconv'] = preconv[0].weight.detach().numpy()
# np.save("yolact_darknet53_54_800000.npy", weights)
# weights_dict = np.load("yolact_darknet53_54_800000.npy", encoding='latin1').item()



