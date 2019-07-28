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
    model.eval()
    model.load_weights("./yolact_darknet53_54_800000.pth")
    modules = model.children()

def parse_layer(layer, weights):
    assert isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d)
    print("=> Parsing ", layer)
    if isinstance(layer, torch.nn.Conv2d):
        weight, bias = layer.weight.detach().numpy(), layer.bias
        weight = np.transpose(weight, [2,3,1,0]) # k_h, h_w, in_channels, out_channels
        if bias is None:
            weights.append([weight])
        else:
            bias = layer.bias.detach().numpy()
            weights.append([weight, bias])
    else:
        weights.append([layer.weight.detach().numpy(), layer.bias.detach().numpy(),
                        layer.running_mean.detach().numpy(), layer.running_var.detach().numpy()])
    return True


def parse_sequential(sequential, weights):
    for layer in sequential:
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d):
            parse_layer(layer, weights)
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
            # if isinstance(child, torch.nn.Sequential):
                # parse_sequential(child, weights)
            if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.BatchNorm2d):
                parse_layer(child, weights)
        except StopIteration:
            break
    return True

darknet53 = next(modules)
darknet53_weights = []
parse_module(darknet53, darknet53_weights)

# x = 2*torch.ones([1, 3, 416, 416])
# y = darknet53(x)
# print(torch.sum(y[0]), torch.sum(y[1]), torch.sum(y[2]))

proto_net = next(modules)
proto_net_weights = []
parse_sequential(proto_net, proto_net_weights)

fpn = next(modules)
fpn_weights = []
parse_module(fpn, fpn_weights)

pred = next(modules)
pred_weights = []
parse_module(pred, pred_weights)

segmantic_seg_conv = next(modules)
segmantic_seg_conv_weights = []
parse_layer(segmantic_seg_conv, segmantic_seg_conv_weights)

yolact_weights = {"darknet53":darknet53_weights,
                  "proto_net": proto_net_weights,
                  "fpn":fpn_weights,
                  "pred":pred_weights,
                  "segmantic_seg_conv":segmantic_seg_conv_weights}


np.save("yolact_darknet53_54_800000.npy", yolact_weights)



