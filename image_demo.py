#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-24 12:49:50
#   Description :
#
#================================================================

import cv2
import torch

from PIL import Image
from yolact import Yolact
from utils import FastBaseTransform, prep_display

image_path = "./docs/boy.jpg"

frame = torch.from_numpy(cv2.imread(image_path)).float()
batch = FastBaseTransform()(frame.unsqueeze(0))

model = Yolact()
with torch.no_grad():
    torch.set_default_tensor_type('torch.FloatTensor')
    model.load_weights("./yolact_darknet53_54_800000.pth")
    model.eval()
    preds = model(batch)
    img_numpy = prep_display(preds, frame, 0.2)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_numpy)
    image.show()

