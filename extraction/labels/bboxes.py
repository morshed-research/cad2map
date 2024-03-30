import sys
sys.path.append('../')

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import modules.craft_utils as craft_utils
from . import test
from utils import imgproc
import modules.file_utils as file_utils
import json
import zipfile
import pandas as pd

from .models.craft import CRAFT
from .test import copyStateDict

from collections import OrderedDict

# from google.colab.patches import cv2_imshow

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def bbox_parser(trained_model):
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args(["--trained_model", trained_model])
    return args

def set_model(args):
    # load net
    net = CRAFT()     # initialize
    t = time.time()
    if args.cuda:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    print(f"Loading weights from checkpoint ({args.trained_model}) in {time.time() - t}s")

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from .models.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True
    
    return net, refine_net

def get_bboxes(floorplan, trained_model, result_folder):
    if not isinstance(trained_model, str)or not isinstance(floorplan, str): 
        raise TypeError("all arguments must be strings")

    # prep parser and model
    args = bbox_parser(trained_model)
    net, refine_net = set_model(args)

    t = time.time()

    # run model on image
    print("floor plan image {:s}".format(floorplan))
    image = imgproc.loadImage(floorplan)

    bboxes, polys, score_text, det_scores = test.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args, refine_net)

    # save score text
    # filename, file_ext = os.path.splitext(os.path.basename(floorplan))
    # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)
    # file_utils.saveResult(floorplan, image[:,:,::-1], polys, dirname=result_folder)

    # return bounding boxes
    print("elapsed time for bboxes: {}s\n".format(time.time() - t))
    return bboxes.astype('int32') if len(bboxes) != 0 else bboxes