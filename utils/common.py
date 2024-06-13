import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import random
import ast
import logging


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_test_labels(args, loader = None):
    if args.in_dataset.startswith("ImageNet_C") or args.in_dataset in ["ImageNet", "ImageNet_sketch"]:
        test_labels = obtain_ImageNet_classes()
    elif args.in_dataset == "ImageNet10":
        test_labels = obtain_ImageNet10_classes()
    elif args.in_dataset == "ImageNet20":
        test_labels = obtain_ImageNet20_classes()
    elif args.in_dataset in ['bird200', 'car196', 'food101','pet37', 
                            'cub100_ID', 'car98_ID', 'food50_ID', 'pet18_ID']:
        test_labels = loader.dataset.class_names_str
    elif args.in_dataset in ['cifar10', 'cifar100']:
        test_labels = loader.dataset.classes
    return test_labels


def obtain_ImageNet_classes():
    loc = os.path.join('data', 'ImageNet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls


def obtain_ImageNet10_classes():
    class_dict = {"warplane": "n04552348", "sports car": "n04285008",
                  'brambling bird': 'n01530575', "Siamese cat": 'n02123597',
                  'antelope': 'n02422699', 'swiss mountain dog': 'n02107574',
                  "bull frog": "n01641577", 'garbage truck': "n03417042",
                  "horse": "n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()


def obtain_ImageNet20_classes():
    class_dict = {"n04147183": "sailboat", "n02951358": "canoe", "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
                  "n02917067": "bullet train", "n02317335": "starfish", "n01632458": "spotted salamander", "n01630670": "common newt", "n01631663": "eft",
                  "n02391049": "zebra", "n01693334": "green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",
                  "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[0])}
    return class_dict.values()


def get_num_cls(args):
    if args.in_dataset.startswith('ImageNet_C'):
        return 1000
    else:
        NUM_CLS_DICT = {
            'bird200':200,
            'car196': 196, 
            'food101': 101, 
            'pet37': 37,
            'ImageNet10': 10,
            'ImageNet20': 20,
            'ImageNet': 1000,
            'ImageNet_sketch': 1000,
            'cub100_ID':100, 
            'cub100_OOD':100, 
            'car98_ID': 98,
            'car98_OOD': 98,
            'food50_ID': 50,
            'food50_OOD': 51,
            'pet18_ID': 18,
            'pet18_OOD': 19,
            'cifar10': 10,
            'cifar100': 100,
        }
        n_cls = NUM_CLS_DICT[args.in_dataset]
        return n_cls