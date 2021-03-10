import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from PIL import Image
import torchvision.transforms.functional as F
import math
import glob


txt_list = glob.glob('/media/KITTI/vkitti/label/*.txt')
txt_list.remove('/media/KITTI/vkitti/label/README_scenegt.txt')

class_list = ['Car', 'Building', 'GuardRail', 'Misc', 'Pole', 'Road', 'Sky', \
              'Terrain', 'TrafficLight', 'TrafficSign', 'Tree', 'Truck', 'Van', 'Vegetation']


seg_rgb_dict = {}

for key in class_list:

    seg_rgb_dict[key] = []


for i in range(len(txt_list)):

    lines = open(txt_list[i], 'r').readlines()


    for line in lines[1:]:

        line = line.rstrip()
        data = line.split(' ')


        key = data[0]

        value = (int(data[1]), int(data[2]), int(data[3]))


        if 'Car' in key:

            key = 'Car'

        elif 'Van' in key:

            key = 'Van'



        if value not in seg_rgb_dict[key]:
            seg_rgb_dict[key].append(value)



class RandomHorizontalFlip(object):
    """
    Random horizontal flip.

    prob = 0.5
    """

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, img, label):
        if (self.prob is None and random.random() < 0.5) or self.prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

        return img, label


class SegRandomImgAugment(object):
    """Randomly shift gamma"""

    def __init__(self, size=(192,640)):

        self.size = size

    def __call__(self, inputs):

        img = inputs[0]
        label = inputs[1]

        h = img.height
        w = img.width
        w0 = w

        # print(h, w, self.size)

        if self.size == [-1]:
            divisor = 32.0
            h = int(math.ceil(h / divisor) * divisor)
            w = int(math.ceil(w / divisor) * divisor)
            self.size = (h, w)

        scale_transform = transforms.Compose([transforms.Resize(self.size, Image.NEAREST)])
        scale_transform2= transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])


        img = scale_transform2(img)

        label = scale_transform(label)


        flip_prob = random.random()
        flip_transform = RandomHorizontalFlip(flip_prob)

        img, label = flip_transform(img, label)


        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.0)
            contrast = random.uniform(0.8, 1.0)
            saturation = random.uniform(0.8, 1.0)

            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)




        label = np.array(label)

        h, w, c = label.shape
        label_map = np.zeros((h, w))

        for class_idx in range(1, len(class_list)):

            if not len(seg_rgb_dict[class_list[class_idx]]) > 1:

                new_map = (label == np.array(seg_rgb_dict[class_list[class_idx]]).reshape(3))
                new_map = np.mean(new_map, axis=2).astype(int) * class_idx

                label_map += new_map

            else:

                for label_rgb in seg_rgb_dict[class_list[class_idx]]:
                    new_map = (label == np.array(label_rgb).reshape(3))
                    new_map = np.mean(new_map, axis=2).astype(int) * class_idx

                    label_map += new_map

        label_map = label_map.astype(int)

        img = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])(img)
        label_map = torch.LongTensor(label_map)

        return img, label_map