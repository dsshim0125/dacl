import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from models import networks
import tqdm
import os
from options.test_options import TestOptions


if __name__ == '__main__':
    opt = TestOptions().parse()

    img_list = glob.glob('%s/kitti/training/image_2/*.png'%opt.root)
    img_list.sort()

    class_list = ['Car', 'Building', 'GuardRail', 'Misc', 'Pole', 'Road', 'Sky', \
                  'Terrain', 'TrafficLight', 'TrafficSign', 'Tree', 'Truck', 'Van', 'Vegetation']


    vkitti_dict = {}

    for i, name in enumerate(class_list):

        vkitti_dict[name] = i



    kitti_dict = {}

    kitti_dict['Car'] = 26
    kitti_dict['Building'] = 11
    kitti_dict['GuardRail'] = 14
    kitti_dict['Pole'] = 17
    kitti_dict['Pole_Group'] = 18

    kitti_dict['Road'] = 7
    kitti_dict['Sky'] = 23
    kitti_dict['Terrain'] = 22
    kitti_dict['TrafficLight'] = 19
    kitti_dict['TrafficSign'] = 20
    kitti_dict['Tree'] = 21
    kitti_dict['Truck'] = 27
    kitti_dict['Van'] = 29
    kitti_dict['Vegetation'] = 21

    net1 = networks.UNetGenerator(output_nc = 14).cuda(int(opt.gpu_ids[0]))


    best_iou = 0
    best_idx = 0

    test_epoch = 'best'
    net1.load_state_dict(torch.load('checkpoints/vkitti2kitti_seg/%s_net_Seg_S.pth'%opt.which_epoch))


    iou_dict = {}

    T_iou = 0
    T_len = 0




    for key in class_list:


        if (key=='Misc' or key=='Vegetation') :

            pass

        else:

            total_iou = 0

            total_len = len(img_list)

            for i in tqdm.tqdm(range(len(img_list))):

                img = Image.open(img_list[i]).convert('RGB')
                label = Image.open(img_list[i].replace('image_2', 'semantic')).convert('L')

                img = transforms.Resize((192,640), Image.BICUBIC)(img)
                label = transforms.Resize((192,640), Image.NEAREST)(label)
                label = np.array(label)

                img_input = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])(img)
                img_input = img_input.unsqueeze(dim=0)



                img_input = img_input.cuda(int(opt.gpu_ids[0]))


                output = net1(img_input)
                output = output[-1].permute(0,2,3,1).detach().cpu().numpy()


                output_label = np.argmax(output, axis=3)[0]



                ### Class mergining & Separation ###

                if (key != 'Tree' and key != 'Pole') and key!= 'Vegetation':

                    total_area = (((label==kitti_dict[key]) + (output_label==vkitti_dict[key])).astype(bool)).astype(int)

                    common = ((label==kitti_dict[key]) * (output_label==vkitti_dict[key]))

                elif key=='Tree' or key=='Vegetation':

                    total_area = (((label == kitti_dict['Tree']) + (output_label == vkitti_dict['Tree']) +\
                                   (output_label == vkitti_dict['Vegetation'])).astype(bool)).astype(int)

                    common = ((label == kitti_dict[key]) * \
                              ((output_label == vkitti_dict['Tree']) + (output_label == vkitti_dict['Vegetation'])))

                else:

                    total_area = (((label == kitti_dict['Pole']) + (label == kitti_dict['Pole_Group'])\
                                   + (output_label == vkitti_dict['Pole'])).astype(bool)).astype(int)

                    common = ((label == kitti_dict['Pole']) + (label == kitti_dict['Pole_Group'])) * (output_label == vkitti_dict['Pole'])

                if np.sum(total_area) !=0:

                    iou = np.sum(common)/np.sum(total_area)
                else:
                    iou = 0
                    total_len = total_len - 1

                total_iou += iou

            T_iou += total_iou
            T_len += total_len


            iou_dict[key] = total_iou/total_len

    print('------------------------------')
    print('mIoU: %.3f'%(T_iou/T_len))
    print('------------------------------')

    print(iou_dict)

