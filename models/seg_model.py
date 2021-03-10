import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
import torch.nn.functional as F
from utils import dataset_util
import math
import torch.nn as nn


class SEGModel(BaseModel):
    def name(self):
        return 'SEGModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:

            parser.add_argument('--g_src_premodel', type=str, default=" ",help='pretrained G_Src model')


            parser.add_argument('--s_seg_premodel', type=str, default=" ", help='pretrained depth estimation model')


        
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        if self.isTrain:
            self.loss_names = ['Seg']
          
        if self.isTrain:
            self.visual_names = ['src_img', 'fake_tgt']
        else:
            self.visual_names = ['pred', 'img']

        if self.isTrain:
            self.model_names = ['Seg_S']

        else:
            self.model_names = ['Seg_S']



        self.netSeg_S = networks.init_net(networks.UNetGenerator(norm='batch', output_nc=14), init_type='normal',\
                                          gpu_ids=opt.gpu_ids)

        self.netG_Src = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        if self.isTrain:


            self.init_with_pretrained_model('G_Src', self.opt.g_src_premodel)
            self.init_with_pretrained_model('Seg_S', self.opt.s_seg_premodel)
            self.netG_Src.eval()




        if self.isTrain:
            # define loss functions
            self.criterionSeg = torch.nn.CrossEntropyLoss()


            self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netSeg_S.parameters()),
                                                lr=opt.lr_task, betas=(0.9, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)




    def set_input(self, data):

        if self.isTrain:
            self.src_img = data['img']
            self.labels = data['label']

        else:
            self.img = data['img'].to(self.device)

    def forward(self):

        if self.isTrain:

            self.fake_tgt = self.netG_Src(self.src_img).detach()
            self.out = self.netSeg_S(self.fake_tgt)

            
        else:
            self.pred = self.netSeg_S(self.img)[-1]



    def backward_G(self):



        self.loss_Seg = 0.0

        labels = dataset_util.scale_pyramid(torch.unsqueeze(self.labels, dim=1).float(), 4)


        for (gen_seg, label) in zip(self.out[1:], labels):

            label = torch.squeeze(label)
            label = label.long()
            label = label.to(self.device)
            self.loss_Seg += self.criterionSeg(gen_seg, label)



        self.loss_G_Seg = self.loss_Seg
        self.loss_G_Seg.backward()

    def optimize_parameters(self):
        
        self.forward()
        self.optimizer_G_task.zero_grad()
        self.backward_G()
        self.optimizer_G_task.step()
