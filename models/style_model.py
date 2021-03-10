import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
import torch.nn.functional as F
from utils import dataset_util
import math
import torch.nn as nn


class STYLEModel(BaseModel):
    def name(self):
        return 'STYLEModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            # cyclegan
            parser.add_argument('--lambda_Src', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_Tgt', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=30.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')


            parser.add_argument('--freeze_bn', action='store_true', help='freeze the bn in mde')
            parser.add_argument('--freeze_in', action='store_true', help='freeze the in in cyclegan')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        if self.isTrain:

            self.loss_names = ['D_Src', 'G_Src', 'cycle_Src', 'idt_Src', 'D_Tgt', 'G_Tgt', 'cycle_Tgt', 'idt_Tgt']

        if self.isTrain:
            visual_names_src = ['src_img', 'fake_tgt']
            visual_names_tgt = ['tgt_left_img', 'fake_src_left']

            if self.opt.lambda_identity > 0.0:
                visual_names_src.append('idt_src_left')
                visual_names_tgt.append('idt_tgt')

            self.visual_names = visual_names_src + visual_names_tgt

        else:
            self.visual_names = ['img', 'img_trans']

        if self.isTrain:
            self.model_names = ['G_Src', 'G_Tgt', 'D_Src', 'D_Tgt']
        else:
            self.model_names = ['G_Tgt']


        self.netG_Src = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)
        self.netG_Tgt = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            self.netD_Src = networks.init_net(networks.Discriminator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)
            self.netD_Tgt = networks.init_net(networks.Discriminator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)


        if self.isTrain:
            # define loss functions

            self.fake_src_pool = ImagePool(opt.pool_size)
            self.fake_tgt_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()


            self.optimizer_G_trans = torch.optim.Adam(itertools.chain(self.netG_Src.parameters(),
                                                                    self.netG_Tgt.parameters()),
                                                                    lr=opt.lr_trans, betas=(0.5, 0.9))


            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Src.parameters(), 
                                                                    self.netD_Tgt.parameters()),
                                                                    lr=opt.lr_trans, betas=(0.5, 0.9))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G_trans)
            self.optimizers.append(self.optimizer_D)
            if opt.freeze_bn:
                self.netG_Depth_S.apply(networks.freeze_bn)
                self.netG_Depth_T.apply(networks.freeze_bn)
            if opt.freeze_in:
                self.netG_Src.apply(networks.freeze_in)
                self.netG_Tgt.apply(networks.freeze_in)



    def set_input(self, input):

        if self.isTrain:
            self.src_img = input['src']['img'].to(self.device)
            self.tgt_left_img = input['tgt']['left_img'].to(self.device)
            self.tgt_right_img = input['tgt']['right_img'].to(self.device)
            self.num = self.src_img.shape[0]
        else:
            self.img = input['left_img'].to(self.device)

    def forward(self):

        if self.isTrain:
            pass
    
        else:
            self.img_trans = self.netG_Tgt(self.img)


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_Src(self):
        fake_tgt = self.fake_tgt_pool.query(self.fake_tgt)
        self.loss_D_Src = self.backward_D_basic(self.netD_Src, self.tgt_left_img, fake_tgt)

    def backward_D_Tgt(self):
        fake_src_left = self.fake_src_pool.query(self.fake_src_left)
        self.loss_D_Tgt = self.backward_D_basic(self.netD_Tgt, self.src_img, fake_src_left)

    def backward_G(self):

        lambda_idt = self.opt.lambda_identity
        lambda_Src = self.opt.lambda_Src
        lambda_Tgt = self.opt.lambda_Tgt

        # =========================== synthetic ==========================
        self.fake_tgt = self.netG_Src(self.src_img)
        self.idt_tgt = self.netG_Tgt(self.src_img)
        self.rec_src = self.netG_Tgt(self.fake_tgt)

        self.loss_G_Src = self.criterionGAN(self.netD_Src(self.fake_tgt), True)
        self.loss_cycle_Src = self.criterionCycle(self.rec_src, self.src_img)
        self.loss_idt_Tgt = self.criterionIdt(self.idt_tgt, self.src_img) * lambda_Src * lambda_idt


        self.loss = self.loss_G_Src + self.loss_cycle_Src + self.loss_idt_Tgt
        self.loss.backward()

        # ============================= real =============================
        self.fake_src_left = self.netG_Tgt(self.tgt_left_img)
        self.idt_src_left = self.netG_Src(self.tgt_left_img)
        self.rec_tgt_left = self.netG_Src(self.fake_src_left)

        self.loss_G_Tgt = self.criterionGAN(self.netD_Tgt(self.fake_src_left), True)
        self.loss_cycle_Tgt = self.criterionCycle(self.rec_tgt_left, self.tgt_left_img)
        self.loss_idt_Src = self.criterionIdt(self.idt_src_left, self.tgt_left_img) * lambda_Tgt * lambda_idt


        self.loss_G = self.loss_G_Tgt + self.loss_cycle_Tgt + self.loss_idt_Src
        self.loss_G.backward()

    def optimize_parameters(self):
        
        self.forward()
        self.set_requires_grad([self.netD_Src, self.netD_Tgt], False)
        self.optimizer_G_trans.zero_grad()
        self.backward_G()
        self.optimizer_G_trans.step()

        self.set_requires_grad([self.netD_Src, self.netD_Tgt], True)
        self.optimizer_D.zero_grad()
        self.backward_D_Src()
        self.backward_D_Tgt()
        self.optimizer_D.step()
