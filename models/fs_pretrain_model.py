import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
import torch.nn.functional as F
from utils import dataset_util
import math
import torch.nn as nn


class FSPRETRAINModel(BaseModel):
    def name(self):
        return 'FSPRETRAINModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            
            parser.add_argument('--g_src_premodel', type=str, default=" ",help='pretrained G_Src model')
        
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        if self.isTrain:
            self.loss_names = ['moco']
          
        if self.isTrain:
            self.visual_names = ['src_img', 'fake_tgt']
        else:
            self.visual_names = ['pred', 'img']

        if self.isTrain:
            self.model_names = ['G_Pretrain_S', 'header_q']

        else:
            self.model_names = ['G_Pretrain_S']

        self.netG_Pretrain_S = networks.init_net(networks.UNetGenerator(norm='batch'), init_type='normal', gpu_ids=opt.gpu_ids)

        self.netG_Src = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            self.init_with_pretrained_model('G_Src', self.opt.g_src_premodel)
            self.netG_Src.eval()

            #################################################
            ################ MoCo Settings ##################
            #################################################

            self.alpha = 65536
            self.dim = 256
            self.K = 1024
            self.T = 0.07
            self.momentum = 0.999

            self.m = self.T * math.log(self.alpha / self.K)

            self.encoder_k = networks.init_net(networks.UNetGenerator(norm='batch'), init_type='normal',
                                               gpu_ids=opt.gpu_ids)

            self.netheader_q = networks.init_net(networks.Header(), init_type='normal', gpu_ids=opt.gpu_ids)
            self.header_k = networks.init_net(networks.Header(), init_type='normal', gpu_ids=opt.gpu_ids)

            for param_q, param_k in zip(self.netG_Pretrain_S.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            for param_q, param_k in zip(self.netheader_q.parameters(), self.header_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            self.register_buffer("queue", torch.randn(self.dim, self.K))
            self.queue = nn.functional.normalize(self.queue, dim=0)

            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))



        if self.isTrain:
            # define loss functions
            self.criterionDepthReg = torch.nn.L1Loss()
            self.criterionSmooth = networks.SmoothLoss()
            self.criterionImgRecon = networks.ReconLoss()

            self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netG_Pretrain_S.parameters(), self.netheader_q.parameters()),
                                                lr=opt.lr_task, betas=(0.9, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)



    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.netG_Pretrain_S.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.netheader_q.parameters(), self.header_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr



    def set_input(self, input):

        if self.isTrain:

            self.src_img = input['src']['img'].to(self.device)


            self.num = self.src_img.shape[0]



        else:
            self.img = input['left_img'].to(self.device)

    def forward(self):

        if self.isTrain:

            self.fake_tgt = self.netG_Src(self.src_img).detach()


            #################################################
            ################# MoCo Forward ##################
            #################################################


            q = self.netG_Pretrain_S(self.fake_tgt)[0]
            # print(len(q), q[0].shape)
            # q = q[-1]

            q = nn.AvgPool2d(16, 64)(q).squeeze(dim=2)
            q = q.squeeze(dim=2)
            q = self.netheader_q(q)  # queries: NxC
            # print(q.shape)
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(self.src_img)[0]
                k = nn.AvgPool2d(16, 64)(k).squeeze(dim=2)
                k = k.squeeze(dim=2)
                k = self.header_k(k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)-self.m
            # negative logits: NxK

            clones = self.queue.clone().detach()
            # clones = clones.half()

            clones = clones.cuda()
            l_neg = torch.einsum('nc,ck->nk', [q, clones])

            # logits: Nx(1+K)
            self.logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            self.logits /= self.T

            # labels: positive key indicators
            self.labels = torch.zeros(self.logits.shape[0], dtype=torch.long)

            self.labels = self.labels.cuda(None, non_blocking=True)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)


            
        else:
            self.pred = self.netG_Pretrain_S(self.img)[-1]

    def backward_G(self):



        ##########################################################
        #################### MoCo Backward ########################
        ###########################################################

        self.loss_moco = nn.CrossEntropyLoss()(self.logits, self.labels)

        self.loss_G_Depth =  self.loss_moco
        self.loss_G_Depth.backward()

    def optimize_parameters(self):
        
        self.forward()
        self.optimizer_G_task.zero_grad()
        self.backward_G()
        self.optimizer_G_task.step()
