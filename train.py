import argparse
import numpy
import math
import random
import copy

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.models import FullNet3
#from test_middlebury import test_middlebury
#from models.flowlib import flow_to_image

import datasets
from datasets import FO_all_Vimeo90K_train
from utils import LambdaLR, Logger

from PIL import Image

import pdb

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

### Progress report: http://localhost:9014
# MAKE SURE TO RUN VISDOM: python -m visdom.server -port 9014
       
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--datapath', type=str, default='data/vimeo_septuplet/', help='directory of the dataset')
parser.add_argument('--modelname', default='./trained_models/FullNet_v5.pth')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='epoch to start linearly decaying the learning rate to 0')
#parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--port', type=int, default=9014, help='visdom port')
#parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
#parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        #out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        out = h_relu_4_3
        #pdb.set_trace()
        return out

def img_diff(img1,img2):
    diff = img1-img2
    diff = diff.abs()
    return diff

################## Definition of variables ##################
### Networks
FullNet = FullNet3()

# Place Network in cuda memory
if opt.cuda:
    FullNet.cuda()

### Print parameter count
print("PARAMETER COUNT:")
print(sum(p.numel() for p in FullNet.parameters()))

### DataParallel
FullNet = nn.DataParallel(FullNet)

### Load weights
#FullNet.load_state_dict(torch.load(opt.modelname))

### Load vgg network
vgg = Vgg16()
if opt.cuda:
    vgg.cuda()


### Losses
criterion_L1 = torch.nn.L1Loss() # Syntax: (Output, Target)!!!
criterion_Lp = torch.nn.MSELoss()
#criterion_GAN = nn.BCELoss()


### Optimizers & LR schedulers
optimizerG = torch.optim.Adam(FullNet.parameters(), lr=opt.lr, betas=(0.9, 0.999))

lr_schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

### Input & target memory allocation (otherwise memory will accumulate?)
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor # define a tensor data structure

input_f1 = Tensor(opt.batchSize, 3, 256, 448)
input_f2 = Tensor(opt.batchSize, 3, 256, 448)
input_f3 = Tensor(opt.batchSize, 3, 256, 448)
input_f4 = Tensor(opt.batchSize, 3, 256, 448)
input_f5 = Tensor(opt.batchSize, 3, 256, 448)

### Dataset loader
composed = transforms.Compose([ datasets.RandomHorizontalFlip(),
                                datasets.RandomVerticalFlip(),
                                datasets.ToTensor() ]) # Order of ToTensor important!
dataloader = DataLoader(FO_all_Vimeo90K_train(opt.datapath, transform=composed), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

# Load validation data
#composed_val = transforms.Compose([ transforms.ToTensor() ])
#val_loader = DataLoader(MiddDataset(opt.datapath, transform=composed_val), batch_size=1, num_workers=opt.n_cpu)

### Loss plot
logger = Logger(opt.n_epochs, len(dataloader), port=opt.port)

################## Training ##################
for epoch in range(opt.epoch, opt.n_epochs):

    if epoch >= 1: ##### Starts at epoch + 1 #####
        print('####### Fine-tuning #######')
        FullNet.module.switch_to_finetune()
    else:
        print('####### Pretraining #######')
        FullNet.module.switch_to_pretrain()

    for i, batch in enumerate(dataloader):
        #pdb.set_trace()
        ### Set model input
        f1 = Variable(input_f1.copy_(batch['frame1']), requires_grad=False)
        f2 = Variable(input_f2.copy_(batch['frame2']), requires_grad=False)
        f3 = Variable(input_f3.copy_(batch['frame3']), requires_grad=False)
        f4 = Variable(input_f4.copy_(batch['frame4']), requires_grad=False)
        f5 = Variable(input_f5.copy_(batch['frame5']), requires_grad=False)

        x2 = batch['x2']
        y2 = batch['y2']
        x4 = batch['x4']
        y4 = batch['y4']

        ### Train FullNet #########################################
        f2_hat, f4_hat = FullNet(f1, f3, f5)

        ### Losses #########################################
        optimizerG.zero_grad()

        # L1 loss
        loss_L1 = criterion_L1(f2_hat, f2) + criterion_L1(f4_hat, f4)

        # Perceptual loss
        #loss_Lp = criterion_Lp(vgg(f2_hat), vgg(f2)) + criterion_Lp(vgg(f4_hat), vgg(f4))

        # Flying object L1 loss
        f2_hat_o = Tensor(opt.batchSize, 3, 64, 64)
        f4_hat_o = Tensor(opt.batchSize, 3, 64, 64)
        f2_o = Tensor(opt.batchSize, 3, 64, 64)
        f4_o = Tensor(opt.batchSize, 3, 64, 64)
        for i in range(opt.batchSize):
            f2_hat_o[i,:,:,:] = f2_hat[i, :, y2[i]:y2[i]+64, x2[i]:x2[i]+64]
            f4_hat_o[i,:,:,:] = f4_hat[i, :, y4[i]:y4[i]+64, x4[i]:x4[i]+64]
            f2_o[i,:,:,:] = f2[i, :, y2[i]:y2[i]+64, x2[i]:x2[i]+64]
            f4_o[i,:,:,:] = f4[i, :, y4[i]:y4[i]+64, x4[i]:x4[i]+64]
        loss_Lo = criterion_L1(f2_hat_o, f2_o) + criterion_L1(f4_hat_o, f4_o)

        loss_Gtotal = loss_L1*2.0 + loss_Lo*1.0
        loss_Gtotal.backward() # retain_graph=True # compute gradient

        optimizerG.step() # update weights using gradient descent

        ### Display image differences
        f2_diff = img_diff(f2, f2_hat)
        f4_diff = img_diff(f4, f4_hat)

        logger.log({'loss_total': loss_Gtotal}, 
                    images={'f1': f1, 'GT_f2': f2, 'f3': f3, 'GT_f4': f4, 'f4_hat': f4_hat, 'f2_hat': f2_hat, 'f2_diff': f2_diff, 'f4_diff': f4_diff})

        #gc.collect()
        #torch.cuda.empty_cache()

    ### Update learning rates
    lr_schedulerG.step()

    ### Test on validation set
    #if (epoch % 5) == 0 or epoch == (opt.n_epochs-1):
    #    validate(opt, netG, val_loader)

    ### Save model checkpoints
    torch.save(FullNet.state_dict(), opt.modelname)

    ### Test on middlebury
    #test_middlebury(FullNet, opt.modelname)








