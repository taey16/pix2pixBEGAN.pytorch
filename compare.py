from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torchvision.utils as vutils
from torch.autograd import Variable

import models.pix2pixBEGAN as netBEGAN
import models.UNet as netGAN
from misc import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--tstDataroot', required=False,
  default='/home1/taey16/storage/pix2pix/facades/test/', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--tstBatchSize', type=int, default=256, help='input batch size')
parser.add_argument('--originalSize', type=int, 
  default=286, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int, 
  default=256, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int, 
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, 
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator')
parser.add_argument('--netG_BEGAN', default='', help="path to netG (to continue training)")
parser.add_argument('--netG_GAN', default='', help="path to netG (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get dataloader
dataloader = getLoader(opt.dataset, 
                       opt.tstDataroot, 
                       opt.originalSize, 
                       opt.imageSize, 
                       opt.tstBatchSize, 
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                       split='test',
                       shuffle=False, 
                       seed=opt.manualSeed)

ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# get models
#import pdb; pdb.set_trace()
netG_BEGAN = netBEGAN.G(inputChannelSize, outputChannelSize, ngf)
netG_BEGAN.apply(weights_init)
if opt.netG_BEGAN != '':
  netG_BEGAN.load_state_dict(torch.load(opt.netG_BEGAN))
print(netG_BEGAN)

netG_GAN = netGAN.G(inputChannelSize, outputChannelSize, ngf)
netG_GAN.apply(weights_init)
if opt.netG_GAN != '':
  netG_GAN.load_state_dict(torch.load(opt.netG_GAN))
print(netG_GAN)

netG_BEGAN.train()
netG_GAN.train()

val_target= torch.FloatTensor(opt.tstBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.tstBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

netG_BEGAN.cuda()
netG_GAN.cuda()
val_target, val_input = val_target.cuda(), val_input.cuda()

# get randomly sampled validation images and save it
val_iter = iter(dataloader)
data_val = val_iter.next()
if opt.mode == 'B2A':
  val_target_cpu, val_input_cpu = data_val
elif opt.mode == 'A2B':
  val_input_cpu, val_target_cpu = data_val
val_target_cpu, val_input_cpu = val_target_cpu.cuda(), val_input_cpu.cuda()
val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

output = torch.FloatTensor(val_input.size(0)*4, 3, val_input.size(2), val_input.size(3)).fill_(0)
for idx in range(val_input.size(0)):
  input_img = val_input[idx,:,:,:].unsqueeze(0)
  target_img = val_target[idx,:,:,:].unsqueeze(0)
  input = Variable(input_img, volatile=True)
  real = Variable(target_img, volatile=True)
  fake_BEGAN = netG_BEGAN(input)
  fake_GAN = netG_GAN(input)

  output[idx*4+0,:,:,:].copy_(input.data.squeeze(0))
  output[idx*4+1,:,:,:].copy_(real.data.squeeze(0))
  output[idx*4+2,:,:,:].copy_(fake_BEGAN.data.squeeze(0))
  output[idx*4+3,:,:,:].copy_(fake_GAN.data.squeeze(0))
vutils.save_image(output, '%s/generated.png' % opt.exp, nrow=4, normalize=True)
