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
from misc import *

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--tstDataroot', required=False,
  default='/path/to/your/pix2pix/facades/test/', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--tstBatchSize', type=int, default=2, help='the batch-size should be a even number.')
parser.add_argument('--imageSize', type=int, 
  default=256, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int, 
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, 
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--interval', type=int, default=20)
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
                       opt.imageSize, 
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
netG = netBEGAN.G(inputChannelSize, outputChannelSize, ngf)
netG.apply(weights_init)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)

netG.train()

val_target= torch.FloatTensor(opt.tstBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.tstBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

netG.cuda()
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


def interpolateInput(model, imgA, imgB, intv=20):
  N = imgA.size(1)*imgA.size(2)*imgA.size(3)
  outA_ = imgA.view(imgA.size(0), N)
  outB_ = imgB.view(imgB.size(0), N)

  zs = torch.cuda.FloatTensor(intv, N)
  for n in range(N):
    valA, valB = outA_[0,n].data[0], outB_[0,n].data[0]
    values = np.linspace(valA, valB, num=intv)
    zs[:,n].copy_(torch.from_numpy(values).cuda())

  zs = zs.view(intv, imgA.size(1), imgA.size(2), imgA.size(3))
  output = torch.cuda.FloatTensor(intv, imgA.size(1), imgA.size(2), imgA.size(3)).fill_(0)
  output = Variable(output)

  for i in range(intv):
    output.data[i] = model.forward(Variable(zs[i,:].unsqueeze(0).cuda(async=True), 
                                   volatile=True)).data.clone()
  return output

interval = opt.interval
N = val_input.size(0)
outputs = torch.FloatTensor((opt.tstBatchSize/2)*interval, 
                             val_target.size(1), 
                             val_target.size(2), 
                             val_target.size(3))
for idx in range(opt.tstBatchSize / 2):
  inputA = val_input[idx,:,:,:].unsqueeze(0)
  targetA = val_target[idx,:,:,:].unsqueeze(0)
  inputB = val_input[(N-1)-idx,:,:,:].unsqueeze(0)
  targetB = val_target[(N-1)-idx,:,:,:].unsqueeze(0)
  inputA = Variable(inputA, volatile=True)
  inputB = Variable(inputB, volatile=True)
  output = interpolateInput(netG, inputA, inputB, interval)
  outputs[(idx*interval):((idx+1)*interval),:].copy_(output.data.squeeze(0).cpu())
vutils.save_image(outputs, '%s/interpolated.png' % opt.exp, nrow=interval, normalize=True)
