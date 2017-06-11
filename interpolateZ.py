from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import Generator

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--fname', default='0', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--ngf', type=int, default=128)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###########   Load netG   ###########
assert opt.netG != '', "netG must be provided!"
nc = 3
netG = Generator(nc, opt.ngf, opt.nz, opt.imageSize)
netG.load_state_dict(torch.load(opt.netG))
print(netG)

###########   Generate   ###########
noise = torch.FloatTensor(opt.batchSize, opt.nz)
noise = Variable(noise)
noise1 = torch.FloatTensor(opt.batchSize, opt.nz)
noise1 = Variable(noise1)

netG.cuda()
noise = noise.cuda()
noise1 = noise1.cuda()

def interpolateZ(model, imgA, imgB, imageSize=128, intv=20):
  N = imgA.size(1)

  zs = torch.cuda.FloatTensor(intv, N)
  for n in range(N):
    valA, valB = imgA[0,n].data[0], imgB[0,n].data[0]
    values = np.linspace(valA, valB, num=intv)
    zs[:,n].copy_(torch.from_numpy(values).cuda())

  output = torch.cuda.FloatTensor(intv, 3, imageSize, imageSize).fill_(0)
  output = Variable(output)
  for i in range(intv):
    input = Variable(zs[i,:].unsqueeze(0), volatile=True)
    output.data[i] = model(input).data.clone()
  return output

noise.data.uniform_(-1,1)
noise1.data.uniform_(-1,1)
interval=14
outputs = interpolateZ(netG, noise, noise1, imageSize=opt.imageSize, intv=interval)
vutils.save_image(outputs.data, '%s/interpolated_%s.png' % (opt.outf, opt.fname), nrow=interval, normalize=True)
"""
fake = netG(noise)
vutils.save_image(fake.data,
            '%s/samples.png' % (opt.outf),
            normalize=True)
"""
