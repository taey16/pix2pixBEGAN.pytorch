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
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.BEGAN as net
from misc import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='folder',  help='dataset name (It does not need to be modified)')
parser.add_argument('--dataroot', default='', help='path to trn dataset')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=64, help='input val. batch size')
parser.add_argument('--originalSize', type=int, 
  default=142, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int, 
  default=128, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int, 
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, 
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator(i.e. dimension of z)')
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for D')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for G')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=30, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambda_k', type=float, default=0.001, help='learning rate of k')
parser.add_argument('--gamma', type=float, default=0.7, help='balance bewteen D and G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=1000, help='interval for evauating(generating) images')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = 101
#opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# NOTE get dataloader
dataloader = getLoader(opt.dataset, 
                       opt.dataroot, 
                       opt.originalSize, 
                       opt.imageSize, 
                       opt.batchSize, 
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                       split='train',
                       shuffle=True, 
                       seed=opt.manualSeed)
# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# NOTE get models
netG = net.Decoder(inputChannelSize, ngf, opt.hidden_size)
netG.apply(weights_init)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)
netD = net.D(inputChannelSize, ndf, ndf, opt.hidden_size)
netD.apply(weights_init)
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))
print(netD)

netG.train()
netD.train()

input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, opt.hidden_size, 1, 1)
# variable for validation
fixed_noise = torch.FloatTensor(opt.valBatchSize, opt.hidden_size, 1, 1).uniform_(-1, 1)

netD.cuda()
netG.cuda()
input = input.cuda()
noise = noise.cuda()
fixed_noise = fixed_noise.cuda()

input = Variable(input)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# get optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.0)

# NOTE training loop
ganIterations = 0
k = 0 # control how much emphasis is put on L(G(z_D)) during gradient descent.
M_global = AverageMeter()
K_avg = AverageMeter()
for epoch in range(opt.niter):
  for i, data in enumerate(dataloader, 0):
    input_cpu, _ = data
    batch_size = input_cpu.size(0)

    input_cpu = input_cpu.cuda(async=True)
    input.data.resize_as_(input_cpu).copy_(input_cpu)

    # max_D first
    for p in netD.parameters(): 
      p.requires_grad = True 
    netD.zero_grad()

    # NOTE: compute L_D
    noise.data.resize_(batch_size, opt.hidden_size, 1, 1).uniform_(-1, 1)
    recon_real = netD(input)
    fake = netG(noise)
    fake = fake.detach()
    recon_fake = netD(fake)
    # compute L(x)
    errD_real = torch.mean(torch.abs(recon_real - input))
    # compute L(G(z_D))
    errD_fake = torch.mean(torch.abs(recon_fake - fake))
    # compute L_D
    errD = errD_real - k * errD_fake
    errD.backward()
    optimizerD.step()

    # prevent computing gradients of weights in Discriminator
    for p in netD.parameters(): 
      p.requires_grad = False
    netG.zero_grad() # start to update G

    # NOTE compute L_G
    noise.data.resize_(batch_size, opt.hidden_size, 1, 1).uniform_(-1, 1)
    fake = netG(noise)
    recon_fake = netD(fake)
    errG = torch.mean(torch.abs(recon_fake - fake))
    errG.backward()
    optimizerG.step()
    ganIterations += 1

    # NOTE compute k_t and M_global
    balance = (opt.gamma * errD_real - errD_fake).data[0]
    k = min(max(k + opt.lambda_k * balance, 0), 1)
    measure = errD_real.data[0] + np.abs(balance)
    M_global.update(measure, input.size(0))
    K_avg.update(k, input.size(0))

    # logging
    if ganIterations % opt.display == 0:
      print('[%d/%d][%d/%d] Ld: %f Lg: %f, M_global: %f(%f), K: %f(%f), balance.: %f lr: %f'
          % (epoch, opt.niter, i, len(dataloader),
             errD.data[0], errG.data[0],
             measure, M_global.avg, k, K_avg.avg, balance,
             optimizerG.param_groups[0]['lr']))
      sys.stdout.flush()
      trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                        (i, errD.data[0], errG.data[0], measure, M_global.avg, k, balance))
      trainLogger.flush()
    if ganIterations % opt.evalIter == 0:
      fake = netG(fixed_noise)
      recon_fake = netD(fake)
      vutils.save_image(fake.data, '%s/epoch_%08d_iter%08d_fake.png' % \
        (opt.exp, epoch, ganIterations), nrow=8, normalize=True)
      vutils.save_image(recon_fake.data, '%s/epoch_%08d_iter%08d_fake_recon.png' % \
        (opt.exp, epoch, ganIterations), nrow=8, normalize=True)
      vutils.save_image(input.data, '%s/epoch_%08d_iter%08d_real.png' % \
        (opt.exp, epoch, ganIterations), nrow=8, normalize=True)
      vutils.save_image(recon_real.data, '%s/epoch_%08d_iter%08d_real_recon.png' % \
        (opt.exp, epoch, ganIterations), nrow=8, normalize=True)

  # learning rate annealing
  if epoch > opt.annealStart:
    adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
    adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)

  # do checkpointing
  torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
  torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.exp, epoch))
trainLogger.close()
