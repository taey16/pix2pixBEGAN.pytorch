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

import models.UNet as net
from misc import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=64, help='input batch size')
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
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=1, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=0.1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
#opt.manualSeed = random.randint(1, 10000)
opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get dataloader
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
valDataloader = getLoader(opt.dataset, 
                          opt.valDataroot, 
                          opt.imageSize, #opt.originalSize, 
                          opt.imageSize, 
                          opt.valBatchSize, 
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# get models
netG = net.G(inputChannelSize, outputChannelSize, ngf)
netG.apply(weights_init)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)
netD = net.D(inputChannelSize + outputChannelSize, ndf)
netD.apply(weights_init)
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))
print(netD)

netG.train()
netD.train()
criterionBCE = nn.BCELoss()
criterionCAE = nn.L1Loss()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
label_d = torch.FloatTensor(opt.batchSize)
# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0

# image pool storing previously generated samples from G
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netD.cuda()
netG.cuda()
criterionBCE.cuda()
criterionCAE.cuda()
target, input, label_d = target.cuda(), input.cuda(), label_d.cuda()
val_target, val_input = val_target.cuda(), val_input.cuda()

target = Variable(target)
input = Variable(input)
label_d = Variable(label_d)

# get randomly sampled validation images and save it
val_iter = iter(valDataloader)
data_val = val_iter.next()
if opt.mode == 'B2A':
  val_target_cpu, val_input_cpu = data_val
elif opt.mode == 'A2B':
  val_input_cpu, val_target_cpu = data_val
val_target_cpu, val_input_cpu = val_target_cpu.cuda(), val_input_cpu.cuda()
val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
vutils.save_image(val_target, '%s/real_target.png' % opt.exp, normalize=True)
vutils.save_image(val_input, '%s/real_input.png' % opt.exp, normalize=True)

# get optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.0)

# NOTE training loop
ganIterations = 0
for epoch in range(opt.niter):
  if epoch > opt.annealStart:
    adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
    adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
  for i, data in enumerate(dataloader, 0):
    if opt.mode == 'B2A':
      target_cpu, input_cpu = data
    elif opt.mode == 'A2B' :
      input_cpu, target_cpu = data
    batch_size = target_cpu.size(0)

    target_cpu, input_cpu = target_cpu.cuda(), input_cpu.cuda()
    # NOTE paired samples
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)

    # max_D first
    for p in netD.parameters(): 
      p.requires_grad = True 
    netD.zero_grad()

    # NOTE: compute L_cGAN in eq.(2)
    label_d.data.resize_((batch_size, 1, sizePatchGAN, sizePatchGAN)).fill_(real_label)
    output = netD(torch.cat([target, input], 1)) # conditional
    errD_real = criterionBCE(output, label_d)
    errD_real.backward()
    D_x = output.data.mean()
    x_hat = netG(input)
    fake = x_hat.detach()
    fake = Variable(imagePool.query(fake.data))
    label_d.data.fill_(fake_label)
    output = netD(torch.cat([fake, input], 1)) # conditional
    errD_fake = criterionBCE(output, label_d)
    errD_fake.backward()
    D_G_z1 = output.data.mean()
    errD = errD_real + errD_fake
    optimizerD.step() # update parameters

    # prevent computing gradients of weights in Discriminator
    for p in netD.parameters(): 
      p.requires_grad = False
    netG.zero_grad() # start to update G

    # compute L_L1 (eq.(4) in the paper
    L_img_ = criterionCAE(x_hat, target)
    L_img = lambdaIMG * L_img_
    if lambdaIMG <> 0: 
      #L_img.backward(retain_graph=True) # in case of current version of pytorch
      L_img.backward(retain_variables=True)

    # compute L_cGAN (eq.(2) in the paper
    label_d.data.fill_(real_label)
    output = netD(torch.cat([x_hat, input], 1))
    errG_ = criterionBCE(output, label_d)
    errG = lambdaGAN * errG_ 
    if lambdaGAN <> 0:
      errG.backward()
    D_G_z2 = output.data.mean()

    optimizerG.step()
    ganIterations += 1

    if ganIterations % opt.display == 0:
      print('[%d/%d][%d/%d] L_D: %f L_img: %f L_G: %f D(x): %f D(G(z)): %f / %f'
          % (epoch, opt.niter, i, len(dataloader),
             errD.data[0], L_img.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
      sys.stdout.flush()
      trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                        (i, errD.data[0], errG.data[0], L_img.data[0], D_x, D_G_z1, D_G_z2))
      trainLogger.flush()
    if ganIterations % opt.evalIter == 0:
      val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
      for idx in range(val_input.size(0)):
        single_img = val_input[idx,:,:,:].unsqueeze(0)
        val_inputv = Variable(single_img, volatile=True)
        x_hat_val = netG(val_inputv)
        val_batch_output[idx,:,:,:].copy_(x_hat_val.data)
      vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' % \
        (opt.exp, epoch, ganIterations), normalize=True)

  # do checkpointing
  torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
  torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.exp, epoch))
trainLogger.close()
