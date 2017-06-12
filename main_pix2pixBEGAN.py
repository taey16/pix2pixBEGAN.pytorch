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

import models.pix2pixBEGAN as net
from misc import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='pix2pix',  help='dataset name (It does not need to be modified)')
parser.add_argument('--dataroot', default='', help='path to trn dataset')
parser.add_argument('--valDataroot', default='', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=64, help='val. input batch size')
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
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=1, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=0.1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--lambda_k', type=float, default=0.001, help='learning rate of k')
parser.add_argument('--gamma', type=float, default=0.7, help='balance bewteen D and G')
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

# NOTE get models
netG = net.G(inputChannelSize, outputChannelSize, ngf)
netG.apply(weights_init)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)
netD = net.D(inputChannelSize, ndf, opt.hidden_size)
netD.apply(weights_init)
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))
print(netD)
criterionCAE = nn.L1Loss()

netG.train()
netD.train()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

# NOTE get sample buffer
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netD.cuda()
netG.cuda()
criterionCAE.cuda()
target, input = target.cuda(), input.cuda()
val_target, val_input = val_target.cuda(), val_input.cuda()

target = Variable(target)
input = Variable(input)

# get randomly sampled validation images
val_iter = iter(valDataloader)
data_val = val_iter.next()
if opt.mode == 'B2A':
  val_target_cpu, val_input_cpu = data_val
elif opt.mode == 'A2B':
  val_input_cpu, val_target_cpu = data_val
val_target_cpu, val_input_cpu = val_target_cpu.cuda(), val_input_cpu.cuda()
val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

# get optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.0)

# NOTE training loop
ganIterations = 0
k = 0 # control how much emphasis is put on L(G(z_D)) during gradient descent.
M_global = AverageMeter() # 
for epoch in range(opt.niter):
  # learning rate annealing
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
    # get paired data
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)

    # max_D first
    for p in netD.parameters(): 
      p.requires_grad = True 
    netD.zero_grad()

    # NOTE: compute L_D
    recon_real = netD(target)
    x_hat = netG(input)
    fake = x_hat.detach()
    fake = Variable(imagePool.query(fake.data)) # sample from image buffer
    recon_fake = netD(fake)
    # compute L(x)
    errD_real = torch.mean(torch.abs(recon_real - target))
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

    # NOTE compute L_L1 (eq.(4) in the pix2pix paper
    L_img_ = criterionCAE(x_hat, target)
    L_img = lambdaIMG * L_img_
    if lambdaIMG <> 0: 
      #L_img.backward(retain_graph=True) # in case of current version of pytorch
      L_img.backward(retain_variables=True)

    # NOTE compute L_G
    recon_fake = netD(x_hat) # reuse previously computed x_hat
    errG_ = torch.mean(torch.abs(recon_fake - x_hat))
    errG = lambdaGAN * errG_
    if lambdaGAN <> 0:
      errG.backward()
    # update praams
    optimizerG.step()
    ganIterations += 1

    # NOTE compute k_t and M_global
    balance = (opt.gamma * errD_real - errD_fake).data[0]
    k = min(max(k + opt.lambda_k * balance, 0), 1)
    measure = errD_real.data[0] + np.abs(balance)
    M_global.update(measure, target.size(0))

    # logging
    if ganIterations % opt.display == 0:
      print('[%d/%d][%d/%d] Ld: %f Lg: %f Limg: %f, M_global: %f(%f), K: %f, balance.: %f lr: %f'
          % (epoch, opt.niter, i, len(dataloader),
             errD.data[0], errG.data[0], L_img.data[0],
             measure, M_global.avg, k, balance,
             optimizerG.param_groups[0]['lr']))
      sys.stdout.flush()
      trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                        (i, errD.data[0], errG.data[0], L_img.data[0], measure, M_global.avg, k, balance))
      trainLogger.flush()
    if ganIterations % opt.evalIter == 0:
      # NOTE generate samples with current G
      val_batch_output = torch.FloatTensor(val_input.size(0)*5, 
                                           3, 
                                           val_input.size(2), 
                                           val_input.size(3)).fill_(0)
      for idx in range(val_input.size(0)):
        single_img = val_input[idx,:,:,:].unsqueeze(0)
        target_img = val_target[idx,:,:,:].unsqueeze(0)
        val_inputv = Variable(single_img, volatile=True)
        val_targetv= Variable(target_img, volatile=True)
        recon_real = netD(val_targetv)
        x_hat_val  = netG(val_inputv)
        recon_fake = netD(x_hat_val)

        val_batch_output[idx*5+0,:,:,:].copy_(val_inputv.data.squeeze(0))
        val_batch_output[idx*5+1,:,:,:].copy_(val_targetv.data.squeeze(0))
        val_batch_output[idx*5+2,:,:,:].copy_(recon_real.data.squeeze(0))
        val_batch_output[idx*5+3,:,:,:].copy_(x_hat_val.data.squeeze(0))
        val_batch_output[idx*5+4,:,:,:].copy_(recon_fake.data.squeeze(0))
      vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' % \
        (opt.exp, epoch, ganIterations), nrow=10, normalize=True)

  # do checkpointing
  torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
  torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.exp, epoch))
trainLogger.close()
