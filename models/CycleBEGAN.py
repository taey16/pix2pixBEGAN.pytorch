import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


class InstanceNormalization(nn.Module):
  """InstanceNormalization
  Improves convergence of neural-style.
  ref: https://arxiv.org/pdf/1607.08022.pdf
  """
  def __init__(self, dim, eps=1e-9, mean=1.0, std=0.02):
    super(InstanceNormalization, self).__init__()
    self.scale = nn.Parameter(torch.FloatTensor(dim))
    self.shift = nn.Parameter(torch.FloatTensor(dim))
    self.eps = eps
    self._reset_parameters(mean, std)

  def _reset_parameters(self, mean, std):
    #self.scale.data.uniform_()
    self.scale.data.normal_(mean, std)
    self.shift.data.zero_()

  def forward(self, x):
    n = x.size(2) * x.size(3)
    t = x.view(x.size(0), x.size(1), n)
    mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
    # Calculate the biased var. torch.var returns unbiased var
    var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
    scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    scale_broadcast = scale_broadcast.expand_as(x)
    shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    shift_broadcast = shift_broadcast.expand_as(x)
    out = (x - mean) / torch.sqrt(var + self.eps)
    out = out * scale_broadcast + shift_broadcast
    return out


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s.instancenorm' % name, InstanceNormalization(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class D(nn.Module):
  def __init__(self, nc, ndf, hidden_size):
    super(D, self).__init__()

    # 64 x 64  256
    self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                               nn.ELU(True),
                               conv_block(ndf,ndf))
    # 32 x 32 128
    self.conv2 = conv_block(ndf, ndf*2)
    # 16 x 16  64
    self.conv3 = conv_block(ndf*2, ndf*3)
    # 16 32
    self.encode = nn.Conv2d(ndf*3, hidden_size, kernel_size=1,stride=1,padding=0)
    self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1,stride=1,padding=0)
    # 8 x 8 32
    self.deconv1 = deconv_block(ndf, ndf)
    # 16 x 16 64
    self.deconv2 = deconv_block(ndf, ndf)
    # 64 128
    self.deconv3 = deconv_block(ndf, ndf)
    # 64 256
    self.deconv4 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())
  def forward(self,x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.encode(out)
    out = self.decode(out)
    out = self.deconv1(out)
    out = self.deconv2(out)
    out = self.deconv3(out)
    out = self.deconv4(out)
    return out


def blockStyleNet(in_c, out_c, name, transposed=False, instanceNorm=True, relu=True, reflectionPadding=False, kernelSize=3, stride=2):
  block = nn.Sequential()
  padding = (kernelSize-1)/2
  if reflectionPadding:
    block.add_module('%s.reflectionpad' % name, nn.ReflectionPad2d(padding))
    padding = 0
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, kernelSize, stride, padding, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, kernelSize, stride, padding, output_padding=padding, bias=False))
  if instanceNorm:
    block.add_module('%s.instancenorm' % name, InstanceNormalization(out_c))
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  return block


def build_conv_block(name, dim):
  block = nn.Sequential() 
  block.add_module('%s-0' % name, blockStyleNet(dim, dim, 'block0', reflectionPadding=True, kernelSize=3, stride=1))
  block.add_module('%s-1' % name, blockStyleNet(dim, dim, 'block1', relu=False, reflectionPadding=True, kernelSize=3, stride=1))
  return block


class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()

    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = blockStyleNet(input_nc, nf, name, reflectionPadding=True, kernelSize=7, stride=1)

    # 256
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockStyleNet(nf, nf*2, name)

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf *= 2
    layer3 = blockStyleNet(nf, nf*2, name)

    # 64
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    nf *= 2
    layer4_res1 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer5_res2 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer6_res3 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer7_res4 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer8_res5 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer9_res6 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer10_res7 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer11_res8 = build_conv_block(name, nf) 
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    layer12_res9 = build_conv_block(name, nf) 

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    dlayer3 = blockStyleNet(nf, nf/2, name, transposed=True)

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf /= 2
    dlayer2 = blockStyleNet(nf, nf/2, name, transposed=True)
    # 256
    name = 'layer%d' % layer_idx
    nf /= 2
    dlayer1 = blockStyleNet(nf, output_nc, name, transposed=False, relu=False, reflectionPadding=True, kernelSize=7, stride=1)
    # 256 
    dlayer1.add_module('%s.tanh' % name, nn.Tanh())

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3

    self.layer4_res1 = layer4_res1
    self.layer5_res2 = layer5_res2
    self.layer6_res3 = layer6_res3
    self.layer7_res4 = layer7_res4
    self.layer8_res5 = layer8_res5
    self.layer9_res6 = layer9_res6
    self.layer10_res7 = layer10_res7
    self.layer11_res8 = layer11_res8
    self.layer12_res9 = layer12_res9

    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)

    out4 = self.layer4_res1(out3) + out3
    out5 = self.layer5_res2(out4) + out4
    out6 = self.layer6_res3(out5) + out5
    out7 = self.layer7_res4(out6) + out6
    out8 = self.layer8_res5(out7) + out7
    out9 = self.layer9_res6(out8) + out8
    out10= self.layer10_res7(out9) + out9
    out11= self.layer11_res8(out10) + out10
    out12= self.layer12_res9(out11) + out11

    dout3= self.dlayer3(out12)
    dout2= self.dlayer2(dout3)
    dout1= self.dlayer1(dout2)
    return dout1
