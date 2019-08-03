import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

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


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class D(nn.Module):
  def __init__(self, nc, ndf, hidden_size):
    super(D, self).__init__()

    # 256
    self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                               nn.ELU(True))
    # 256
    self.conv2 = conv_block(ndf,ndf)
    # 128
    self.conv3 = conv_block(ndf, ndf*2)
    # 64
    self.conv4 = conv_block(ndf*2, ndf*3)
    # 32
    self.encode = nn.Conv2d(ndf*3, hidden_size, kernel_size=1,stride=1,padding=0)
    self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1,stride=1,padding=0)
    # 32
    self.deconv4 = deconv_block(ndf, ndf)
    # 64
    self.deconv3 = deconv_block(ndf, ndf)
    # 128
    self.deconv2 = deconv_block(ndf, ndf)
    # 256
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())
    """
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,nc,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh())
    """
  def forward(self,x):
    out1 = self.conv1(x) 
    out2 = self.conv2(out1)
    out3 = self.conv3(out2)
    out4 = self.conv4(out3)
    out5 = self.encode(out4)
    dout5= self.decode(out5)
    dout4= self.deconv4(dout5)
    dout3= self.deconv3(dout4)
    dout2= self.deconv2(dout3)
    dout1= self.deconv1(dout2)
    return dout1


class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()

    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 2 x  2
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

    #import pdb; pdb.set_trace()
    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 32
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 128
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
    dlayer1.add_module('%s_tanh' % name, nn.Tanh())

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    return dout1
