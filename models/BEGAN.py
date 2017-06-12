import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

def conv_block(in_dim, out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim, out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


class Encoder(nn.Module):
  def __init__(self, nc, ndf, hidden_size):
    super(Encoder, self).__init__()

    # 256
    self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                               nn.ELU(True))
    # 256
    self.conv2 = conv_block(ndf, ndf)
    # 128
    self.conv3 = conv_block(ndf, ndf*2)
    # 64
    self.conv4 = conv_block(ndf*2, ndf*3)
    # 32
    self.conv5 = conv_block(ndf*3, ndf*4)
    # 16
    #self.conv6 = conv_block(ndf*4, ndf*4)
    # 8
    self.encode = nn.Conv2d(ndf*4, hidden_size, kernel_size=8,stride=1,padding=0)
    # 1

  def forward(self, x):
    x = self.conv1(x) 
    x = self.conv2(x) 
    x = self.conv3(x) 
    x = self.conv4(x) 
    x = self.conv5(x) 
    #x = self.conv6(x) 
    x = self.encode(x) 
    return x


class Decoder(nn.Module):
  def __init__(self, nc, ngf, hidden_size):
    super(Decoder, self).__init__()

    # 1
    self.decode = nn.ConvTranspose2d(hidden_size, ngf, kernel_size=8,stride=1,padding=0)
    # 8
    self.dconv6 = deconv_block(ngf, ngf)
    # 16
    self.dconv5 = deconv_block(ngf, ngf)
    # 32
    self.dconv4 = deconv_block(ngf, ngf)
    # 64 
    self.dconv3 = deconv_block(ngf, ngf)
    # 128 
    #self.dconv2 = deconv_block(ngf, ngf)
    # 256
    self.dconv1 = nn.Sequential(nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                                nn.ELU(True),
                                nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                                nn.ELU(True),
                                nn.Conv2d(ngf, nc,kernel_size=3, stride=1,padding=1),
                                nn.Tanh())

  def forward(self, x):
    x = self.decode(x) 
    x = self.dconv6(x) 
    x = self.dconv5(x) 
    x = self.dconv4(x) 
    x = self.dconv3(x) 
    #x = self.dconv2(x) 
    x = self.dconv1(x) 
    return x


class D(nn.Module):
  def __init__(self, nc, ndf, ngf, hidden_size):
    super(D, self).__init__()

    enc = Encoder(nc, ndf, hidden_size)
    dec = Decoder(nc, ngf, hidden_size)
    self.encoder = enc
    self.decoder = dec

  def forward(self, x):
    h = self.encoder(x)
    out = self.decoder(h)
    return out
