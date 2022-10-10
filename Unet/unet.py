from torch import nn
import torch

class Conv2d_Block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3):
    super().__init__()
    self.conv2d = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size) , padding='same'),
      nn.ReLU(inplace=True), 
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), padding='same'),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv2d(x)

class Encoder_Block(nn.Module):
  def __init__(self, in_channels, out_channels, pool_size=(2,2) , dropout=0.3):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout, inplace=True)
    self.max_pool = nn.MaxPool2d(pool_size)
    self.conv2d_block = Conv2d_Block(in_channels, out_channels)

  def forward(self, x):
    f = self.conv2d_block(x)
    P = self.max_pool(f)
    P = self.dropout(P)

    return f, P

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder_block_1 = Encoder_Block(in_channels=3, out_channels=64, pool_size=(2,2) , dropout=0.3)
    self.encoder_block_2 = Encoder_Block(in_channels=64, out_channels=128, pool_size=(2,2) , dropout=0.3)
    self.encoder_block_3 = Encoder_Block(in_channels=128, out_channels=256, pool_size=(2,2) , dropout=0.3)
    self.encoder_block_4 = Encoder_Block(in_channels=256, out_channels=512, pool_size=(2,2) , dropout=0.3)

  def forward(self, x):
    f1, P1 = self.encoder_block_1(x)
    f2, P2 = self.encoder_block_2(P1)
    f3, P3 = self.encoder_block_3(P2)
    f4, P4 = self.encoder_block_4(P3)

    return P4, (f1, f2, f3, f4)

class Bottle_Neck(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv2d_block = Conv2d_Block(in_channels=512, out_channels=1024)

  def forward(self, x):
    bottleneck = self.conv2d_block(x)
    
    return bottleneck

class Decoder_Block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=(3, 3), strides=2, padding=1, output_padding=1, dropout=0.3):
    super().__init__()
    self.conv2d_block = Conv2d_Block(in_channels, out_channels)
    self.u = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                stride=strides, padding=padding, output_padding=output_padding)
    # self.dropout = nn.Dropout(p=dropout, inplace=True)

  def forward(self, x, conv_outputs):
    c = torch.cat([self.u(x), conv_outputs], 1)
    # c = self.dropout(c),
    c = self.conv2d_block(c)
    
    return c

class Decoder(nn.Module):
  def __init__(self, last_out_channels):
    super().__init__()
    self.decoder_block_1 = Decoder_Block(in_channels=1024, out_channels=512)
    self.decoder_block_2 = Decoder_Block(in_channels=512, out_channels=256)
    self.decoder_block_3 = Decoder_Block(in_channels=256, out_channels=128)
    self.decoder_block_4 = Decoder_Block(in_channels=128, out_channels=64)    
    self.conv2d_output = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=last_out_channels, kernel_size=(1,1)),
      nn.Sigmoid()
    )

  def forward(self, x, convs):
    f1, f2, f3, f4 = convs
    c6 = self.decoder_block_1(x, f4)
    c7 = self.decoder_block_2(c6, f3)
    c8 = self.decoder_block_3(c7, f2)
    c9 = self.decoder_block_4(c8, f1)
    outputs = self.conv2d_output(c9)

    return outputs

LAST_OUT_CHANNELS = 1

class UNet(nn.Module):
  def __init__(self, LAST_OUT_CHANNELS):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder(LAST_OUT_CHANNELS)
    self.bottle_neck = Bottle_Neck()

  def forward(self, x):
    encoder_output, convs = self.encoder(x)
    bottleneck = self.bottle_neck(encoder_output)
    outputs = self.decoder(bottleneck, convs)

    return outputs