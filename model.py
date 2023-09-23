import torch
from torch import nn

## Put your Architecture here

## Here is a sample and this one not supposed to work so delete it and put your own
class ARA_Model(nn.Module):
  def __init__(self):
    super(ARA_Model, self).__init__()

    self.encoder = nn.Sequential( 
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (7, 7), padding = 0),
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (7, 7), padding = 0),
        nn.MaxPool2d((2, 2 )),
        nn.ReLU(),
        )

    self.regressor = nn.Linear(4 * 4 * 512, 2)

  def forward(self, x):
    x = self.encoder(x)
    x = nn.Flatten()(x)
    x = self.regressor(x)
    return(x)