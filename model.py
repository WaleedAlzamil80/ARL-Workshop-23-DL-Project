import torch
from torch import nn

def reshape_images(images: torch.Tensor) -> torch.Tensor:
    """
    Reshape a tensor of images to a new shape.

    Args:
    images (torch.Tensor): Input images tensor. Should be of shape (3, 3, 160, 320).

    Returns:
    torch.Tensor: The reshaped images tensor which your model will accept.
    """
    reshaped_images = images.reshape((3, 3, 160, 320))
    return reshaped_images


## Put your Architecture here

## Here is a sample and this one not supposed to work so delete it and put your own
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

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