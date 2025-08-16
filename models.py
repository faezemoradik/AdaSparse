
import torch
from torch import nn
import warnings
import torch.nn.functional as F


#-------------------------------------------
class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats = False)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats = False)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion*planes, track_running_stats = False)
      ) 
  def forward(self, x):
    out = nn.functional.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = nn.functional.relu(out)
    return out
#--------------------------------------------
class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 16
    # torch.random.manual_seed(0)
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16, track_running_stats = False)
    self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    self.linear = nn.Linear(64*block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion      
    return nn.Sequential(*layers)

  def forward(self, x):
    out = nn.functional.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = nn.functional.avg_pool2d(out, 8)
    out = out.reshape(out.size(0), -1)
    out = self.linear(out)
    return out
#-------------------------------------------------
class LogisticRegression(torch.nn.Module):
  def __init__(self):
    super(LogisticRegression, self).__init__()
    torch.random.manual_seed(0)
    self.linear = torch.nn.Linear(784, 10)
    # nn.init.zeros_(self.linear.weight)
    # nn.init.zeros_(self.linear.bias)
  def forward(self, x):
    output = self.linear(x)
    return output
# --------------------------------
def conv_bn_act(in_channels, out_channels, pool=False, act_func=nn.Mish, num_groups=None):
  if num_groups is not None:
    warnings.warn("num_groups has no effect with BatchNorm")
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats = False),
            act_func(),]
  if pool:
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

# --------------------------------
def conv_gn_act(in_channels, out_channels, pool=False, act_func=nn.Mish, num_groups=32):
  """Conv-GroupNorm-Activation"""
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            act_func(),]
  if pool:
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)
#--------------------------------
class ResNet9(nn.Module):
  def __init__(self, in_channels: int = 3,
               num_classes: int = 10,
               act_func: nn.Module = nn.Mish,
               scale_norm: bool = False,
               norm_layer: str = "batch",
               num_groups: tuple[int, ...] = (32, 32, 32, 32),):
              """9-layer Residual Network. Architecture:
              conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC
              Args:
                  in_channels (int, optional): Channels in the input image. Defaults to 3.
                  num_classes (int, optional): Number of classes. Defaults to 10.
                  act_func (nn.Module, optional): Activation function to use. Defaults to nn.Mish.
                  scale_norm (bool, optional): Whether to add an extra normalisation layer after each residual block. Defaults to False.
                  norm_layer (str, optional): Normalisation layer. One of `batch` or `group`. Defaults to "batch".
                  num_groups (tuple[int], optional): Number of groups in GroupNorm layers.\
                  Must be a tuple with 4 elements, corresponding to the GN layer in the first conv block, \
                  the first res block, the second conv block and the second res block. Defaults to (32, 32, 32, 32).
              """
              super(ResNet9, self).__init__()

              if norm_layer == "batch":
                conv_block = conv_bn_act
              elif norm_layer == "group":
                conv_block = conv_gn_act
              else:
                raise ValueError("`norm_layer` must be `batch` or `group`")

              assert (
                  isinstance(num_groups, tuple) and len(num_groups) == 4
              ), "num_groups must be a tuple with 4 members"
              groups = num_groups

              self.conv1 = conv_block(
                  in_channels, 64, act_func=act_func, num_groups=groups[0]
              )
              self.conv2 = conv_block(
                  64, 128, pool=True, act_func=act_func, num_groups=groups[0]
              )

              self.res1 = nn.Sequential(
                  *[
                      conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
                      conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
                  ]
              )

              self.conv3 = conv_block(
                  128, 256, pool=True, act_func=act_func, num_groups=groups[2])
              self.conv4 = conv_block(
                  256, 256, pool=True, act_func=act_func, num_groups=groups[2])

              self.res2 = nn.Sequential(
                  *[conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
                      conv_block(256, 256, act_func=act_func, num_groups=groups[3]),])

              self.MP = nn.AdaptiveMaxPool2d((2, 2))
              self.FlatFeats = nn.Flatten()
              self.classifier = nn.Linear(1024, num_classes)

              if scale_norm:
                self.scale_norm_1 = (
                    nn.BatchNorm2d(128)
                    if norm_layer == "batch"
                    else nn.GroupNorm(min(num_groups[1], 128), 128))  # type:ignore
                
                self.scale_norm_2 = (
                    nn.BatchNorm2d(256)
                    if norm_layer == "batch"
                    else nn.GroupNorm(min(groups[3], 256), 256))  # type:ignore
              else:
                self.scale_norm_1 = nn.Identity()  # type:ignore
                self.scale_norm_2 = nn.Identity()  # type:ignore

  def forward(self, xb):
    out = self.conv1(xb)
    out = self.conv2(out)
    out = self.res1(out) + out
    out = self.scale_norm_1(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.res2(out) + out
    out = self.scale_norm_2(out)
    out = self.MP(out)
    out_emb = self.FlatFeats(out)
    out = self.classifier(out_emb)
    return out
  
# --------------------------------