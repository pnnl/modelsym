import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
# https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb
# "Basic Net"
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))

def mCNN_k(c=64, num_classes=10):  # no Batch Norm
    return nn.Sequential(
        # Prep
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 8, num_classes, bias=True),
    )

def mCNN_k_sigmoid(c=64, num_classes=10):  # no Batch Norm
    return nn.Sequential(
        # Prep
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Sigmoid(),
        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Sigmoid(),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Sigmoid(),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Sigmoid(),
        nn.MaxPool2d(2),
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 8, num_classes, bias=True),
    )

def cwCNN_k(c=512, num_classes=10):  # no Batch Norm
    return nn.Sequential(
        # Prep
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        # Layer 1
        nn.Conv2d(c, c , kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c , c , kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c , c , kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c , num_classes, bias=True),
    )


def mCNN_bn_k(c=64, num_classes=10):
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c * 4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c * 8),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 8, num_classes, bias=True),
    )


def cwCNN_bn_k(c=512, num_classes=10):
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        # Layer 1
        nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c, num_classes, bias=True),
    )


def mCNN(c=64, num_classes=10):
    return mCNN_bn_k(c, num_classes)


def sCNN_k(c=64, num_classes=10):
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(),
        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(),
        nn.MaxPool2d(4),
        # Layer 3
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c * 4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 4, num_classes, bias=True),
    )

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.r1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.r2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.r2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.r1 = nn.ReLU(inplace=True)
        block_list = (
            self._make_layer(block, 16, num_blocks[0], stride=1) +
            self._make_layer(block, 32, num_blocks[1], stride=2)+
            self._make_layer(block, 64, num_blocks[2], stride=2)
            )
        
        self.block_seq = nn.Sequential(*block_list)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers
        # return nn.Sequential(*layers)

    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        out = self.block_seq(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.r1 = nn.ReLU(inplace=True)
        block_list = (
            self._make_layer(block, 64, num_blocks[0], stride=1) +
            self._make_layer(block, 128, num_blocks[1], stride=2) +
            self._make_layer(block, 256, num_blocks[2], stride=2) +
            self._make_layer(block, 512, num_blocks[3], stride=2)
            )
        self.block_seq = nn.Sequential(*block_list)
        self.linear = nn.Linear(512, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option="B"))
            self.in_planes = planes * block.expansion
        return layers
        # return nn.Sequential(*layers)

    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        out = self.block_seq(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet18():
    return ResNetImageNet(BasicBlock, [2, 2, 2, 2])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])