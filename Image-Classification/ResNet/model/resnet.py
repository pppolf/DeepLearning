import torch
import torch.nn as nn

# 适用于 ResNet18、ResNet34
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self, x):
        indentity = x
        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += indentity
        out = self.relu(out)

        return out

# 适用于 ResNet50、ResNet101
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        indentity = x
        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += indentity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=100, block=Bottleneck, num_blocks=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channel = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],stride=2)
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion*7*7, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.maxpool1(self.bn1(self.conv1(x)))  # (1, 3, 224, 224) -> (1, 64, 56, 56)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.shape[0], -1)  # out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
    def _make_layer(self, block, block_channel, block_num, stride):
        layers = []
        # 注意：第二个layer的输入是256通道个56*56，下采样后是512通道个28*28。
        downsample = nn.Conv2d(self.in_channel, block_channel * block.expansion, kernel_size=1, stride=stride,bias=False)

        # 先加一个带有下采样的layer
        layers += [block(self.in_channel, block_channel, stride=stride, downsample=downsample)]
        self.in_channel = block_channel * block.expansion

        # 再加block_num-1个默认不带下采样的layer，由于输入输出通道数相同，所以不需要下采样
        for _ in range(1, block_num):
            layers += [block(self.in_channel, block_channel, stride=1)]
        return nn.Sequential(*layers)
    
def resnet18(in_channel=3, num_classes=10):
    return ResNet(
        in_channel, 
        num_classes, 
        block=BasicBlock, 
        num_blocks=[2, 2, 2, 2]
    )

def resnet34(in_channel=3, num_classes=10):
    return ResNet(
        in_channel, 
        num_classes, 
        block=BasicBlock, 
        num_blocks=[3, 4, 6, 3]
    )

def resnet50(in_channel=3, num_classes=10):
    return ResNet(
        in_channel, 
        num_classes, 
        block=Bottleneck, 
        num_blocks=[3, 4, 6, 3]
    )

def resnet101(in_channel=3, num_classes=10):
    return ResNet(
        in_channel, 
        num_classes, 
        block=Bottleneck, 
        num_blocks=[3, 4, 23, 3]
    )

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    resnet18 = ResNet(in_channel=3, num_classes=10, block=BasicBlock, num_blocks=[2, 2, 2, 2])
    resnet34 = ResNet(in_channel=3, num_classes=10, block=BasicBlock, num_blocks=[3, 4, 6, 3])
    resnet50 = ResNet(in_channel=3, num_classes=10, block=Bottleneck, num_blocks=[3, 4, 6, 3])
    resnet101 = ResNet(in_channel=3, num_classes=10, block=Bottleneck, num_blocks=[3, 4, 23, 3])

    y1 = resnet18(x)
    y2 = resnet34(x)
    y3 = resnet50(x)
    y4 = resnet101(x)

    print(y1.shape, y2.shape, y3.shape, y4.shape)