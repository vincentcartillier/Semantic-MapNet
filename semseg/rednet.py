import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.checkpoint import checkpoint


class RedNet(nn.Module):
    def __init__(self, cfg):

        super(RedNet, self).__init__()

        num_classes = cfg["n_classes"]
        pretrained = cfg["resnet_pretrained"]

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64 * 4, 64)
        self.agant2 = self._make_agant_layer(128 * 4, 128)
        self.agant3 = self._make_agant_layer(256 * 4, 256)
        self.agant4 = self._make_agant_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv_custom = nn.ConvTranspose2d(
            self.inplanes, num_classes, kernel_size=2, stride=2, padding=0, bias=True
        )

        self.out5_conv_custom = nn.Conv2d(
            256, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out4_conv_custom = nn.Conv2d(
            128, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out3_conv_custom = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out2_conv_custom = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, bias=True
        )

        if pretrained:
            self._load_resnet_pretrained()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes,
                    kernel_size=2,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        return layers

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(utils.model_urls["resnet50"])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith("conv1"):  # the first conv_op
                    model_dict[k] = v
                    model_dict[k.replace("conv1", "conv1_d")] = torch.mean(
                        v, 1
                    ).data.view_as(state_dict[k.replace("conv1", "conv1_d")])

                elif k.startswith("bn1"):
                    model_dict[k] = v
                    model_dict[k.replace("bn1", "bn1_d")] = v
                elif k.startswith("layer"):
                    model_dict[k] = v
                    model_dict[k[:6] + "_d" + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward_downsample(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        fuse0 = x + depth

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        depth = self.layer1_d(depth)
        fuse1 = x + depth
        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        fuse2 = x + depth
        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        fuse3 = x + depth
        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        fuse4 = x + depth

        agant4 = self.agant4(fuse4)

        return fuse0, fuse1, fuse2, fuse3, agant4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, agant4):

        # upsample 1
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv_custom(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv_custom(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv_custom(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv_custom(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)

        last_layer = x

        out = self.final_deconv_custom(x)

        if self.training:
            return out, out2, out3, out4, out5

        return out, last_layer

    def forward(self, rgb, depth):

        fuses = self.forward_downsample(rgb, depth)
        features_encoder = fuses[-1]
        scores, features_lastlayer = self.forward_upsample(*fuses)

        return features_lastlayer


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1,
                bias=False,
            )
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
