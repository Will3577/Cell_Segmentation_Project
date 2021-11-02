# VGG model from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_vgg_implementation.py

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 2

        # define frequently used functions
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) 

        # specify layers
        self.layers = [16,32,64,128,256]
        self.linear_layers = [2048,1024,512]

        # first convblock
        self.convlayer1 = self.convnet_generator(1,self.layers[0])
        self.convlayer2 = self.convnet_generator(self.layers[0],self.layers[0])

        # second convblock
        self.convlayer3 = self.convnet_generator(self.layers[0],self.layers[1])
        self.convlayer4 = self.convnet_generator(self.layers[1],self.layers[1])

        # third convblock
        self.convlayer5 = self.convnet_generator(self.layers[1],self.layers[2])
        self.convlayer6 = self.convnet_generator(self.layers[2],self.layers[2])
        self.convlayer7 = self.convnet_generator(self.layers[2],self.layers[2])

        # forth convblock
        self.convlayer8 = self.convnet_generator(self.layers[2],self.layers[3])
        self.convlayer9 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer10 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer14 = self.convnet_generator(self.layers[3],self.layers[3])

        # fifth convblock
        self.convlayer11 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer12 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer13 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer15 = self.convnet_generator(self.layers[3],self.layers[3])

        # Dense layers
        self.linearlayer1 = self.linearnet_generator(512,self.linear_layers[0])
        self.linearlayer3 = self.linearnet_generator(self.linear_layers[0],self.linear_layers[1])
        self.linearlayer4 = self.linearnet_generator(self.linear_layers[1],self.linear_layers[2])
        
        self.outputlayer = nn.Linear(self.linear_layers[2],self.n_classes)

    # used to generate convolution layer and apply batchnorm and relu
    def convnet_generator(self,n_in,n_out):
        layers = [
            nn.Conv2d(in_channels=n_in,out_channels=n_out,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(n_out), # prevent gradient explode and vanish
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)
    
    # used to generate linear layer combined with batchnorm, relu and dropout
    def linearnet_generator(self,n_in,n_out):
        layers = [
            nn.Linear(n_in,n_out),
            nn.BatchNorm1d(n_out), # prevent gradient explode and vanish also improve performance
            nn.ReLU(),
            nn.Dropout(p=0.5) # prevent overfitting        
            ]
        return nn.Sequential(*layers)

    def forward(self, t):

        x = self.convlayer1(t)
        x = self.convlayer2(x)
        x = self.maxpool(x)

        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.maxpool(x)

        x = self.convlayer5(x)
        x = self.convlayer6(x)
        x = self.convlayer7(x)
        x = self.maxpool(x)
#------------------------------------
        x = self.convlayer8(x)
        x = self.convlayer9(x)
        x = self.convlayer10(x)
        x = self.convlayer14(x)
        x = self.maxpool(x)

        x = self.convlayer11(x)
        x = self.convlayer12(x)
        x = self.convlayer13(x)
        x = self.convlayer15(x)
        x = self.maxpool(x)

        x = x.view(x.size(0),-1)
        x = self.linearlayer1(x)
        x = self.linearlayer3(x)
        x = self.linearlayer4(x)

        x = self.outputlayer(x)

        return x
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG_net1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG19"])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


# code from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)