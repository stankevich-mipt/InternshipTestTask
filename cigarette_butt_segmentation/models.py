import torch
import torch.nn as nn


class UNet2(torch.nn.Module):
    """
    A class with implementation of slightly modified U-Net architecture
    See https://arxiv.org/pdf/1505.04597.pdf for original article
    In this version I use 3x3 kernels with padding 1 to preserve
    image size
    """

    def __init__(self, activation='ReLU'):
      
        super(UNet2, self).__init__()

        if activation == 'ReLU':
            activation_function  = nn.ReLU()
        elif activation == 'LeakyReLU':
            activation_function  = nn.LeakyReLU()
        else:
            raise NotImplementedError

        self.sigmoid = nn.Sigmoid()

        # ----------------- Pooling  -------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128)
        )

        self.conv3 = nn.Sequential(       
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(512)
        )


        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        


        # ------------- bottom layer -----------------
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(1024)
        )
      
        # ---------------------------------------------
        # ----------------- Upsampling ----------------

        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(512)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256)  
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_function,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_function
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_function,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_function,
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )


        self.upsampling1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(512)
        )

        self.upsampling2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256)
        )

        self.upsampling3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128)
        )

        self.upsampling4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(self.pooling1(x1))
        x3 = self.conv3(self.pooling2(x2))
        x4 = self.conv4(self.pooling3(x3))
        x5 = self.conv5(self.pooling4(x4))
        x6 = self.conv6(torch.cat((x4, self.upsampling1(x5)), 1))
        x7 = self.conv7(torch.cat((x3, self.upsampling2(x6)), 1))
        x8 = self.conv8(torch.cat((x2, self.upsampling3(x7)), 1))
        x9 = self.conv9(torch.cat((x1, self.upsampling4(x8)), 1))
        return self.sigmoid(x9)


class UNetShallow(torch.nn.Module):
    """
    Implementation of shallow UNet-like architecture
    See https://arxiv.org/pdf/1505.04597.pdf for original article
    In this version I use 3x3 kernels with padding 1 to preserve
    image size
    """

    def __init__(self, activation='ReLU'):
      
        super(UNetShallow, self).__init__()

        if activation == 'ReLU':
            activation_function  = nn.ReLU()
        elif activation == 'LeakyReLU':
            activation_function  = nn.LeakyReLU()
        else:
            raise NotImplementedError

        self.sigmoid = nn.Sigmoid()

        # ----------------- Pooling  -------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32)
        )

        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)      


        # ------------- bottom layer -----------------
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5)
        )
      
        # ---------------------------------------------
        # ----------------- Upsampling ----------------

       
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_function,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_function
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            activation_function,
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            activation_function,
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        self.upsampling1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32)
        )

        self.upsampling2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(16)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(self.pooling1(x1))
        x3 = self.conv3(self.pooling2(x2))
        x4 = self.conv4(torch.cat((x2, self.upsampling1(x3)), 1))
        x5 = self.conv5(torch.cat((x1, self.upsampling2(x4)), 1))
        
        return self.sigmoid(x5)


class UNet3(torch.nn.Module):
    """
    A class with implementation of modified U-Net architecture
    See https://arxiv.org/pdf/1505.04597.pdf for original article
    In this version I use 3x3 kernels with padding 1 to preserve
    image size. Due to limited GPU memory, 4x4 maxpool is used on the
    first two pooling layers to reduce computational cost
    """

    def __init__(self, activation='ReLU'):
      
        super(UNet3, self).__init__()

        if activation == 'ReLU':
            activation_function  = nn.ReLU()
        elif activation == 'LeakyReLU':
            activation_function  = nn.LeakyReLU()
        else:
            raise NotImplementedError

        self.sigmoid = nn.Sigmoid()

        # ----------------- Pooling  -------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(       
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128)
        )


        self.pooling1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pooling2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        


        # ------------- bottom layer -----------------
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256)
        )
      
        # ---------------------------------------------
        # ----------------- Upsampling ----------------

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64)  
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_function,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_function
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            activation_function,
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            activation_function,
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )


        self.upsampling1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128)
        )

        self.upsampling2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64)
        )

        self.upsampling3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32)
        )

        self.upsampling4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(16)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(self.pooling1(x1))
        x3 = self.conv3(self.pooling2(x2))
        x4 = self.conv4(self.pooling3(x3))
        x5 = self.conv5(self.pooling4(x4))
        x6 = self.conv6(torch.cat((x4, self.upsampling1(x5)), 1))
        x7 = self.conv7(torch.cat((x3, self.upsampling2(x6)), 1))
        x8 = self.conv8(torch.cat((x2, self.upsampling3(x7)), 1))
        x9 = self.conv9(torch.cat((x1, self.upsampling4(x8)), 1))
        return self.sigmoid(x9)


class UNet4(torch.nn.Module):
    """
    A class with implementation of modified U-Net architecture
    See https://arxiv.org/pdf/1505.04597.pdf for original article
    In this version I use 3x3 kernels with padding 1 to preserve
    image size. Also all upconv layers are replaced with transposed
    covolutions. Due to limited GPU memory, 4x4 maxpool is used on the
    first two pooling layers to reduce computational cost
    """

    def __init__(self, activation='ReLU'):
      
        super(UNet4, self).__init__()

        if activation == 'ReLU':
            activation_function  = nn.ReLU()
        elif activation == 'LeakyReLU':
            activation_function  = nn.LeakyReLU()
        else:
            raise NotImplementedError

        self.sigmoid = nn.Sigmoid()

        # ----------------- Pooling  -------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(       
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128)
        )


        self.pooling1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pooling2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        


        # ------------- bottom layer -----------------
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(256)
        )
      
        # ---------------------------------------------
        # ----------------- Upsampling ----------------

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(128)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_function,
            nn.BatchNorm2d(64)  
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_function,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_function
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            activation_function,
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            activation_function,
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )


        self.transposed1 = nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            activation_function,
            nn.BatchNorm2d(128)
        )

        self.transposed2 = nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            activation_function,
            nn.BatchNorm2d(64)
        )

        self.transposed3 = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            activation_function,
            nn.BatchNorm2d(32)
        )

        self.transposed4 = nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            activation_function,
            nn.BatchNorm2d(16)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(self.pooling1(x1))
        x3 = self.conv3(self.pooling2(x2))
        x4 = self.conv4(self.pooling3(x3))
        x5 = self.conv5(self.pooling4(x4))
        x6 = self.conv6(torch.cat((x4, self.transposed1(x5)), 1))
        x7 = self.conv7(torch.cat((x3, self.transposed2(x6)), 1))
        x8 = self.conv8(torch.cat((x2, self.transposed3(x7)), 1))
        x9 = self.conv9(torch.cat((x1, self.transposed4(x8)), 1))
        return self.sigmoid(x9)