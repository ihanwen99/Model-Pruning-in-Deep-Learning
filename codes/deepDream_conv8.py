import math
from torchvision.ops.boxes import batched_nms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import json

# import instaboost

PATH = '/home/hwliu/workspace/network/CAM/cifar_net_gpu.pth'


class VGG19(nn.Module):
    '''
      Creates VGG19 acrh model
    '''

    def __init__(self, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # conv 1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # conv 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # conv 3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # conv 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # conv 5
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv 6
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv 7
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # conv 9
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv 10
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv 11
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv 13
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv 14
            nn.ReLU(inplace=True),
        )

        self.target = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv 15 - 研究的target
        )

        self.left = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # conv 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.target(x)
        target_weight = self.target[0].weight
        x = self.left(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, target_weight

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DeepDream():
    '''
    Given a label (number between 0 and 1000) of the ImageNet and
    an input image(zero image by default),label specific 'deep dream'
    images can be created

    Following is an example implementation:
    a = DeepDream()
    dreamtImage = a.dream(130)  # 130 is the label for flamingo, see https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    a.show() # shows image
    a.save(dreamtImage,"myImage.png") # saves image
    '''

    def __init__(self):
        self.device = None
        self.net = None
        self.zeroImage = None
        self.gaussian_filter = None
        self.ouputImage = None
        self.setDevice()
        self.createNet()
        self.createGaussianFilter()

    def setDevice(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used to run this program: ", self.device)

    def createNet(self):
        print("Loading the network...")

        self.net = VGG19()
        pretrained_dict = torch.load(PATH)
        pretrained_dict['target.0.weight'] = pretrained_dict['features.32.weight']
        pretrained_dict['target.0.bias'] = pretrained_dict['features.32.bias']

        pretrained_dict['left.1.weight'] = pretrained_dict['features.32.weight']
        pretrained_dict['left.1.bias'] = pretrained_dict['features.32.bias']
        model_dict = self.net.state_dict()
        # i=0
        # for k, v in model_dict.items():
        #     print(i,k)
        #     i+=1

        cut_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # filter out unnecessary keys
        self.net.load_state_dict(cut_dict)
        self.net.eval()  # inference mode

        if self.net != None:
            self.net.to(self.device)
            print("Vgg19 created, weights and bias set to Cifer10 trained Vgg19")

    def dream(self, label=1, nItr=500, lr=0.1):
        """Does activation maximization on a specific label
        """
        self.createInputImage()
        im = self.prepInputImage()
        im = im.to(self.device)

        img = Variable(im.unsqueeze(0), requires_grad=True)

        # offset by the min value to 0 (shift data to positive)
        min_val = torch.min(img.data)
        img.data = img.data - min_val

        print("Dreaming...")

        for i in range(nItr):
            optimizer = torch.optim.SGD([img], lr)
            out, _ = self.net(img)
            loss = -torch.sum(out[0, label])

            loss.backward()
            optimizer.step()

            img.data = self.gaussian_filter(img.data)

            img.grad.data.zero_()

        self.outputImage = self.postProcess(img)

        return self.outputImage

    def label_img(self, img, label):
        """Does activation maximization on a specific label
        """
        zeroImage_np = np.array(img[0])
        # zeroImage_np = plt.imread(img)
        self.zeroImage = Image.fromarray((zeroImage_np).astype('uint8'), 'RGB')
        self.zeroImage.save(str(label), 'PNG')
        im = self.prepInputImage()
        im = im.to(self.device)
        img = Variable(im.unsqueeze(0), requires_grad=True)

        out, activation = self.net(img)
        out[0, label].backward()
        return activation

    def label_img_hanwen(self, img, label):
        """Return Gradient
        """
        img = np.array(img[0])
        # 原来的注释 zeroImage_np = plt.imread(img)

        # hanwen：只是利用原来的预处理方式 - 图片还是在的
        self.zeroImage = Image.fromarray((img).astype('uint8'), 'RGB')
        im = self.prepInputImage()
        im = im.to(self.device)
        img = Variable(im.unsqueeze(0), requires_grad=True)

        out, target_weight = self.net(img)

        out[0, label].backward()  # 是一个标量，求导反向传播，return 我们想要的
        ans=target_weight.grad.clone()
        target_weight.grad.zero_()
        return ans

    def createInputImage(self):
        zeroImage_np = np.ones((32, 32, 3))
        self.zeroImage = Image.fromarray((zeroImage_np).astype('uint8'), 'RGB')

    def prepInputImage(self):
        # standard normalization for ImageNet data
        normalise = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalise
        ])

        return preprocess(self.zeroImage)

    def createGaussianFilter(self, kernelSize=3, sigma=0.5):

        # Create a x, y coordinate grid of shape (kernelSize, kernelSize, 2)
        x_cord = torch.arange(kernelSize)
        x_grid = x_cord.repeat(kernelSize).view(kernelSize, kernelSize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.float()

        mean = (kernelSize - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernelSize, kernelSize)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        pad = math.floor(kernelSize / 2)

        gauss_filter = nn.Conv2d(in_channels=3, out_channels=3, padding=pad,
                                 kernel_size=kernelSize, groups=3, bias=False)

        gauss_filter.weight.data = gaussian_kernel
        gauss_filter.weight.requires_grad = False
        self.gaussian_filter = gauss_filter.to(self.device)
        # print("gaussian_filter created")

    def postProcess(self, image):
        image = image.data.squeeze()  # remove the batch dimension

        image.transpose_(0, 1)  # convert from CxHxW to HxWxC format
        image.transpose_(1, 2)

        image = image.cpu()  # back to host

        # TRUNCATE TO THROW OFF DATA OUTSIDE 5 SIGMA
        mean = torch.mean(image)
        std = torch.std(image)
        upper_limit = mean + 5 * std
        lower_limit = mean - 5 * std
        image.data = torch.clamp_(image.data, lower_limit, upper_limit)

        # normalize data to lie between 0 and 1
        image.data = (image.data - lower_limit) / (10 * std)

        img = Image.fromarray((image.data.numpy() * 255).astype('uint8'), 'RGB')  # torch tensor to PIL image

        return img

    def show(self):
        plt.figure(num=1, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
        plt.imshow(np.asarray(self.outputImage))

    def save(self, image, fileName):
        output = image.resize((224, 224), Image.ANTIALIAS)
        output.save(fileName, 'PNG')
        print('{} saved'.format(fileName))
