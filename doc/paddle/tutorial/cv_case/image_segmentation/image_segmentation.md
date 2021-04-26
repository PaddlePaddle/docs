# 基于U-Net卷积神经网络实现宠物图像分割

**作者:** [PaddlePaddle](https://github.com/PaddlePaddle)<br>
**日期:** 2021.03<br>
**摘要:** 本示例教程使用U-Net实现图像分割。

## 一、简要介绍

在计算机视觉领域，图像分割指的是将数字图像细分为多个图像子区域的过程。图像分割的目的是简化或改变图像的表示形式，使得图像更容易理解和分析。图像分割通常用于定位图像中的物体和边界（线，曲线等）。更精确的，图像分割是对图像中的每个像素加标签的一个过程，这一过程使得具有相同标签的像素具有某种共同视觉特性。图像分割的领域非常多，无人车、地块检测、表计识别等等。

本示例简要介绍如何通过飞桨开源框架，实现图像分割。这里采用了一个在图像分割领域比较熟知的U-Net网络结构，是一个基于FCN做改进后的一个深度学习网络，包含下采样（编码器，特征提取）和上采样（解码器，分辨率还原）两个阶段，因模型结构比较像U型而命名为U-Net。

## 二、环境设置

导入一些比较基础常用的模块，确认自己的飞桨版本。


```python
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage

import paddle
from paddle.nn import functional as F

paddle.__version__
```




    '2.0.1'



## 三、数据集

### 3.1 数据集下载

本案例使用Oxford-IIIT Pet数据集，官网：https://www.robots.ox.ac.uk/~vgg/data/pets 。

数据集统计如下：

![alt 数据集统计信息](https://www.robots.ox.ac.uk/~vgg/data/pets/breed_count.jpg)

数据集包含两个压缩文件：

1. 原图：https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
2. 分割图像：https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz


```python
!curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz
```

### 3.2 数据集概览

首先先看看下载到磁盘上的文件结构是什么样，来了解一下数据集。

1. 首先看一下images.tar.gz这个压缩包，该文件解压后得到一个images目录，这个目录比较简单，里面直接放的是用类名和序号命名好的图片文件，每个图片是对应的宠物照片。

```bash
.
├── samoyed_7.jpg
├── ......
└── samoyed_81.jpg
```

2. 然后在看下annotations.tar.gz，文件解压后的目录里面包含以下内容，目录中的README文件将每个目录和文件做了比较详细的介绍，可以通过README来查看每个目录文件的说明。

```bash
.
├── README
├── list.txt
├── test.txt
├── trainval.txt
├── trimaps
│    ├── Abyssinian_1.png
│    ├── Abyssinian_10.png
│    ├── ......
│    └── yorkshire_terrier_99.png
└── xmls
      ├── Abyssinian_1.xml
      ├── Abyssinian_10.xml
      ├── ......
      └── yorkshire_terrier_190.xml
```

本次主要使用到images和annotations/trimaps两个目录，即原图和三元图像文件，前者作为训练的输入数据，后者是对应的标签数据。

来看看这个数据集提供了多少个训练样本。


```python
IMAGE_SIZE = (160, 160)
train_images_path = "images/"
label_images_path = "annotations/trimaps/"
image_count = len([os.path.join(train_images_path, image_name)
          for image_name in os.listdir(train_images_path)
          if image_name.endswith('.jpg')])
print("用于训练的图片样本数量:", image_count)

# 对数据集进行处理，划分训练集、测试集
def _sort_images(image_dir, image_type):
    """
    对文件夹内的图像进行按照文件名排序
    """
    files = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.{}'.format(image_type)) \
                and not image_name.startswith('.'):
            files.append(os.path.join(image_dir, image_name))

    return sorted(files)

def write_file(mode, images, labels):
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(images)):
            f.write('{}\t{}\n'.format(images[i], labels[i]))

"""
由于所有文件都是散落在文件夹中，在训练时需要使用的是数据集和标签对应的数据关系，
所以第一步是对原始的数据集进行整理，得到数据集和标签两个数组，分别一一对应。
这样可以在使用的时候能够很方便的找到原始数据和标签的对应关系，否则对于原有的文件夹图片数据无法直接应用。
在这里是用了一个非常简单的方法，按照文件名称进行排序。
因为刚好数据和标签的文件名是按照这个逻辑制作的，名字都一样，只有扩展名不一样。
"""
images = _sort_images(train_images_path, 'jpg')
labels = _sort_images(label_images_path, 'png')
eval_num = int(image_count * 0.15)

write_file('train', images[:-eval_num], labels[:-eval_num])
write_file('test', images[-eval_num:], labels[-eval_num:])
write_file('predict', images[-eval_num:], labels[-eval_num:])
```

    用于训练的图片样本数量: 7390


### 3.3 PetDataSet数据集抽样展示

划分好数据集之后，来查验一下数据集是否符合预期，通过划分的配置文件读取图片路径后再加载图片数据来用matplotlib进行展示，这里要注意的是对于分割的标签文件因为是1通道的灰度图片，需要在使用imshow接口时注意下传参cmap='gray'。


```python
with open('./train.txt', 'r') as f:
    i = 0

    for line in f.readlines():
        image_path, label_path = line.strip().split('\t')
        image = np.array(PilImage.open(image_path))
        label = np.array(PilImage.open(label_path))

        if i > 2:
            break
        # 进行图片的展示
        plt.figure()

        plt.subplot(1,2,1),
        plt.title('Train Image')
        plt.imshow(image.astype('uint8'))
        plt.axis('off')

        plt.subplot(1,2,2),
        plt.title('Label')
        plt.imshow(label.astype('uint8'), cmap='gray')
        plt.axis('off')

        plt.show()
        i = i + 1
```


![png](output_10_0.png)



![png](output_10_1.png)



![png](output_10_2.png)


### 3.4 数据集类定义

飞桨（PaddlePaddle）数据集加载方案是统一使用Dataset（数据集定义） + DataLoader（多进程数据集加载）。

首先进行数据集的定义，数据集定义主要是实现一个新的Dataset类，继承父类paddle.io.Dataset，并实现父类中以下两个抽象方法，`__getitem__`和`__len__`：

```python
class MyDataset(Dataset):
    def __init__(self):
        ...

    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, idx):
        return x, y

    # 返回整个数据集的总数
    def __len__(self):
        return count(samples)
```

在数据集内部可以结合图像数据预处理相关API进行图像的预处理（改变大小、反转、调整格式等）。

由于加载进来的图像不一定都符合自己的需求，举个例子，已下载的这些图片里面就会有RGBA格式的图片，这个时候图片就不符合所需3通道的需求，需要进行图片的格式转换，那么这里直接实现了一个通用的图片读取接口，确保读取出来的图片都是满足需求。

另外图片加载出来的默认shape是HWC，这个时候要看看是否满足后面训练的需要，如果Layer的默认格式和这个不是符合的情况下，需要看下Layer有没有参数可以进行格式调整。不过如果layer较多的话，还是直接调整原数据Shape比较好，否则每个layer都要做参数设置，如果有遗漏就会导致训练出错，那么在本案例中是直接对数据源的shape做了统一调整，从HWC转换成了CHW，因为飞桨的卷积等API的默认输入格式为CHW，这样处理方便后续模型训练。


```python
import random

from paddle.io import Dataset
from paddle.vision.transforms import transforms as T


class PetDataset(Dataset):
    """
    数据集定义
    """
    def __init__(self, mode='train'):
        """
        构造函数
        """
        self.image_size = IMAGE_SIZE
        self.mode = mode.lower()

        assert self.mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(self.mode)

        self.train_images = []
        self.label_images = []

        with open('./{}.txt'.format(self.mode), 'r') as f:
            for line in f.readlines():
                image, label = line.strip().split('\t')
                self.train_images.append(image)
                self.label_images.append(label)

    def _load_img(self, path, color_mode='rgb', transforms=[]):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        with open(path, 'rb') as f:
            img = PilImage.open(io.BytesIO(f.read()))
            if color_mode == 'grayscale':
                # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
                # convert it to an 8-bit grayscale image.
                if img.mode not in ('L', 'I;16', 'I'):
                    img = img.convert('L')
            elif color_mode == 'rgba':
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif color_mode == 'rgb':
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')

            return T.Compose([
                T.Resize(self.image_size)
            ] + transforms)(img)

    def __getitem__(self, idx):
        """
        返回 image, label
        """
        train_image = self._load_img(self.train_images[idx],
                                     transforms=[
                                         T.Transpose(),
                                         T.Normalize(mean=127.5, std=127.5)
                                     ]) # 加载原始图像
        label_image = self._load_img(self.label_images[idx],
                                     color_mode='grayscale',
                                     transforms=[T.Grayscale()]) # 加载Label图像

        # 返回image, label
        train_image = np.array(train_image, dtype='float32')
        label_image = np.array(label_image, dtype='int64')
        return train_image, label_image

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.train_images)
```

## 四、模型组网

U-Net是一个U型网络结构，可以看做两个大的阶段，图像先经过Encoder编码器进行下采样得到高级语义特征图，再经过Decoder解码器上采样将特征图恢复到原图片的分辨率。

### 4.1 定义SeparableConv2D接口

为了减少卷积操作中的训练参数来提升性能，是继承paddle.nn.Layer自定义了一个SeparableConv2D Layer类，整个过程是把`filter_size * filter_size * num_filters`的Conv2D操作拆解为两个子Conv2D，先对输入数据的每个通道使用`filter_size * filter_size * 1`的卷积核进行计算，输入输出通道数目相同，之后在使用`1 * 1 * num_filters`的卷积核计算。


```python
from paddle.nn import functional as F

class SeparableConv2D(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(SeparableConv2D, self).__init__()

        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._in_channels = in_channels
        self._data_format = data_format

        # 第一次卷积参数，没有偏置参数
        filter_shape = [in_channels, 1] + self.convert_to_list(kernel_size, 2, 'kernel_size')
        self.weight_conv = self.create_parameter(shape=filter_shape, attr=weight_attr)

        # 第二次卷积参数
        filter_shape = [out_channels, in_channels] + self.convert_to_list(1, 2, 'kernel_size')
        self.weight_pointwise = self.create_parameter(shape=filter_shape, attr=weight_attr)
        self.bias_pointwise = self.create_parameter(shape=[out_channels],
                                                    attr=bias_attr,
                                                    is_bias=True)

    def convert_to_list(self, value, n, name, dtype=np.int):
        if isinstance(value, dtype):
            return [value, ] * n
        else:
            try:
                value_list = list(value)
            except TypeError:
                raise ValueError("The " + name +
                                "'s type must be list or tuple. Received: " + str(
                                    value))
            if len(value_list) != n:
                raise ValueError("The " + name + "'s length must be " + str(n) +
                                ". Received: " + str(value))
            for single_value in value_list:
                try:
                    dtype(single_value)
                except (ValueError, TypeError):
                    raise ValueError(
                        "The " + name + "'s type must be a list or tuple of " + str(
                            n) + " " + str(dtype) + " . Received: " + str(
                                value) + " "
                        "including element " + str(single_value) + " of type" + " "
                        + str(type(single_value)))
            return value_list

    def forward(self, inputs):
        conv_out = F.conv2d(inputs,
                            self.weight_conv,
                            padding=self._padding,
                            stride=self._stride,
                            dilation=self._dilation,
                            groups=self._in_channels,
                            data_format=self._data_format)

        out = F.conv2d(conv_out,
                       self.weight_pointwise,
                       bias=self.bias_pointwise,
                       padding=0,
                       stride=1,
                       dilation=1,
                       groups=1,
                       data_format=self._data_format)

        return out
```

### 4.2 定义Encoder编码器

将网络结构中的Encoder下采样过程进行了一个Layer封装，方便后续调用，减少代码编写，下采样是有一个模型逐渐向下画曲线的一个过程，这个过程中是不断的重复一个单元结构将通道数不断增加，形状不断缩小，并且引入残差网络结构，将这些都抽象出来进行统一封装。


```python
class Encoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.relus = paddle.nn.LayerList(
            [paddle.nn.ReLU() for i in range(2)])
        self.separable_conv_01 = SeparableConv2D(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.bns = paddle.nn.LayerList(
            [paddle.nn.BatchNorm2D(out_channels) for i in range(2)])

        self.separable_conv_02 = SeparableConv2D(out_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.residual_conv = paddle.nn.Conv2D(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=2,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relus[0](inputs)
        y = self.separable_conv_01(y)
        y = self.bns[0](y)
        y = self.relus[1](y)
        y = self.separable_conv_02(y)
        y = self.bns[1](y)
        y = self.pool(y)

        residual = self.residual_conv(previous_block_activation)
        y = paddle.add(y, residual)

        return y
```

### 4.3 定义Decoder解码器

在通道数达到最大得到高级语义特征图后，网络结构会开始进行decode操作，进行上采样，通道数逐渐减小，对应图片尺寸逐步增加，直至恢复到原图像大小，那么这个过程里面也是通过不断的重复相同结构的残差网络完成，也是为了减少代码编写，将这个过程定义一个Layer来放到模型组网中使用。


```python
class Decoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.relus = paddle.nn.LayerList(
            [paddle.nn.ReLU() for i in range(2)])
        self.conv_transpose_01 = paddle.nn.Conv2DTranspose(in_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding=1)
        self.conv_transpose_02 = paddle.nn.Conv2DTranspose(out_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding=1)
        self.bns = paddle.nn.LayerList(
            [paddle.nn.BatchNorm2D(out_channels) for i in range(2)]
        )
        self.upsamples = paddle.nn.LayerList(
            [paddle.nn.Upsample(scale_factor=2.0) for i in range(2)]
        )
        self.residual_conv = paddle.nn.Conv2D(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relus[0](inputs)
        y = self.conv_transpose_01(y)
        y = self.bns[0](y)
        y = self.relus[1](y)
        y = self.conv_transpose_02(y)
        y = self.bns[1](y)
        y = self.upsamples[0](y)

        residual = self.upsamples[1](previous_block_activation)
        residual = self.residual_conv(residual)

        y = paddle.add(y, residual)

        return y
```

### 4.4 训练模型组网

按照U型网络结构格式进行整体的网络结构搭建，三次下采样，四次上采样。


```python
class PetNet(paddle.nn.Layer):
    def __init__(self, num_classes):
        super(PetNet, self).__init__()

        self.conv_1 = paddle.nn.Conv2D(3, 32,
                                       kernel_size=3,
                                       stride=2,
                                       padding='same')
        self.bn = paddle.nn.BatchNorm2D(32)
        self.relu = paddle.nn.ReLU()

        in_channels = 32
        self.encoders = []
        self.encoder_list = [64, 128, 256]
        self.decoder_list = [256, 128, 64, 32]

        # 根据下采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.encoder_list:
            block = self.add_sublayer('encoder_{}'.format(out_channels),
                                      Encoder(in_channels, out_channels))
            self.encoders.append(block)
            in_channels = out_channels

        self.decoders = []

        # 根据上采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.decoder_list:
            block = self.add_sublayer('decoder_{}'.format(out_channels),
                                      Decoder(in_channels, out_channels))
            self.decoders.append(block)
            in_channels = out_channels

        self.output_conv = paddle.nn.Conv2D(in_channels,
                                            num_classes,
                                            kernel_size=3,
                                            padding='same')

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.bn(y)
        y = self.relu(y)

        for encoder in self.encoders:
            y = encoder(y)

        for decoder in self.decoders:
            y = decoder(y)

        y = self.output_conv(y)
        return y
```

### 4.5 模型可视化

调用飞桨提供的summary接口对组建好的模型进行可视化，方便进行模型结构和参数信息的查看和确认。


```python
num_classes = 4
network = PetNet(num_classes)
model = paddle.Model(network)
model.summary((-1, 3,) + IMAGE_SIZE)
```

    -----------------------------------------------------------------------------
      Layer (type)        Input Shape          Output Shape         Param #
    =============================================================================
        Conv2D-1       [[1, 3, 160, 160]]    [1, 32, 80, 80]          896
      BatchNorm2D-1    [[1, 32, 80, 80]]     [1, 32, 80, 80]          128
         ReLU-1        [[1, 32, 80, 80]]     [1, 32, 80, 80]           0
         ReLU-2        [[1, 32, 80, 80]]     [1, 32, 80, 80]           0
    SeparableConv2D-1  [[1, 32, 80, 80]]     [1, 64, 80, 80]         2,400
      BatchNorm2D-2    [[1, 64, 80, 80]]     [1, 64, 80, 80]          256
         ReLU-3        [[1, 64, 80, 80]]     [1, 64, 80, 80]           0
    SeparableConv2D-2  [[1, 64, 80, 80]]     [1, 64, 80, 80]         4,736
      BatchNorm2D-3    [[1, 64, 80, 80]]     [1, 64, 80, 80]          256
       MaxPool2D-1     [[1, 64, 80, 80]]     [1, 64, 40, 40]           0
        Conv2D-2       [[1, 32, 80, 80]]     [1, 64, 40, 40]         2,112
        Encoder-1      [[1, 32, 80, 80]]     [1, 64, 40, 40]           0
         ReLU-4        [[1, 64, 40, 40]]     [1, 64, 40, 40]           0
    SeparableConv2D-3  [[1, 64, 40, 40]]     [1, 128, 40, 40]        8,896
      BatchNorm2D-4    [[1, 128, 40, 40]]    [1, 128, 40, 40]         512
         ReLU-5        [[1, 128, 40, 40]]    [1, 128, 40, 40]          0
    SeparableConv2D-4  [[1, 128, 40, 40]]    [1, 128, 40, 40]       17,664
      BatchNorm2D-5    [[1, 128, 40, 40]]    [1, 128, 40, 40]         512
       MaxPool2D-2     [[1, 128, 40, 40]]    [1, 128, 20, 20]          0
        Conv2D-3       [[1, 64, 40, 40]]     [1, 128, 20, 20]        8,320
        Encoder-2      [[1, 64, 40, 40]]     [1, 128, 20, 20]          0
         ReLU-6        [[1, 128, 20, 20]]    [1, 128, 20, 20]          0
    SeparableConv2D-5  [[1, 128, 20, 20]]    [1, 256, 20, 20]       34,176
      BatchNorm2D-6    [[1, 256, 20, 20]]    [1, 256, 20, 20]        1,024
         ReLU-7        [[1, 256, 20, 20]]    [1, 256, 20, 20]          0
    SeparableConv2D-6  [[1, 256, 20, 20]]    [1, 256, 20, 20]       68,096
      BatchNorm2D-7    [[1, 256, 20, 20]]    [1, 256, 20, 20]        1,024
       MaxPool2D-3     [[1, 256, 20, 20]]    [1, 256, 10, 10]          0
        Conv2D-4       [[1, 128, 20, 20]]    [1, 256, 10, 10]       33,024
        Encoder-3      [[1, 128, 20, 20]]    [1, 256, 10, 10]          0
         ReLU-8        [[1, 256, 10, 10]]    [1, 256, 10, 10]          0
    Conv2DTranspose-1  [[1, 256, 10, 10]]    [1, 256, 10, 10]       590,080
      BatchNorm2D-8    [[1, 256, 10, 10]]    [1, 256, 10, 10]        1,024
         ReLU-9        [[1, 256, 10, 10]]    [1, 256, 10, 10]          0
    Conv2DTranspose-2  [[1, 256, 10, 10]]    [1, 256, 10, 10]       590,080
      BatchNorm2D-9    [[1, 256, 10, 10]]    [1, 256, 10, 10]        1,024
       Upsample-1      [[1, 256, 10, 10]]    [1, 256, 20, 20]          0
       Upsample-2      [[1, 256, 10, 10]]    [1, 256, 20, 20]          0
        Conv2D-5       [[1, 256, 20, 20]]    [1, 256, 20, 20]       65,792
        Decoder-1      [[1, 256, 10, 10]]    [1, 256, 20, 20]          0
         ReLU-10       [[1, 256, 20, 20]]    [1, 256, 20, 20]          0
    Conv2DTranspose-3  [[1, 256, 20, 20]]    [1, 128, 20, 20]       295,040
     BatchNorm2D-10    [[1, 128, 20, 20]]    [1, 128, 20, 20]         512
         ReLU-11       [[1, 128, 20, 20]]    [1, 128, 20, 20]          0
    Conv2DTranspose-4  [[1, 128, 20, 20]]    [1, 128, 20, 20]       147,584
     BatchNorm2D-11    [[1, 128, 20, 20]]    [1, 128, 20, 20]         512
       Upsample-3      [[1, 128, 20, 20]]    [1, 128, 40, 40]          0
       Upsample-4      [[1, 256, 20, 20]]    [1, 256, 40, 40]          0
        Conv2D-6       [[1, 256, 40, 40]]    [1, 128, 40, 40]       32,896
        Decoder-2      [[1, 256, 20, 20]]    [1, 128, 40, 40]          0
         ReLU-12       [[1, 128, 40, 40]]    [1, 128, 40, 40]          0
    Conv2DTranspose-5  [[1, 128, 40, 40]]    [1, 64, 40, 40]        73,792
     BatchNorm2D-12    [[1, 64, 40, 40]]     [1, 64, 40, 40]          256
         ReLU-13       [[1, 64, 40, 40]]     [1, 64, 40, 40]           0
    Conv2DTranspose-6  [[1, 64, 40, 40]]     [1, 64, 40, 40]        36,928
     BatchNorm2D-13    [[1, 64, 40, 40]]     [1, 64, 40, 40]          256
       Upsample-5      [[1, 64, 40, 40]]     [1, 64, 80, 80]           0
       Upsample-6      [[1, 128, 40, 40]]    [1, 128, 80, 80]          0
        Conv2D-7       [[1, 128, 80, 80]]    [1, 64, 80, 80]         8,256
        Decoder-3      [[1, 128, 40, 40]]    [1, 64, 80, 80]           0
         ReLU-14       [[1, 64, 80, 80]]     [1, 64, 80, 80]           0
    Conv2DTranspose-7  [[1, 64, 80, 80]]     [1, 32, 80, 80]        18,464
     BatchNorm2D-14    [[1, 32, 80, 80]]     [1, 32, 80, 80]          128
         ReLU-15       [[1, 32, 80, 80]]     [1, 32, 80, 80]           0
    Conv2DTranspose-8  [[1, 32, 80, 80]]     [1, 32, 80, 80]         9,248
     BatchNorm2D-15    [[1, 32, 80, 80]]     [1, 32, 80, 80]          128
       Upsample-7      [[1, 32, 80, 80]]    [1, 32, 160, 160]          0
       Upsample-8      [[1, 64, 80, 80]]    [1, 64, 160, 160]          0
        Conv2D-8      [[1, 64, 160, 160]]   [1, 32, 160, 160]        2,080
        Decoder-4      [[1, 64, 80, 80]]    [1, 32, 160, 160]          0
        Conv2D-9      [[1, 32, 160, 160]]    [1, 4, 160, 160]        1,156
    =============================================================================
    Total params: 2,059,268
    Trainable params: 2,051,716
    Non-trainable params: 7,552
    -----------------------------------------------------------------------------
    Input size (MB): 0.29
    Forward/backward pass size (MB): 117.77
    Params size (MB): 7.86
    Estimated Total Size (MB): 125.92
    -----------------------------------------------------------------------------






    {'total_params': 2059268, 'trainable_params': 2051716}



## 五、模型训练

### 5.1 启动模型训练

使用模型代码进行Model实例生成，使用prepare接口定义优化器、损失函数和评价指标等信息，用于后续训练使用。在所有初步配置完成后，调用fit接口开启训练执行过程，调用fit时只需要将前面定义好的训练数据集、测试数据集、训练轮次（Epoch）和批次大小（batch_size）配置好即可。


```python
train_dataset = PetDataset(mode='train') # 训练数据集
val_dataset = PetDataset(mode='test') # 验证数据集

optim = paddle.optimizer.RMSProp(learning_rate=0.001,
                                 rho=0.9,
                                 momentum=0.0,
                                 epsilon=1e-07,
                                 centered=False,
                                 parameters=model.parameters())
model.prepare(optim, paddle.nn.CrossEntropyLoss(axis=1))
model.fit(train_dataset,
          val_dataset,
          epochs=15,
          batch_size=32,
          verbose=1)
```

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/15
    step 197/197 [==============================] - loss: 0.7987 - 259ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.6574 - 238ms/step
    Eval samples: 1108
    Epoch 2/15
    step 197/197 [==============================] - loss: 0.4407 - 253ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.5894 - 236ms/step
    Eval samples: 1108
    Epoch 3/15
    step 197/197 [==============================] - loss: 0.4761 - 254ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.6003 - 239ms/step
    Eval samples: 1108
    Epoch 4/15
    step 197/197 [==============================] - loss: 0.5745 - 255ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.5393 - 239ms/step
    Eval samples: 1108
    Epoch 5/15
    step 197/197 [==============================] - loss: 0.5740 - 255ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4729 - 235ms/step
    Eval samples: 1108
    Epoch 6/15
    step 197/197 [==============================] - loss: 0.4172 - 255ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4371 - 236ms/step
    Eval samples: 1108
    Epoch 7/15
    step 197/197 [==============================] - loss: 0.2865 - 256ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4426 - 237ms/step
    Eval samples: 1108
    Epoch 8/15
    step 197/197 [==============================] - loss: 0.2916 - 255ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4000 - 238ms/step
    Eval samples: 1108
    Epoch 9/15
    step 197/197 [==============================] - loss: 0.4454 - 254ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4701 - 239ms/step
    Eval samples: 1108
    Epoch 10/15
    step 197/197 [==============================] - loss: 0.3905 - 256ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4632 - 237ms/step
    Eval samples: 1108
    Epoch 11/15
    step 197/197 [==============================] - loss: 0.3227 - 254ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4522 - 236ms/step
    Eval samples: 1108
    Epoch 12/15
    step 197/197 [==============================] - loss: 0.3093 - 255ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4541 - 235ms/step
    Eval samples: 1108
    Epoch 13/15
    step 197/197 [==============================] - loss: 0.3452 - 256ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4722 - 236ms/step
    Eval samples: 1108
    Epoch 14/15
    step 197/197 [==============================] - loss: 0.3925 - 254ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.4808 - 234ms/step
    Eval samples: 1108
    Epoch 15/15
    step 197/197 [==============================] - loss: 0.3241 - 256ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 35/35 [==============================] - loss: 0.5109 - 236ms/step
    Eval samples: 1108


## 六、模型预测

### 6.1 预测数据集准备和预测

继续使用PetDataset来实例化待预测使用的数据集。这里为了方便没有在另外准备预测数据，复用了评估数据。

可以直接使用model.predict接口来对数据集进行预测操作，只需要将预测数据集传递到接口内即可。


```python
predict_dataset = PetDataset(mode='predict')
predict_results = model.predict(predict_dataset)
```

    Predict begin...
    step  746/1108 [===================>..........] - ETA: 5s - 15ms/ste

### 6.2 预测结果可视化

从预测数据集中抽3个动物来看看预测的效果，展示一下原图、标签图和预测结果。


```python
plt.figure(figsize=(10, 10))

i = 0
mask_idx = 0

with open('./predict.txt', 'r') as f:
    for line in f.readlines():
        image_path, label_path = line.strip().split('\t')
        resize_t = T.Compose([
            T.Resize(IMAGE_SIZE)
        ])
        image = resize_t(PilImage.open(image_path))
        label = resize_t(PilImage.open(label_path))

        image = np.array(image).astype('uint8')
        label = np.array(label).astype('uint8')

        if i > 8:
            break
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis("off")

        plt.subplot(3, 3, i + 2)
        plt.imshow(label, cmap='gray')
        plt.title('Label')
        plt.axis("off")

        # 模型只有一个输出，所以通过predict_results[0]来取出1000个预测的结果
        # 映射原始图片的index来取出预测结果，提取mask进行展示
        data = predict_results[0][mask_idx][0].transpose((1, 2, 0))
        mask = np.argmax(data, axis=-1)

        plt.subplot(3, 3, i + 3)
        plt.imshow(mask.astype('uint8'), cmap='gray')
        plt.title('Predict')
        plt.axis("off")
        i += 3
        mask_idx += 1

plt.show()
```


![png](output_31_0.png)
