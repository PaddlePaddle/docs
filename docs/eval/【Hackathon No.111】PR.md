# 飞桨动静转换评估-评估报告

| 领域 |飞桨动静转换评估 |
| --- | --- |
|提交作者 | 王源 袁闯闯 | 
|提交时间 | 2022-05-04 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | V2.2 | 
|文件名 | 【Hackathon No.111】PR.md | 

一个完整的使用动静转换@to_static导出、可部署的模型完整代码（参考以图搜图）

以下为 AI Studio 任务链接  (cifar10动态图转为静态图（完整demo）)

AI Studio 任务链接:https://aistudio.baidu.com/aistudio/projectdetail/3937674?channel=0&channelType=0&shared=1

# 1、任务描述：

	飞桨框架于 2.0 正式版之后正式发布了动静转换@to_static功能，并在2.1、2.2 两个大版本中不断新增了各项功能，以及详细的使用文档和最佳实践教程（以图搜图）。
	
	在本任务中，我们希望你全面体验飞桨的动静转换@to_static功能，即参考飞桨官网 -> 使用指南 -> 动态图转静态 下的内容，体验动转静模型导出、动转静训练等功能，
	
	产出一份整体功能体验报告。

# 2、环境配置：

	因为需要体验飞桨paddlepaddle框架的动转静模型导出、动转静训练等功能，所以首先需要安装飞桨paddlepaddle框架，运行环境使用pycharm和anaconda。
	
	所以在进行PaddlePaddle安装之前应确保Anaconda软件环境已经正确安装。软件下载和安装参见Anaconda官网(https://www.anaconda.com/)。
	
	在已经正确安装Anaconda的情况下请按照下列步骤安装PaddlePaddle。

	-Windows 7/8/10 专业版/企业版 (64bit)

	-GPU版本支持CUDA 10.1/10.2/11.2，且仅支持单卡

	-conda 版本 4.8.3+ (64 bit)

## 2.1、创建虚拟环境：

- 1、安装环境
	首先根据具体的Python版本创建Anaconda虚拟环境，PaddlePaddle的Anaconda安装支持以下四种Python安装环境。
如果您想使用的python版本为3.6:
```python
conda create -n paddle_env python=3.6
```
如果您想使用的python版本为3.7:
```python
conda create -n paddle_env python=3.7
```	
如果您想使用的python版本为3.8:
```python
conda create -n paddle_env python=3.8
```
如果您想使用的python版本为3.9:
```python
conda create -n paddle_env python=3.9
```	
本实验使用python版本为3.9,虚拟环境命名为paddlepaddle-gpu，即：
```python
conda create -n paddlepaddle-gpu python=3.9
```	
- 2、进入Anaconda虚拟环境
```python
activate paddlepaddle-gpu
```
- 3、开始安装
GPU版的PaddlePaddle
本实验为 CUDA 10.2，需要搭配cuDNN 7 (cuDNN>=7.6.5, 多卡环境下 NCCL>=2.7)
添加清华源（可选），对于国内用户无法连接到Anaconda官方源的可以按照以下命令添加清华源:
```python
conda install paddlepaddle-gpu==2.2.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```
您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法。
- 4、验证安装
安装完成后您可以使用 python 或 python3 进入python解释器，输入import paddle ，再输入 paddle.utils.run_check()

如果出现PaddlePaddle is installed successfully!，说明您已成功安装。

## 2.2、paddlepaddle环境配置：
```python
import paddle
import paddle.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

print(paddle.__version__)
```
```python
2.2.2
```

# 3、数据加载：

## 3.1、数据集介绍

本示例采用 CIFAR-10 数据集。

CIFAR-10 和 CIFAR-100 是8000 万个微小图像数据集的标记子集。它们由 Alex Krizhevsky、Vinod Nair 和 Geoffrey Hinton 收集。

数据集分为五个训练批次和一个测试批次，每个批次有 10000 张图像。测试批次恰好包含来自每个类别的 1000 个随机选择的图像。训练批次包含随机顺序的剩余图像，

但一些训练批次可能包含来自一个类的图像多于另一个。在它们之间，训练批次恰好包含来自每个类别的 5000 张图像。

以下是数据集中的类：


                                              飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车


这些类是完全互斥的。汽车和卡车之间没有重叠。“汽车”包括轿车、SUV 之类的东西。“卡车”只包括大卡车。两者都不包括皮卡车。

数据集下载网址为：https://www.cs.toronto.edu/~kriz/cifar.html
```python
import paddle.vision.transforms as T

transform = T.Compose([T.Transpose((2, 0, 1))])

cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
x_train = np.zeros((50000, 3, 32, 32))
y_train = np.zeros((50000, 1), dtype='int32')

for i in range(len(cifar10_train)):
    train_image, train_label = cifar10_train[i]
    
    # normalize the data
    x_train[i,:, :, :] = train_image / 255.
    y_train[i, 0] = train_label

y_train = np.squeeze(y_train)

cifar10_test = paddle.vision.datasets.cifar.Cifar10(mode='test', transform=transform)
x_test = np.zeros((10000, 3, 32, 32), dtype='float32')
y_test = np.zeros((10000, 1), dtype='int64')

for i in range(len(cifar10_test)):
    test_image, test_label = cifar10_test[i]
   
    # normalize the data
    x_test[i,:, :, :] = test_image / 255.
    y_test[i, 0] = test_label

y_test = np.squeeze(y_test)

height_width = 32

def show_collage(examples):
    box_size = height_width + 2
    num_rows, num_cols = examples.shape[:2]

    collage = Image.new(
        mode="RGB",
        size=(num_cols * box_size, num_rows * box_size),
        color=(255, 255, 255),
    )
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
            array = array.transpose(1,2,0)
            collage.paste(
                Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
            )

    collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
    return collage

sample_idxs = np.random.randint(0, 50000, size=(5, 5))
examples = x_train[sample_idxs]
show_collage(examples)
```
## 3.2、构建训练数据

图片检索的模型的训练样本跟常见的分类任务的训练样本不太一样的地方在于，每个训练样本并不是一个(image, class)这样的形式。而是（image0, image1, similary_or_not)的形式，即，每

一个训练样本由两张图片组成，而其label是这两张图片是否相似的标志位（0或者1）。

很自然的能够想到，来自同一个类别的两张图片，是相似的图片，而来自不同类别的两张图片，应该是不相似的图片。

为了能够方便的抽样出相似图片（以及不相似图片）的样本，先建立能够根据类别找到该类别下所有图片的索引。
```python
class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)

num_classes = 10

def reader_creator(num_batchs):
    def reader():
        iter_step = 0
        while True:
            if iter_step >= num_batchs:
                break
            iter_step += 1
            x = np.empty((2, num_classes, 3, height_width, height_width), dtype=np.float32)
            for class_idx in range(num_classes):
                examples_for_class = class_idx_to_train_idxs[class_idx]
                anchor_idx = random.choice(examples_for_class)
                positive_idx = random.choice(examples_for_class)
                while positive_idx == anchor_idx:
                    positive_idx = random.choice(examples_for_class)
                x[0, class_idx] = x_train[anchor_idx]
                x[1, class_idx] = x_train[positive_idx]
            yield x

    return reader

def anchor_positive_pairs(num_batchs=100):
    return reader_creator(num_batchs)

pairs_train_reader = anchor_positive_pairs(num_batchs=1000)
```
# 4、模型组网：
把图片转换为高维的向量表示的网络
目标是首先把图片转换为高维空间的表示，然后计算图片在高维空间表示时的相似度。 下面的网络结构用来把一个形状为(3, 32, 32)的图片转换成形状为(8,)的向量。在有些资料中也会把这个转换成的向量称为Embedding，请注意，这与自然语言处理领域的词向量的区别。 下面的模型由三个连续的卷积加一个全局均值池化，然后用一个线性全链接层映射到维数为8的向量空间。为了后续计算余弦相似度时的便利，还在最后做了归一化。（即，余弦相似度的分母部分）
```python
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, 
                                      out_channels=32, 
                                      kernel_size=(3, 3),
                                      stride=2)
         
        self.conv2 = paddle.nn.Conv2D(in_channels=32, 
                                      out_channels=64, 
                                      kernel_size=(3,3), 
                                      stride=2)       
        
        self.conv3 = paddle.nn.Conv2D(in_channels=64, 
                                      out_channels=128, 
                                      kernel_size=(3,3),
                                      stride=2)
       
        self.gloabl_pool = paddle.nn.AdaptiveAvgPool2D((1,1))

        self.fc1 = paddle.nn.Linear(in_features=128, out_features=8)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.gloabl_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self.fc1(x)
        x = x / paddle.norm(x, axis=1, keepdim=True)
        return x
```
# 5、模型训练：

将epoch设置为10，进行模型训练

```python
def train(model):
    print('start training ... ')
    model.train()

    inverse_temperature = paddle.to_tensor(np.array([1.0/0.2], dtype='float32'))

    epoch_num = 10
    
    opt = paddle.optimizer.Adam(learning_rate=0.0001,
                                parameters=model.parameters())
    
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(pairs_train_reader()):
            anchors_data, positives_data = data[0], data[1]

            anchors = paddle.to_tensor(anchors_data)
            positives = paddle.to_tensor(positives_data)
            
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            
            similarities = paddle.matmul(anchor_embeddings, positive_embeddings, transpose_y=True) 
            similarities = paddle.multiply(similarities, inverse_temperature)
            
            sparse_labels = paddle.arange(0, num_classes, dtype='int64')

            loss = F.cross_entropy(similarities, sparse_labels)
            
            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()

model = MyNet()
train(model)
```

```python
start training ... 
epoch: 0, batch_id: 0, loss is: [2.2915785]
epoch: 0, batch_id: 500, loss is: [2.4421422]
epoch: 0, batch_id: 1000, loss is: [1.7860969]
epoch: 0, batch_id: 1500, loss is: [2.3819976]
epoch: 1, batch_id: 0, loss is: [2.2505655]
epoch: 1, batch_id: 500, loss is: [1.6781666]
epoch: 1, batch_id: 1000, loss is: [1.8037045]
epoch: 1, batch_id: 1500, loss is: [1.8967863]
epoch: 2, batch_id: 0, loss is: [1.775666]
epoch: 2, batch_id: 500, loss is: [2.0136874]
epoch: 2, batch_id: 1000, loss is: [2.103913]
epoch: 2, batch_id: 1500, loss is: [1.7592652]
epoch: 3, batch_id: 0, loss is: [1.7325624]
epoch: 3, batch_id: 500, loss is: [1.9885247]
epoch: 3, batch_id: 1000, loss is: [2.3454063]
epoch: 3, batch_id: 1500, loss is: [1.9360502]
epoch: 4, batch_id: 0, loss is: [2.1657584]
epoch: 4, batch_id: 500, loss is: [2.0958445]
epoch: 4, batch_id: 1000, loss is: [1.9509046]
epoch: 4, batch_id: 1500, loss is: [1.8858738]
epoch: 5, batch_id: 0, loss is: [1.9648739]
epoch: 5, batch_id: 500, loss is: [1.8831095]
epoch: 5, batch_id: 1000, loss is: [1.9274123]
epoch: 5, batch_id: 1500, loss is: [2.2648232]
epoch: 6, batch_id: 0, loss is: [2.131785]
epoch: 6, batch_id: 500, loss is: [1.7363421]
epoch: 6, batch_id: 1000, loss is: [2.2151723]
epoch: 6, batch_id: 1500, loss is: [1.5245721]
epoch: 7, batch_id: 0, loss is: [1.7423642]
epoch: 7, batch_id: 500, loss is: [1.5562365]
epoch: 7, batch_id: 1000, loss is: [1.6524445]
epoch: 7, batch_id: 1500, loss is: [1.9120047]
epoch: 8, batch_id: 0, loss is: [1.8247225]
epoch: 8, batch_id: 500, loss is: [1.5704175]
epoch: 8, batch_id: 1000, loss is: [1.9273182]
epoch: 8, batch_id: 1500, loss is: [1.7724463]
epoch: 9, batch_id: 0, loss is: [1.5964721]
epoch: 9, batch_id: 500, loss is: [1.5145239]
epoch: 9, batch_id: 1000, loss is: [1.8208185]
epoch: 9, batch_id: 1500, loss is: [2.3465972]

```
# 6、模型预测：
前述的模型训练训练结束之后，就可以用该网络结构来计算出任意一张图片的高维向量表示（embedding)，通过计算该图片与图片库中其他图片的高维向量表示之间的相似度，
就可以按照相似程度进行排序，排序越靠前，则相似程度越高。

下面对测试集中所有的图片都两两计算相似度，然后选一部分相似的图片展示出来。

```python
near_neighbours_per_example = 10

x_test_t = paddle.to_tensor(x_test)
test_images_embeddings = model(x_test_t)
similarities_matrix = paddle.matmul(test_images_embeddings, test_images_embeddings, transpose_y=True) 

indicies = paddle.argsort(similarities_matrix, descending=True)
indicies = indicies.numpy()

examples = np.empty(
    (
        num_classes,
        near_neighbours_per_example + 1,
        3,
        height_width,
        height_width,
    ),
    dtype=np.float32,
)

for row_idx in range(num_classes):
    examples_for_class = class_idx_to_test_idxs[row_idx]
    anchor_idx = random.choice(examples_for_class)
    
    examples[row_idx, 0] = x_test[anchor_idx]
    anchor_near_neighbours = indicies[anchor_idx][1:near_neighbours_per_example+1]
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = x_test[nn_idx]

show_collage(examples)
```
# 7、使用 @to_static 进行动静转换：
动静转换（@to_static）通过解析 Python 代码（抽象语法树，下简称：AST） 实现一行代码即可将动态图转为静态图的功能，只需在待转化的函数前添加一个装饰器 @paddle.jit.to_static 

使用 @to_static 即支持 可训练可部署 ，也支持只部署（详见模型导出） ，常见使用方式如下：

方式一：使用 @to_static 装饰器装饰 SimpleNet (继承了 nn.Layer) 的 forward 函数:
```python
import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static # 动静转换
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()
net.eval()
x = paddle.rand([2, 10])
y = paddle.rand([2, 3])
out = net(x, y)                # 动转静训练
paddle.jit.save(net, './net')  # 导出预测模型
```
方式二：调用 paddle.jit.to_static() 函数，仅做预测模型导出时推荐此种用法。
```python
import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()
net.eval()
net = paddle.jit.to_static(net)  # 动静转换
x = paddle.rand([2, 10])
y = paddle.rand([2, 3])
out = net(x, y)                  # 动转静训练
paddle.jit.save(net, './net')    # 导出预测模型

```
方式一和方式二的主要区别是，前者直接在 forward() 函数定义处装饰，后者显式调用了 jit.to_static()方法，默认会对 net.forward进行动静转换。

本实验使用 paddle.jit.to_static 实现动转静：
飞桨推荐使用 @paddle.jit.to_static 实现动转静，也被称为基于源代码转写的动态图转静态图，其基本原理是通过分析 Python 代码来将动态图代码转写为静态图代码，并在底层自动使用执行器运行，使用起来非常方便，只需要在原网络结构的 forward 前添加一个装饰器 paddle.jit.to_static 即可。

## 7.1、改写组网代码
```python
class MyNet2(paddle.nn.Layer):
    def __init__(self):
        super(MyNet2, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, 
                                      out_channels=32, 
                                      kernel_size=(3, 3),
                                      stride=2)
         
        self.conv2 = paddle.nn.Conv2D(in_channels=32, 
                                      out_channels=64, 
                                      kernel_size=(3,3), 
                                      stride=2)       
        
        self.conv3 = paddle.nn.Conv2D(in_channels=64, 
                                      out_channels=128, 
                                      kernel_size=(3,3),
                                      stride=2)
       
        self.gloabl_pool = paddle.nn.AdaptiveAvgPool2D((1,1))

        self.fc1 = paddle.nn.Linear(in_features=128, out_features=8)
    
    # 在forward 前添加 paddle.jit.to_static 装饰器
    @paddle.jit.to_static()
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.gloabl_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self.fc1(x)
        x = x / paddle.norm(x, axis=1, keepdim=True)
        return x
```
通过 model.summary 查看网络结构。
```python
model_2 = MyNet2()
model_info = paddle.summary(model_2, (10, 3, 32, 32))
print(model_info)
```

```python
-------------------------------------------------------------------------------
   Layer (type)         Input Shape          Output Shape         Param #    
===============================================================================
     Conv2D-4        [[10, 3, 32, 32]]     [10, 32, 15, 15]         896      
     Conv2D-5        [[10, 32, 15, 15]]     [10, 64, 7, 7]        18,496     
     Conv2D-6         [[10, 64, 7, 7]]     [10, 128, 3, 3]        73,856     
AdaptiveAvgPool2D-2  [[10, 128, 3, 3]]     [10, 128, 1, 1]           0       
     Linear-2           [[10, 128]]            [10, 8]             1,032     
===============================================================================
Total params: 94,280
Trainable params: 94,280
Non-trainable params: 0
-------------------------------------------------------------------------------
Input size (MB): 0.12
Forward/backward pass size (MB): 0.89
Params size (MB): 0.36
Estimated Total Size (MB): 1.36
-------------------------------------------------------------------------------


{'total_params': 94280, 'trainable_params': 94280}

```

## 7.2、模型训练
使用 paddle.jit.to_static 装饰器后，训练方式仍与原动态图训练一致。因此这里直接传入 model_2 完成模型的训练。
```python
train(model_2)
```

```python
start training ... 
epoch: 0, batch_id: 0, loss is: [2.2707999]
epoch: 0, batch_id: 500, loss is: [2.2578232]
epoch: 0, batch_id: 1000, loss is: [2.0026908]
epoch: 0, batch_id: 1500, loss is: [2.1243637]
epoch: 1, batch_id: 0, loss is: [2.5174978]
epoch: 1, batch_id: 500, loss is: [1.9235588]
epoch: 1, batch_id: 1000, loss is: [2.241912]
epoch: 1, batch_id: 1500, loss is: [2.2027836]
epoch: 2, batch_id: 0, loss is: [2.0674071]
epoch: 2, batch_id: 500, loss is: [1.8517029]
epoch: 2, batch_id: 1000, loss is: [1.9565346]
epoch: 2, batch_id: 1500, loss is: [2.2468033]
epoch: 3, batch_id: 0, loss is: [1.5385025]
epoch: 3, batch_id: 500, loss is: [2.1791337]
epoch: 3, batch_id: 1000, loss is: [2.0335]
epoch: 3, batch_id: 1500, loss is: [1.8313652]
epoch: 4, batch_id: 0, loss is: [1.8956888]
epoch: 4, batch_id: 500, loss is: [1.6906776]
epoch: 4, batch_id: 1000, loss is: [2.0118344]
epoch: 4, batch_id: 1500, loss is: [2.002913]
epoch: 5, batch_id: 0, loss is: [1.8000762]
epoch: 5, batch_id: 500, loss is: [1.7253144]
epoch: 5, batch_id: 1000, loss is: [1.5976737]
epoch: 5, batch_id: 1500, loss is: [1.5003413]
epoch: 6, batch_id: 0, loss is: [1.8904054]
epoch: 6, batch_id: 500, loss is: [2.1880364]
epoch: 6, batch_id: 1000, loss is: [2.0464098]
epoch: 6, batch_id: 1500, loss is: [1.7705017]
epoch: 7, batch_id: 0, loss is: [1.8255459]
epoch: 7, batch_id: 500, loss is: [1.8008741]
epoch: 7, batch_id: 1000, loss is: [1.9753224]
epoch: 7, batch_id: 1500, loss is: [2.376344]
epoch: 8, batch_id: 0, loss is: [1.7606968]
epoch: 8, batch_id: 500, loss is: [1.9435362]
epoch: 8, batch_id: 1000, loss is: [2.3564293]
epoch: 8, batch_id: 1500, loss is: [1.9458401]
epoch: 9, batch_id: 0, loss is: [1.7150522]
epoch: 9, batch_id: 500, loss is: [1.9844353]
epoch: 9, batch_id: 1000, loss is: [1.962418]
epoch: 9, batch_id: 1500, loss is: [1.7263882]

```
## 7.3、动转静模型导出
动转静模块是架在动态图与静态图的一个桥梁，旨在打破动态图模型训练与静态部署的鸿沟，消除部署时对模型代码的依赖，打通与预测端的交互逻辑。
在处理逻辑上，动转静主要包含两个主要模块：

代码层面：将模型中所有的 layers 接口在静态图模式下执行以转为 Op ，从而生成完整的静态 Program

Tensor层面：将所有的 Parameters 和 Buffers 转为可导出的 Variable 参数（ persistable=True ）
通过 forward 导出预测模型
通过 forward 导出预测模型导出一般包括三个步骤：

切换 eval() 模式：类似 Dropout 、LayerNorm 等接口在 train() 和 eval() 的行为存在较大的差异，在模型导出前，请务必确认模型已切换到正确的模式，否则导出的模型在预测阶段可能出现输出结果不符合预期的情况。

构造 InputSpec 信息：InputSpec 用于表示输入的shape、dtype、name信息，且支持用 None 表示动态shape（如输入的 batch_size 维度），是辅助动静转换的必要描述信息。

调用 save 接口：调用 paddle.jit.save接口，若传入的参数是类实例，则默认对 forward 函数进行 @to_static 装饰，并导出其对应的模型文件和参数文件。

如下是一个简单的示例：

```python
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

    def another_func(self, x):
        out = self.linear(x)
        out = out * 2
        return out

net = SimpleNet()
# train(net)  模型训练 (略)

# step 1: 切换到 eval() 模式
net.eval()

# step 2: 定义 InputSpec 信息
x_spec = InputSpec(shape=[None, 3], dtype='float32', name='x')
y_spec = InputSpec(shape=[3], dtype='float32', name='y')

# step 3: 调用 jit.save 接口
net = paddle.jit.save(net, path='simple_net', input_spec=[x_spec, y_spec])  # 动静转换

```
本实验使用 paddle.jit.save 保存动转静模型
使用 paddle.jit.to_static 转换模型后，需要调用 paddle.jit.save 将保存模型，以供后续的预测部署。保存后，会产生 model.pdmodel 、model.pdiparams.info、model.pdiparams 三个文件。
```python
paddle.jit.save(model_2, 'model')
```
使用 InputSpec 指定模型输入 Tensor 信息
动静转换在生成静态图 Program 时，依赖输入 Tensor 的 shape、dtype 和 name 信息。因此，Paddle 提供了 InputSpec 接口，用于指定输入 Tensor 的描述信息，并支持动态 shape 特性。

构造 InputSpec
方式一：直接构造

InputSpec 接口在 paddle.static 目录下， 只有 shape 是必须参数， dtype 和 name 可以缺省，默认取值分别为 float32 和 None 。使用样例如下：
```python
from paddle.static import InputSpec

x = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')

print(x)      # InputSpec(shape=(-1, 784), dtype=paddle.float32, name=x)
print(label)  # InputSpec(shape=(-1, 1), dtype=paddle.int64, name=label)

```
方式二：由 Tensor 构造

可以借助 InputSpec.from_tensor 方法，从一个 Tensor 直接创建 InputSpec 对象，其拥有与源 Tensor 相同的 shape 和 dtype 。 使用样例如下：
```python
import numpy as np
import paddle
from paddle.static import InputSpec

x = paddle.to_tensor(np.ones([2, 2], np.float32))
x_spec = InputSpec.from_tensor(x, name='x')
print(x_spec)  # InputSpec(shape=(2, 2), dtype=paddle.float32, name=x)

```
注：若未在 from_tensor 中指定新的 name，则默认使用与源 Tensor 相同的 name。

方式三：由 numpy.ndarray

也可以借助 InputSpec.from_numpy 方法，从一个 Numpy.ndarray 直接创建 InputSpec 对象，其拥有与源 ndarray 相同的 shape 和 dtype 。使用样例如下：
```python
import numpy as np
from paddle.static import InputSpec

x = np.ones([2, 2], np.float32)
x_spec = InputSpec.from_numpy(x, name='x')
print(x_spec)  # InputSpec(shape=(2, 2), dtype=paddle.float32, name=x)

```
注：若未在 from_numpy 中指定新的 name，则默认使用 None 。
基本用法
方式一： 在 @to_static 装饰器中调用

如下是一个简单的使用样例：

```python
import paddle
from paddle.nn import Layer
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()

# save static graph model for inference directly
paddle.jit.save(net, './simple_net')

```
在上述的样例中， @to_static 装饰器中的 input_spec 为一个 InputSpec 对象组成的列表，用于依次指定参数 x 和 y 对应的 Tensor 签名信息。在实例化 SimpleNet 后，可以直接调用 paddle.jit.save 保存静态图模型，不需要执行任何其他的代码。

注：

input_spec 参数中不仅支持 InputSpec 对象，也支持 int 、 float 等常见 Python 原生类型。

若指定 input_spec 参数，则需为被装饰函数的所有必选参数都添加对应的 InputSpec 对象，如上述样例中，不支持仅指定 x 的签名信息。

若被装饰函数中包括非 Tensor 参数，推荐函数的非 Tensor 参数设置默认值，如 forward(self, x, use_bn=False)

方式二：在 to_static 函数中调用

若期望在动态图下训练模型，在训练完成后保存预测模型，并指定预测时需要的签名信息，则可以选择在保存模型时，直接调用 to_static 函数。使用样例如下：
```python
class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()

# train process (Pseudo code)
for epoch_id in range(10):
    train_step(net, train_reader)

net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])

# save static graph model for inference directly
paddle.jit.save(net, './simple_net')

```
如上述样例代码中，在完成训练后，可以借助 to_static(net, input_spec=...) 形式对模型实例进行处理。Paddle 会根据 input_spec 信息对 forward 函数进行递归的动转静，得到完整的静态图，且包括当前训练好的参数数据。

方式三：通过 list 和 dict 推导

上述两个样例中，被装饰的 forward 函数的参数均为 Tensor 。这种情况下，参数个数必须与 InputSpec 个数相同。但当被装饰的函数参数为 list 或 dict 类型时，input_spec 需要与函数参数保持相同的嵌套结构。

当函数的参数为 list 类型时，input_spec 列表中对应元素的位置，也必须是包含相同元素的 InputSpec 列表。使用样例如下：
```python
class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static(input_spec=[[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')]])
    def forward(self, inputs):
        x, y = inputs[0], inputs[1]
        out = self.linear(x)
        out = out + y
        return out

```
其中 input_spec 参数是长度为 2 的 list ，对应 forward 函数的 x 和 bias_info 两个参数。 input_spec 的最后一个元素是包含键名为 x 的 InputSpec 对象的 dict ，对应参数 bias_info 的 Tensor 签名信息。

方式四：指定非Tensor参数类型

目前，to_static 装饰器中的 input_spec 参数仅接收 InputSpec 类型对象。若被装饰函数的参数列表除了 Tensor 类型，还包含其他如 Int、 String 等非 Tensor 类型时，推荐在函数中使用 kwargs 形式定义非 Tensor 参数，如下述样例中的 use_act 参数。


```python

class SimpleNet(Layer):
    def __init__(self, ):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)
        self.relu = paddle.nn.ReLU()

    def forward(self, x, use_act=False):
        out = self.linear(x)
        if use_act:
            out = self.relu(out)
        return out

net = SimpleNet()
# 方式一：save inference model with use_act=False
net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x')])
paddle.jit.save(net, path='./simple_net')


# 方式二：save inference model with use_act=True
net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x'), True])
paddle.jit.save(net, path='./simple_net')

```
在上述样例中，假设 step 为奇数时，use_act 取值为 False ； step 为偶数时， use_act 取值为 True 。动转静支持非 Tensor 参数在训练时取不同的值，且保证了取值不同的训练过程都可以更新模型的网络参数，行为与动态图一致。

在借助 paddle.jit.save 保存预测模型时，动转静会根据 input_spec 和 kwargs 的默认值保存推理模型和网络参数。建议将 kwargs 参数默认值设置为预测时的取值。

## 7.4、使用 paddle.jit.load 加载动转静模型
将模型导出后，需要使用 paddle.jit.load 加载模型。加载后的模型可以直接用于预测。
```python
model_2 = paddle.jit.load('model')
```
## 7.5、使用动转静模型
前述的模型训练训练结束之后，就可以用该网络结构来计算出任意一张图片的高维向量表示（embedding)，通过计算该图片与图片库中其他图片的高维向量表示之间的相似度，就可以按照相似程度进行排序，排序越靠前，则相似程度越高。

下面对测试集中所有的图片都两两计算相似度，然后选一部分相似的图片展示出来。
```python
near_neighbours_per_example = 10

x_test_t = paddle.to_tensor(x_test)
test_images_embeddings = model_2(x_test_t)
similarities_matrix = paddle.matmul(test_images_embeddings, test_images_embeddings, transpose_y=True) 

indicies = paddle.argsort(similarities_matrix, descending=True)
indicies = indicies.numpy()

examples = np.empty(
    (
        num_classes,
        near_neighbours_per_example + 1,
        3,
        height_width,
        height_width,
    ),
    dtype=np.float32,
)

for row_idx in range(num_classes):
    examples_for_class = class_idx_to_test_idxs[row_idx]
    anchor_idx = random.choice(examples_for_class)
    
    examples[row_idx, 0] = x_test[anchor_idx]
    anchor_near_neighbours = indicies[anchor_idx][1:near_neighbours_per_example+1]
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = x_test[nn_idx]

show_collage(examples)
```
通过上述的内容，就使用 @jit.to_static 完成了动转静并使用该模型进行了预测。
# 8、 总结：
上述动态图转静态图的过程中，总体来说感觉还是可以的，具体总结了以下几个层面：

- 1、接口层面：
	接口功能目前使用中覆盖了所有使用场景；InputSpec 指定信息也好用
- 2、语法层面：
	语法支持方面尚未发现问题，我认为语法支持比较完备，但是图片展示功能（show_collage(examples)）在运行时展示不出来；控制流语法转换比较流畅
- 3、报错层面：
	报错信息可读性差比较好，比如：
	Traceback (most recent call last):
	  File "D:\Postgraduate\deep_learning\jiaoliu\pp\train.py", line 191, in <module>
	    indicies = paddle.argsort(similarities_matrix, descending=True)
	  File "F:\Users\ASUS\anaconda3\envs\paddlepaddle-gpu\lib\site-packages\paddle\tensor\search.py", line 92, in argsort
	    _, ids = _C_ops.argsort(x, 'axis', axis, 'descending', descending)
	SystemError: (Fatal) Operator argsort raises an struct paddle::memory::allocation::BadAlloc exception.
	The exception content is
	:ResourceExhaustedError: 

	Out of memory error on GPU 0. Cannot allocate 762.939697MB memory on GPU 0, 3.256321GB memory has been allocated and available memory is only 761.527736MB.

	Please check whether there is any other process using GPU 0.
	1. If yes, please stop them, or start PaddlePaddle on another GPU.
	2. If no, please decrease the batch size of your model. 
	但如果没有详细的提示信息还需上网查找解决方案；调试工具比较易用
- 4、文档层面：
	从官方文档来说，动态图转静态图的示例文档感觉不太完善，例如paddle.jit.load等的API没有在使用指南的示例文档中展现，教程文档还有待完善，其他方面感觉内容比较详细丰富，具有较好的指导性
- 5、意见建议：
	动态图转静态图的代码还是比较方便的，但对新手来说有一定难度，建议在使用指南中可以适当增加一些重点难点视频解说；
	有一些官网上的使用指南用于学习，但是有一些是在其使用指南中没有介绍的，例如数据集的拆分等这些需要自己去琢磨，可再丰富一下指南内容；
	使用指南中的方式或方法比较多的可以用序号组合一下，方便学习者更快的了解每种方式或方法的优缺点;
	paddle自带数据集太大，数据处理需消耗大量资源，使用起来不太方便，建议新增一些小的数据集。

