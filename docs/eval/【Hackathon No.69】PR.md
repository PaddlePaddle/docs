# 飞桨动态图单机训练功能评估报告

| 领域         | 飞桨动态图单机训练功能评估报告 |
| ------------ | ------------------------------ |
| 提交作者     | 王源 袁闯闯                    |
| 提交时间     | 2022-05-03                     |
| 版本号       | V1.0                           |
| 依赖飞桨版本 | paddlepaddle-gpu==2.2          |
| 文件名       | 【Hackathon No.69】 PR.md      |


# 一、摘要

相关背景：飞桨框架于 2.0 正式版全面支持了动态图训练，并在2.1、2.2 两个大版本中不断新增了API以及大幅增强了训练功能。希望有人对于飞桨框架动态图下单机训练功能整体的使用感受，可以与其他深度学习框架做功能对比，包括API、Tensor 索引、NumPy Compatibility、报错信息提示、训练性能、以及各种 trick 的用法等，并产出一份对应的评估报告。

本评估方案将从以下几个方面对paddle动态图单机训练功能进行体验评估：

1、环境配置及开启动态图模式

2、API使用及对比

调用高层API:如：paddle.Model、paddle.vision，与pytorch框架做对比。并在LeNet、ResNet等网络模型或模型自己组网（Sequential组网、SubClass组网）训练中进行评估。

3、Tensor 索引

在模型训练中体验了Tensor在数据传递过程中的表现（如：了解索引和 其切片规则、访问与修改Tensor、逻辑相关函数重写规则），并体验了使用指南里有关Tensor的所有基本操作。

4、NumPy兼容性分析及对比

在动态图模型代码中，所有与组网相关的 numpy 操作都必须用 paddle 的 API 重新实现，所以在模型训练过程中体验Paddle.API来感受对比Pytorch的表现；分析了Tensor兼容Numpy数组的同时，优先使用Tensor的两种场景。

5、动态图单机训练

体验控制流和共享权重的使用效果，然后在数据集定义、加载和数据预处理、数据增强方面感受与Pytorch使用的区别，最后通过LeNet举例说明训练结果，并进行了对比分析

6、各种 trick 的用法体验

7、报错汇总

# 二、环境配置及开启动态图模式

本次训练评估在个人电脑上进行：

|   名称   |                     参数                     |
| :------: | :------------------------------------------: |
|   CPU    |    Intel(R)Core(TM)i5-7200U CPU @2.50GHz     |
|   内存   |                  12GB DDR4                   |
|   GPU    |             NVIDIA GeForce 940MX             |
| 系统平台 |         Window 10 家庭中文版（64位）         |
| 软件环境 | Paddle2.2、 Pytorch3.8、Cuda 10.1、Anaconda3 |

Paddle环境安装参考： https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html 

安装CPU版本时候使用到了清华镜像源：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paddlepaddle

GPU版本：python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

Paddle与Pytorch环境配置使用对比： 在单机安装中，都安装在了conda环境，Paddle安装比较顺利，直接按照文档安装即可，与Pytorch安装没有太大区别，单机测试稳定性也都比较良好。



# 三、API使用及对比

在API使用上，首先感觉paddle升级后的 paddle.xxx  （例如：paddle.device  paddle.nn  paddle.vision ）比之前的 padddle.fluid.xxx 好用很多，还有就是新增加的高层API个人比较喜欢，一是对初学者比较友好、易用，二是对于开发者可以节省代码量，更简洁直观一些，在（六、动态图单机训练）中进行了代码展示和对比分析。

与Pytorch相比，基础API的结构和调用没有太大区别，但是在速度上，paddle的基础API会更快一点，如果是利用了paddle高层API，速度会快很多，在同样epoch的情况下，能减少大约三分之二的训练时间。

总体来说，使用像paddle.Model、paddle.vision这样的高级API进行封装调用，使用体验比较好，个人感觉在以后深度学习模型普遍使用时，高层API会更受欢迎，也会成为模型训练测试中更为流行的一种方法。



# 四、Tensor 索引

在了解Paddle的Tensor索引和其切片规则以及逻辑相关函数重写规则等内容后，结合指南内容（ https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html#id1 ）和模型训练过程中的Tensor索引体验，感觉在通过索引或切片修改 Tensor 的整体过程有些冗余，稳定性也会下降。虽然使用指南里说明了修改会导致原值不会被保存，可能会给梯度计算引入风险 ，但是在这点上个人感觉Pytorch的体验要好于Paddle。

总的来说，在模型训练中利用Tensor加载数据集等操作上 Pytorch与 Paddle的体验并没有太大区别，但整体的感觉Pytorch的Tensor 索引更好一些，个人感觉Paddle在修改 Tensor的部分上可以增加一些文档说明。



# 五、NumPy兼容性分析及对比

NumPy在Paddle的体验，感觉和Pytorch的体验并无区别，但是在阅读使用文档时的体验感较好，内容叙述很详细 （文档链接：https://www.paddlepaddle.org.cn/tutorials/projectdetail/3466356 ）

这部分个人体验较好的第一点就是 飞桨的Tensor高度兼容Numpy数组（array），在基础数据结构和方法上，增加了很多适用于深度学习任务的参数和方法，如：反向计算梯度，更灵活的指定运行硬件等。 

第二点就是对于刚使用Paddle的新手，这部分需要注意的就是 Paddle的Tensor虽然可以与Numpy的数组方便的互相转换 ，但是有两个场景优先使用Paddle的Tensor 比较好:

- 场景一：在组网程序中，对网络中向量的处理，务必使用Tensor，而不建议转成Numpy的数组。如果在组网过程中转成Numpy的数组，并使用Numpy的函数会拖慢整体性能；
- 场景二：在数据处理和模型后处理等场景，建议优先使用Tensor，主要是飞桨为AI硬件做了大量的适配和性能优化工作，部分情况下会获得更好的使用体验和性能。

建议：这两个场景内容可以增加一些实例，可能会使新手在这部分的理解更为透彻。

总体来说：Tensor与Numpy数组的兼容与转换，Paddle体验更好一点，兼容性上与Pytorch感觉没区别，但是Paddle的兼容转换处理上更具有一些前瞻性。



# 六、动态图单机训练

（1）使用 Pytorch 完成一个图像分类的动态图单机训练例子（MNIST数据集）

```python
import torch
from torch import nn
from net import LeNet
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.ToTensor()     # 仅对数据做转换为 tensor 格式操作
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据集加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# 给测试集创建一个数据集加载器
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
model = LeNet().to(device)

# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义优化器（SGD：随机梯度下降）
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔 10 个 epoch 变为原来的 0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print('train_loss：' + str(loss / n))
    print('train_acc：' + str(current / n))

# 定义测试函数
def test(dataloader, model, loss_fn):
    # 将模型转换为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 非训练，推理期用到（测试时模型参数不用更新，所以 no_grad）
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('test_loss：' + str(loss / n))
        print('test_acc：' + str(current / n))

# 开始训练
epoch = 5
for t in range(epoch):
    lr_scheduler.step()
    print(f"Epoch {t + 1}\n----------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    torch.save(model.state_dict(), "save_model/{}model.pth".format(t))    # 模型保存
print("Done!")
```

（2）使用 Paddle 完成一个图像分类的动态图单机训练例子（MNIST数据集）

```python
import paddle
from paddle.vision.transforms import Compose, Normalize
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

import paddle
import paddle.nn.functional as F
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
    
#方法一 高层API
from paddle.metric import Accuracy
model = paddle.Model(LeNet())   # 用Model封装模型
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )

# 训练模型
model.fit(train_dataset,
        epochs=5,
        batch_size=64,
        verbose=1
        )
model.evaluate(test_dataset, batch_size=64, verbose=1)

#方法2 基础API
import paddle.nn.functional as F
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
# 加载训练集 batch_size 设为 64
def train(model):
    model.train()
    epochs = 2
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()
model = LeNet()
train(model)

test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
# 加载测试数据集
def test(model):
    model.eval()
    batch_size = 64
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 20 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))
test(model)
```

（3）两个程序的运行结果

一、Pytorch程序运行结果

```python
#下载数据
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./data\MNIST\raw\train-images-idx3-ubyte.gz
9913344it [00:03, 2813467.92it/s]                             
Extracting ./data\MNIST\raw\train-images-idx3-ubyte.gz to ./data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./data\MNIST\raw\train-labels-idx1-ubyte.gz
29696it [00:00, 29740700.00it/s]         
Extracting ./data\MNIST\raw\train-labels-idx1-ubyte.gz to ./data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./data\MNIST\raw\t10k-images-idx3-ubyte.gz
1649664it [00:01, 1119159.12it/s]                             
Extracting ./data\MNIST\raw\t10k-images-idx3-ubyte.gz to ./data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./data\MNIST\raw\t10k-labels-idx1-ubyte.gz
5120it [00:00, 5302428.76it/s]          
Extracting ./data\MNIST\raw\t10k-labels-idx1-ubyte.gz to ./data\MNIST\raw
```

```python
#五次epoch结果
Epoch 1
----------------------
train_loss：2.301828587023417
train_acc：0.10976666666666667
test_loss：2.2998157671610513
test_acc：0.186
Epoch 2
----------------------
train_loss：2.292567366727193
train_acc：0.13415
test_loss：2.268421948369344
test_acc：0.20193333333333333
Epoch 3
----------------------
train_loss：1.2924396817684174
train_acc：0.5939
test_loss：0.5503138323009014
test_acc：0.82955
Epoch 4
----------------------
train_loss：0.45181470778187116
train_acc：0.86275
test_loss：0.3795674962008993
test_acc：0.88765
Epoch 5
----------------------
train_loss：0.35491258655836183
train_acc：0.8926666666666667
test_loss：0.3223567478398482
test_acc：0.9044666666666666
```

二、Paddle程序运行结果

由于paddle文档中提供的数据集下载代码一直报错(已在报错汇总中展示)，故进行了手动下载数据集

​	1、使用高层API结果

```python
#附第5个epoch
Epoch 5/5
step  10/938 [..............................] - loss: 0.0325 - acc: 0.9938 - ETA: 35s - 38ms/step
step  20/938 [..............................] - loss: 0.0050 - acc: 0.9922 - ETA: 32s - 35ms/step
step  30/938 [..............................] - loss: 0.0094 - acc: 0.9932 - ETA: 29s - 32ms/step
step  40/938 [>.............................] - loss: 0.0344 - acc: 0.9910 - ETA: 27s - 31ms/step
step  50/938 [>.............................] - loss: 0.0020 - acc: 0.9916 - ETA: 26s - 30ms/step
step  60/938 [>.............................] - loss: 0.0121 - acc: 0.9917 - ETA: 25s - 29ms/step
step  70/938 [=>............................] - loss: 0.0026 - acc: 0.9913 - ETA: 25s - 29ms/step
step  80/938 [=>............................] - loss: 0.0151 - acc: 0.9914 - ETA: 24s - 29ms/step
step  90/938 [=>............................] - loss: 0.0030 - acc: 0.9910 - ETA: 24s - 28ms/step
step 100/938 [==>...........................] - loss: 0.1395 - acc: 0.9905 - ETA: 23s - 28ms/step
step 110/938 [==>...........................] - loss: 0.0344 - acc: 0.9902 - ETA: 23s - 28ms/step
step 120/938 [==>...........................] - loss: 0.0175 - acc: 0.9904 - ETA: 23s - 28ms/step
step 130/938 [===>..........................] - loss: 0.0511 - acc: 0.9899 - ETA: 22s - 28ms/step
step 140/938 [===>..........................] - loss: 0.0136 - acc: 0.9903 - ETA: 22s - 28ms/step
step 150/938 [===>..........................] - loss: 0.0068 - acc: 0.9901 - ETA: 21s - 28ms/step
step 160/938 [====>.........................] - loss: 0.0128 - acc: 0.9898 - ETA: 21s - 28ms/step
step 170/938 [====>.........................] - loss: 0.0447 - acc: 0.9898 - ETA: 21s - 28ms/step
step 180/938 [====>.........................] - loss: 0.0275 - acc: 0.9900 - ETA: 20s - 28ms/step
step 190/938 [=====>........................] - loss: 0.0488 - acc: 0.9901 - ETA: 20s - 28ms/step
step 200/938 [=====>........................] - loss: 0.0593 - acc: 0.9899 - ETA: 20s - 28ms/step
step 210/938 [=====>........................] - loss: 0.0049 - acc: 0.9899 - ETA: 20s - 28ms/step
step 220/938 [======>.......................] - loss: 0.0186 - acc: 0.9898 - ETA: 19s - 27ms/step
step 230/938 [======>.......................] - loss: 0.0214 - acc: 0.9900 - ETA: 19s - 27ms/step
step 240/938 [======>.......................] - loss: 0.0067 - acc: 0.9902 - ETA: 19s - 27ms/step
step 250/938 [======>.......................] - loss: 0.0195 - acc: 0.9902 - ETA: 18s - 27ms/step
step 260/938 [=======>......................] - loss: 0.0310 - acc: 0.9901 - ETA: 18s - 27ms/step
step 270/938 [=======>......................] - loss: 0.0248 - acc: 0.9902 - ETA: 18s - 27ms/step
step 280/938 [=======>......................] - loss: 0.0213 - acc: 0.9901 - ETA: 17s - 27ms/step
step 290/938 [========>.....................] - loss: 0.0156 - acc: 0.9903 - ETA: 17s - 27ms/step
step 300/938 [========>.....................] - loss: 0.0069 - acc: 0.9906 - ETA: 17s - 27ms/step
step 310/938 [========>.....................] - loss: 0.0361 - acc: 0.9904 - ETA: 17s - 27ms/step
step 320/938 [=========>....................] - loss: 0.0418 - acc: 0.9904 - ETA: 16s - 27ms/step
step 330/938 [=========>....................] - loss: 0.0060 - acc: 0.9903 - ETA: 16s - 27ms/step
step 340/938 [=========>....................] - loss: 0.0587 - acc: 0.9903 - ETA: 16s - 27ms/step
step 350/938 [==========>...................] - loss: 0.0434 - acc: 0.9904 - ETA: 16s - 27ms/step
step 360/938 [==========>...................] - loss: 6.9384e-04 - acc: 0.9904 - ETA: 15s - 27ms/step
step 370/938 [==========>...................] - loss: 0.0134 - acc: 0.9904 - ETA: 15s - 27ms/step    
step 380/938 [===========>..................] - loss: 0.0278 - acc: 0.9903 - ETA: 15s - 27ms/step
step 390/938 [===========>..................] - loss: 5.5189e-04 - acc: 0.9902 - ETA: 14s - 27ms/step
step 400/938 [===========>..................] - loss: 0.0023 - acc: 0.9904 - ETA: 14s - 27ms/step    
step 410/938 [============>.................] - loss: 0.0105 - acc: 0.9904 - ETA: 14s - 27ms/step
step 420/938 [============>.................] - loss: 0.0398 - acc: 0.9901 - ETA: 14s - 27ms/step
step 430/938 [============>.................] - loss: 0.0169 - acc: 0.9902 - ETA: 13s - 27ms/step
step 440/938 [=============>................] - loss: 0.0013 - acc: 0.9902 - ETA: 13s - 27ms/step
step 450/938 [=============>................] - loss: 0.0074 - acc: 0.9901 - ETA: 13s - 27ms/step
step 460/938 [=============>................] - loss: 0.0651 - acc: 0.9899 - ETA: 12s - 27ms/step
step 470/938 [==============>...............] - loss: 0.0130 - acc: 0.9900 - ETA: 12s - 27ms/step
step 480/938 [==============>...............] - loss: 0.0677 - acc: 0.9900 - ETA: 12s - 27ms/step
step 490/938 [==============>...............] - loss: 0.0147 - acc: 0.9901 - ETA: 12s - 27ms/step
step 500/938 [==============>...............] - loss: 0.0120 - acc: 0.9901 - ETA: 11s - 27ms/step
step 510/938 [===============>..............] - loss: 0.0191 - acc: 0.9901 - ETA: 11s - 27ms/step
step 520/938 [===============>..............] - loss: 0.0296 - acc: 0.9901 - ETA: 11s - 27ms/step
step 530/938 [===============>..............] - loss: 0.0488 - acc: 0.9900 - ETA: 11s - 27ms/step
step 540/938 [================>.............] - loss: 0.0239 - acc: 0.9901 - ETA: 10s - 27ms/step
step 550/938 [================>.............] - loss: 0.0303 - acc: 0.9900 - ETA: 10s - 27ms/step
step 560/938 [================>.............] - loss: 0.0287 - acc: 0.9900 - ETA: 10s - 27ms/step
step 570/938 [=================>............] - loss: 0.0375 - acc: 0.9900 - ETA: 9s - 27ms/step 
step 580/938 [=================>............] - loss: 0.0197 - acc: 0.9900 - ETA: 9s - 27ms/step
step 590/938 [=================>............] - loss: 0.0265 - acc: 0.9900 - ETA: 9s - 27ms/step
step 600/938 [==================>...........] - loss: 0.0615 - acc: 0.9901 - ETA: 9s - 27ms/step
step 610/938 [==================>...........] - loss: 0.0036 - acc: 0.9901 - ETA: 8s - 27ms/step
step 620/938 [==================>...........] - loss: 0.0079 - acc: 0.9900 - ETA: 8s - 27ms/step
step 630/938 [===================>..........] - loss: 0.0071 - acc: 0.9901 - ETA: 8s - 27ms/step
step 640/938 [===================>..........] - loss: 6.9407e-04 - acc: 0.9902 - ETA: 8s - 27ms/step
step 650/938 [===================>..........] - loss: 0.0024 - acc: 0.9902 - ETA: 7s - 27ms/step    
step 660/938 [====================>.........] - loss: 0.0016 - acc: 0.9902 - ETA: 7s - 27ms/step
step 670/938 [====================>.........] - loss: 0.0069 - acc: 0.9901 - ETA: 7s - 27ms/step
step 680/938 [====================>.........] - loss: 0.0023 - acc: 0.9901 - ETA: 6s - 27ms/step
step 690/938 [=====================>........] - loss: 0.0089 - acc: 0.9901 - ETA: 6s - 27ms/step
step 700/938 [=====================>........] - loss: 0.0108 - acc: 0.9900 - ETA: 6s - 27ms/step
step 710/938 [=====================>........] - loss: 0.0155 - acc: 0.9899 - ETA: 6s - 27ms/step
step 720/938 [======================>.......] - loss: 0.0303 - acc: 0.9898 - ETA: 5s - 27ms/step
step 730/938 [======================>.......] - loss: 0.0405 - acc: 0.9898 - ETA: 5s - 27ms/step
step 740/938 [======================>.......] - loss: 0.0304 - acc: 0.9899 - ETA: 5s - 27ms/step
step 750/938 [======================>.......] - loss: 0.0065 - acc: 0.9897 - ETA: 5s - 27ms/step
step 760/938 [=======================>......] - loss: 0.0091 - acc: 0.9898 - ETA: 4s - 27ms/step
step 770/938 [=======================>......] - loss: 0.0371 - acc: 0.9896 - ETA: 4s - 27ms/step
step 780/938 [=======================>......] - loss: 0.0048 - acc: 0.9896 - ETA: 4s - 27ms/step
step 790/938 [========================>.....] - loss: 0.0036 - acc: 0.9897 - ETA: 4s - 27ms/step
step 800/938 [========================>.....] - loss: 0.0233 - acc: 0.9896 - ETA: 3s - 27ms/step
step 810/938 [========================>.....] - loss: 0.0547 - acc: 0.9896 - ETA: 3s - 27ms/step
step 820/938 [=========================>....] - loss: 0.0011 - acc: 0.9896 - ETA: 3s - 27ms/step
step 830/938 [=========================>....] - loss: 0.0079 - acc: 0.9896 - ETA: 2s - 27ms/step
step 840/938 [=========================>....] - loss: 0.0132 - acc: 0.9896 - ETA: 2s - 27ms/step
step 850/938 [==========================>...] - loss: 0.0134 - acc: 0.9896 - ETA: 2s - 27ms/step
step 860/938 [==========================>...] - loss: 0.0065 - acc: 0.9896 - ETA: 2s - 27ms/step
step 870/938 [==========================>...] - loss: 0.0106 - acc: 0.9897 - ETA: 1s - 27ms/step
step 880/938 [===========================>..] - loss: 0.0312 - acc: 0.9896 - ETA: 1s - 27ms/step
step 890/938 [===========================>..] - loss: 0.0169 - acc: 0.9897 - ETA: 1s - 27ms/step
step 900/938 [===========================>..] - loss: 0.0187 - acc: 0.9897 - ETA: 1s - 27ms/step
step 910/938 [============================>.] - loss: 0.0925 - acc: 0.9897 - ETA: 0s - 27ms/step
step 920/938 [============================>.] - loss: 0.0317 - acc: 0.9898 - ETA: 0s - 27ms/step
step 930/938 [============================>.] - loss: 0.0448 - acc: 0.9898 - ETA: 0s - 27ms/step
step 938/938 [==============================] - loss: 0.0140 - acc: 0.9897 - 27ms/step          
Eval begin...
step  10/157 [>.............................] - loss: 0.2273 - acc: 0.9828 - ETA: 1s - 12ms/step
step  20/157 [==>...........................] - loss: 0.1525 - acc: 0.9773 - ETA: 1s - 11ms/step
step  30/157 [====>.........................] - loss: 0.1391 - acc: 0.9771 - ETA: 1s - 11ms/step
step  40/157 [======>.......................] - loss: 0.0088 - acc: 0.9785 - ETA: 1s - 11ms/step
step  50/157 [========>.....................] - loss: 0.0051 - acc: 0.9803 - ETA: 1s - 11ms/step
step  60/157 [==========>...................] - loss: 0.1621 - acc: 0.9797 - ETA: 1s - 10ms/step
step  70/157 [============>.................] - loss: 0.0265 - acc: 0.9795 - ETA: 0s - 10ms/step
step  80/157 [==============>...............] - loss: 0.0019 - acc: 0.9801 - ETA: 0s - 10ms/step
step  90/157 [================>.............] - loss: 0.0439 - acc: 0.9814 - ETA: 0s - 10ms/step
step 100/157 [==================>...........] - loss: 0.0033 - acc: 0.9828 - ETA: 0s - 10ms/step
step 110/157 [====================>.........] - loss: 3.9403e-04 - acc: 0.9837 - ETA: 0s - 10ms/step
step 120/157 [=====================>........] - loss: 6.5309e-04 - acc: 0.9846 - ETA: 0s - 10ms/step
step 130/157 [=======================>......] - loss: 0.0735 - acc: 0.9849 - ETA: 0s - 10ms/step    
step 140/157 [=========================>....] - loss: 9.8257e-05 - acc: 0.9856 - ETA: 0s - 10ms/step
step 150/157 [===========================>..] - loss: 0.0412 - acc: 0.9859 - ETA: 0s - 10ms/step    
step 157/157 [==============================] - loss: 2.9252e-04 - acc: 0.9860 - 10ms/step      
Eval samples: 10000
```

​	2、使用基础API结果

```python
#附5次epoch
epoch: 0, batch_id: 0, loss is: [2.9994564], acc is: [0.0625]
epoch: 0, batch_id: 300, loss is: [0.08384503], acc is: [0.96875]
epoch: 0, batch_id: 600, loss is: [0.06951822], acc is: [0.984375]
epoch: 0, batch_id: 900, loss is: [0.1054411], acc is: [0.953125]
epoch: 1, batch_id: 0, loss is: [0.0715376], acc is: [0.96875]
epoch: 1, batch_id: 300, loss is: [0.14129372], acc is: [0.953125]
epoch: 1, batch_id: 600, loss is: [0.00361754], acc is: [1.]
epoch: 1, batch_id: 900, loss is: [0.00827341], acc is: [1.]
epoch: 2, batch_id: 0, loss is: [0.05238173], acc is: [0.984375]
epoch: 2, batch_id: 300, loss is: [0.00865405], acc is: [1.]
epoch: 2, batch_id: 600, loss is: [0.03549637], acc is: [0.984375]
epoch: 2, batch_id: 900, loss is: [0.02600437], acc is: [1.]
epoch: 3, batch_id: 0, loss is: [0.02365134], acc is: [1.]
epoch: 3, batch_id: 300, loss is: [0.0848916], acc is: [0.953125]
epoch: 3, batch_id: 600, loss is: [0.01307216], acc is: [1.]
epoch: 3, batch_id: 900, loss is: [0.01843782], acc is: [1.]
epoch: 4, batch_id: 0, loss is: [0.00281677], acc is: [1.]
epoch: 4, batch_id: 300, loss is: [0.01466173], acc is: [1.]
epoch: 4, batch_id: 600, loss is: [0.04725911], acc is: [0.984375]
epoch: 4, batch_id: 900, loss is: [0.00772327], acc is: [1.]
batch_id: 0, loss is: [0.03467739], acc is: [0.984375]
batch_id: 20, loss is: [0.15250863], acc is: [0.953125]
batch_id: 40, loss is: [0.13340972], acc is: [0.984375]
batch_id: 60, loss is: [0.06206714], acc is: [0.953125]
batch_id: 80, loss is: [0.00384411], acc is: [1.]
batch_id: 100, loss is: [0.00386263], acc is: [1.]
batch_id: 120, loss is: [0.00981056], acc is: [1.]
batch_id: 140, loss is: [0.07646853], acc is: [0.984375]

Process finished with exit code 0
```

这部分简单说就是Paddle的高层API比基础API运行速度快，且简单好用，体验感较好。

与Pytorch相比，Paddle文档中提供的代码下载不了数据集，需要手动下载。

# 七、各种 trick 的用法

这部分在Paddle使用过程中，优化器等trick的使用体验与Pytorch感觉没有区别

# 八、报错汇总

Paddle加载数据集报错，无法下载MNIST数据集，需要手动进行下载，（使用了多台电脑测试，均会出现此情况）

```python
File "E:\anaconda\lib\site-packages\paddle\vision\datasets\mnist.py", line 98, in __init__
    self.image_path = _check_exists_and_download(
  File "E:\anaconda\lib\site-packages\paddle\dataset\common.py", line 207, in _check_exists_and_download
    return paddle.dataset.common.download(url, module_name, md5)
  File "E:\anaconda\lib\site-packages\paddle\dataset\common.py", line 82, in download
    raise RuntimeError("Cannot download {0} within retry limit {1}".
RuntimeError: Cannot download https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz within retry limit 3
```

```python
File "E:\anaconda\lib\site-packages\paddle\vision\datasets\cifar.py", line 122, in __init__
    self.data_file = _check_exists_and_download(
  File "E:\anaconda\lib\site-packages\paddle\dataset\common.py", line 207, in _check_exists_and_download
    return paddle.dataset.common.download(url, module_name, md5)
  File "E:\anaconda\lib\site-packages\paddle\dataset\common.py", line 82, in download
    raise RuntimeError("Cannot download {0} within retry limit {1}".
RuntimeError: Cannot download https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz within retry limit 3
```

按照文档提供的'DatasetFolder', 'ImageFolder', 'MNIST', 'FashionMNIST', 'Flowers', 'Cifar10',多种数据集进行了下载测试，均无法在单机上加载数据集，需要手动下载数据集。 

 且数据集的保存地址为一个缓存空间，用户在使用的时候可能找不到数据集，如/public/home/username/.cache/paddle/dataset目录。 

 而pytorch的加载数据集API会把数据集加载到当前目录，这一点的体验要优于Paddle。 
