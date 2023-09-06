# 深度体验飞桨分布式训练功能，并产出一份评估报告

|       |              |
| ------------ | ------------------------------------------------- |
| 提交作者     | mkm  wjc                                             |
| 提交时间     | 2022-04-02                                        |
| 版本号       | V1.0                                              |
| 依赖飞桨版本 | develop                                            |
| 文件名       | 【Hackathon No.112】 PR.md |

# 功能对比表格

|       功能对比表格            |    |      |             |
| ------------ | ------------------------------------------------- | ------------------------------ | -------------------- |
| 类型     | 选项（选择一项）                                           |  您的选择               |    使用时遇到的问题或建议 |
| 任务模式     |     单机多卡 or 多机多卡                |    单机多卡             |   - |
| 运行方式     |     物理机 or docker or AI开发平台       |             docker               |  -        |
| 任务类型       |  集合通信（GPU）or 参数服务器（CPU/GPU）             |     集合通信（GPU）                | -        |
| 分布式训练使用程度 |  成功实践过 or 熟悉基本过程 or 了解基本过程 or 不熟悉        |  成功实践过     |  在使用曙光平台docker创建镜像后难以使用多节点进行分布式训练，建议推出一种可以直接在E-Shell安装的方式，可以使得任务通过队列的方式提交    |
| 分布式策略使用程度       | 使用过调优策略 or 了解部分优化策略 or 没有使用过优化策略       |   了解部分优化策略     |    无      |
| 分布式数据处理使用程度      |   熟悉 or 了解 or 不清楚        |  熟悉                  |    无      |
| 分布式模型保存和加载使用程度       |    熟悉 or 了解 or 不清楚    |  熟悉                  |  无        |
| 遇到问题如何解决（可多选）       |   日志排查 or 社区反馈 or 提Issue or 其他     | 日志排查、社区反馈         |  建议在官方文档中对常见问题进行汇总，便于使用者解决问题     |


# 一、摘要

本评估方案将从以下几个方面对paddle分布式框架进行评估：
- 1、环境配置，对比pytorch环境以及paddle环境在曙光超算上的配置
  对曙光超算如何使用paddle进行分布式计算进行了介绍

- 2、Fleet API的使用，对比Pytorch API与Fleet API的区别

- 3、分布式动态图的训练，使用pytorch和paddle完成在曙光超算上的分布式训练
  对文档中代码进行的重写，导入了paddle.vision的部分包
  对鲜花数据集加载的代码进行了修改，改为：
```python
  train_dataset = paddle.vision.datasets.Flowers(mode='train', transform=transform)
```
```python
  optimizer.minimize(avg_loss)改为optimizer.step()
```    
   鲜花数据集的label索引是从1开始的，不是从0开始的，需要手工减1。

- 4、文档质量评估，对paddle文档质量进行评估
  文档中的代码有些旧，比如分布式训练快速开始中1.3动态图训练的代码许多API比较旧。
  鲜花数据集的label索引是从1开始的，不是从0开始的，需要手工减1。在文档质量评估方面，认为文档对错误报告的解决方案不足。

- 5、错误汇总
 
 
#  二、环境配置

## （1）在曙光超算昆山计算服务器部署pytorch分布式环境，给出部署步骤
- 首先安装anaconda
```python
bash Anaconda3-2020.07-Linux-x86_64.sh –u
```    
- 创建并进入python3.6环境
```python
conda create -n pytorch-1.9 python=3.6
conda activate pytorch-1.9
```    
- 安装pytorch-1.9（适配rocm-4.0.1及以上）PyTorch1.8和PyTorch1.9安装wheel包在公共目录：
```python
/public/software/apps/DeepLearning/whl/rocm-4.0.1/
```    
- 安装pytorch_1.9-rocm_4.0.1(使用清华源)
```python
pip install /public/software/apps/DeepLearning/whl/rocm-4.0.1/torch-1.9.0+rocm4.0.1-cp36-cp36m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple/
```    
- 对于torchverion的安装不能按照曙光官方给定的方法来，否则在torchversion在运行自定义算子时会出现错误，所以需要进行源码安装，安装方法如下:
```python
1、本地下载对应的torchvision分支源码包：https://github.com/pytorch/vision上传集群，
2、进入对应的conda环境，加载对应的rocm（这里rocm4.0.1）版本；
3、conda install libpng -y 
4、conda install jpeg -y 
5、pip3 install numpy pillow matplotlib ninja -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
6、使用salloc申请计算结点，使用ssh登录至计算节点，并进入对应的conda环境加载rocm（这里rocm4.0.1），执行编译：CC=clang CXX=clang++ python setup.py install
```    
## （2）在曙光超算昆山计算服务器部署paddle分布式环境，给出部署步骤
若在曙光平台不能使用sudo指令，从而导致不能直接使用docker pull方式安装镜像，可以直接使用曙光内置的可视化容器方式安装：
- 1、点击我的服务，计算智能服务
- 2、点击容器服务
- 3、点击容器管理添加镜像
- 4、镜像添加，源镜像名称填：paddlepaddle/paddle，源镜像标签填：latest-dev-rocm4.0-miopen2.11
- 5、添加完成后添加完成即可添加
- 6、使用镜像，对镜像进行推送，快速访问选择，是
- 7、点击AI 服务，点击之前创建的容器
- 8、点击启动容器，即可启动rocm-4.0.1环境
- 9、打开容器后，该计算环境不能连接互联网，该环境可以调用此前终端设置的一切文件以及环境（包括conda环境），故需要提取打开命令行（E-Shell）创建一个conda环境安装paddle。
- 10、曙光服务器conda环境安装方式参考曙光超算官方链接：
```python
https://www.hpccube.com/doc/1.0.6/11250/general-handbook/compile/Anaconda.html
```    
- 11、激活conda环境并新建一个conda环境，进入该环境
```python
source activate
conda create -n paddle python=3.7
conda activate paddle
```  
- 12、在曙光上使用paddle官网给出的安装方式会出现错误。
```python
python -m pip install paddlepaddle-rocm==2.2.2.rocm401.miopen211 -f https://www.paddlepaddle.org.cn/whl/rocm/stable.whl（此方法在曙光无法安装）
```  
- 13、故需要提前下载whl文件，下载链接：
```python
https://www.paddlepaddle.org.cn/whl/rocm/stable.whl
```  
- 14、paddlepaddle_rocm-2.2.2-cp37-cp37m-linux_x86_64.whl，版本经过测试可以安装。安装指令:
```python
pip install paddlepaddle_rocm-2.2.2-cp37-cp37m-linux_x86_64.whl -i  https://pypi.tuna.tsinghua.edu.cn/simple/
```  
- 15、在安装完上述操作后还需要手动安装两个库opencv-python以及scipy
```python
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
```  

## （3）对比两者的易用性与区别
Pytorch的分布式环境在曙光平台安装时需要手动编译torchversion，这一点上pytorch比较繁琐。但是pytorch的环境在曙光平台比较稳定，而paddle环境在曙光平台经常不稳定，有时候能运行，有时候不能运行。
![image](https://user-images.githubusercontent.com/102226413/164142960-e956efce-a8fe-40ea-bfba-a83b8f8203c5.png)

上述问题是rocm版本问题，需要使用rocm-4.0.1版本。 修改rocm版本的方法为.  module switch compiler/rocm/4.0.1

另外有一些问题没办法解决，我们使用的办法是重新开启镜像（多次开启后就会有可以使用的时候）无法解决的问题截图如下：
![image](https://user-images.githubusercontent.com/102226413/164143125-70d0e4ff-46d7-4461-8cb0-72c14e98b8e0.png)

![image](https://user-images.githubusercontent.com/102226413/164143166-cde2793b-eb06-43a3-92d1-bfa68c2f1558.png)


另外，我们在曙光上使用paddle的方法为开启镜像的方式，但是曙光平台对docker镜像的支持不太好，每次镜像保持的时间最多为72小时，而且每次关闭镜像后，无法重新开启原先镜像。为了方便使用，希望能够支持 任务提交方式运行的paddle分布式框架。而且任务提交的方式还方便管理多节点运行。


#  三、Fleet API的使用
## （1）分析pytorch分布式框架DDP某些API的使用
- 导入必要的分布式训练依赖包
```python
import torch.distributed as dist
```  
- 初始化DDP分布式环境，需要指定backend, init_method,world_size, rank四个参数
```python
dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
```  
- 优化器，不需要进行分布式函数的包装
- 通过DistributedDataParallel获取分布式model，用于支持分布式训练
```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
``` 

## （2）按照文档内容使用Fleet API
文档链接如下：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/cluster_quick_start_cn.html
本次实验测试的Fleet API如下：
- 导入必要的分布式训练依赖包
```python
from paddle.distributed import fleet
```  
- 初始化Fleet环境
```python
fleet.init(is_collective=True)
```  
- 分布式优化器
```python
optimizer = fleet.distributed_optimizer(optimizer)
``` 
- 通过Fleet API获取分布式model，用于支持分布式训练
```python
resnet = fleet.distributed_model(resnet)
``` 
## （3）比较
pytorch和paddle的分布式代码基本相似，pytorch初始化DDP分布式环境，需要指定backend, init_method,world_size, rank四个参数，比较麻烦。

# 四、分布式动态图训练 

## （1）使用pytorch完成一个图像分类的动态图分布式例子

DDP分布式代码，测试flower数据集：
```python
import torch.nn as nn
import torch.utils.data as D
from torchvision import transforms
import torchvision
import torch
import os
import scipy.io as scio
import tarfile
import numpy as np
from PIL import Image
import torch.distributed as dist
MODE_FLAG_MAP = {'train': 'tstid', 'test': 'trnid', 'valid': 'valid'}


class Flowers(torch.utils.data.Dataset):
    def __init__(self,
                 data_file=None,
                 label_file=None,
                 setid_file=None,
                 mode='train',
                 transform=None,
                 backend=None):
        assert mode.lower() in ['train', 'valid', 'test'], \
            "mode should be 'train', 'valid' or 'test', but got {}".format(mode)

        if backend is None:
            backend = torchvision.get_image_backend()
        self.backend = backend

        flag = MODE_FLAG_MAP[mode.lower()]
        self.transform = transform
        data_tar = tarfile.open(data_file)
        self.data_path = data_file.replace(".tgz", "/")
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        data_tar.extractall(self.data_path)

        self.labels = scio.loadmat(label_file)['labels'][0]
        self.indexes = scio.loadmat(setid_file)[flag][0]

    def __getitem__(self, idx):
        index = self.indexes[idx]
        label = np.array([self.labels[index - 1]])
        img_name = "jpg/image_%05d.jpg" % index
        image = os.path.join(self.data_path, img_name)
        image = Image.open(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, label.astype('int64')

    def __len__(self):
        return len(self.indexes)


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
    
def main():
    # 初始化
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ['LOCAL_RANK'])
    device = gpu
    torch.cuda.set_device(gpu)
    print("rank:", rank, "world size:", world_size, "gpu:", gpu)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    # 数据增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机
            transforms.CenterCrop(224),  # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    batch_size = 32
    # 数据集构建与切分
    train_dataset = Flowers(mode='train', data_file="./102flowers.tgz", label_file="./imagelabels.mat", setid_file="./setid.mat", transform=data_transforms['train'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    # 模型构建
    model = torchvision.models.resnet50(pretrained=False, num_classes=102).to(device)
    # 构建DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9)
    # 开始训练
    epochs = 100
    best_acc = 0.0
    for epoch in range(epochs):
        if int(rank) == 0:
            print('Epoch {}/{}'.format(epoch + 1, epochs))
        running_loss = 0.0
        running_corrects = 0
        train_sampler.set_epoch(epoch)
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze() - 1
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if int(rank) == 0:
                print("loss:", loss)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)
        running_corrects = reduce_value(running_corrects, average=False)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 1.0 * running_corrects.item() / train_sampler.total_size
        best_acc = max(best_acc, epoch_acc)
        if int(rank) == 0:
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))
            
    if int(rank) == 0:
        print('Training complete Best val Acc: {:4f}'.format(best_acc))

if __name__ == "__main__":
    main()

```

```python
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## （2）使用paddle完成一个图像分类的分布式例子
由于文档中提供的代码出现了较多问题，我们对代码进行了重新的编写。修改的部分有：
- 1、导入了paddle.vision的部分包，
- 2、对鲜花数据集加载的代码进行了修改，改为：
- train_dataset = paddle.vision.datasets.Flowers(mode='train', transform=transform)

官网的代码不能再曙光平台下载鲜花数据集，所以我们需要提取下载离线鲜花数据集报错如下：
![image](https://user-images.githubusercontent.com/102226413/164143513-24236f90-975d-47f1-a1db-7e18e6c94c9c.png)

且数据集的保存地址为一个缓存空间，用户在使用的时候可能找不到数据集，如/public/home/username/.cache/paddle/dataset目录。
而pytorch的加载数据集API会吧数据集加载到当前目录，方便了使用者。
![image](https://user-images.githubusercontent.com/102226413/164144065-2fea8ac3-dcf5-48ac-a4c7-05cace99c611.png)



- 3、optimizer.minimize(avg_loss)改为optimizer.step()
- 4、鲜花数据集的label索引是从1开始的，不是从0开始的，需要手工减1。

train_fleet_dygraph.py重构后的代码： 
```python
# -*- coding: UTF-8 -*-
import numpy as np
import paddle
# 导入必要分布式训练的依赖包
from paddle.distributed import fleet
# 导入模型文件
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
from paddle.io import Dataset, BatchSampler, DataLoader, DistributedBatchSampler
from paddle.vision.transforms import Compose
from paddle.vision.transforms import ToTensor
from paddle.vision.transforms import Resize
base_lr = 0.1  # 学习率
momentum_rate = 0.9  # 冲量
l2_decay = 1e-4  # 权重衰减

epoch = 10  # 训练迭代次数
batch_size = 32  # 训练批次大小
class_dim = 102


# 设置数据读取器
# def reader_decorator(reader):
#     def __reader__():
#         for item in reader():
#             img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
#             label = np.array(item[1]).astype('int64').reshape(1)
#             yield img, label
#
#     return __reader__


# 设置优化器
def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer


# 设置训练函数
def train_resnet():
    # 初始化Fleet环境
    fleet.init(is_collective=True)

    # resnet = resnet34(class_dim=class_dim, layers=50)
    resnet = ResNet(BottleneckBlock, 50, num_classes=class_dim)

    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    # 分布式优化器
    optimizer = fleet.distributed_optimizer(optimizer)
    # 通过Fleet API获取分布式model，用于支持分布式训练
    resnet = fleet.distributed_model(resnet)
    # 构建分布式数据集  归一化 / 255 并且转成HWC --> CHW格式
    # transform = ToTensor()
    transform = Compose([
        Resize(size=(224, 224)),
        ToTensor()
    ])
    train_dataset = paddle.vision.datasets.Flowers(mode='train', transform=transform)
    # 数据集的拆分  构建分布式数据集
    train_sampler = DistributedBatchSampler(train_dataset, 16, shuffle=True)
    # , num_workers=2
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    # train_reader = paddle.batch(
    #     reader_decorator(paddle.dataset.flowers.train(use_xmap=True)),
    #     # reader_decorator(paddle.vision.datasets.Cifar10(mode='train', backend='cv2')),
    #     batch_size=batch_size,
    #     drop_last=True)
    #
    # train_loader = paddle.io.DataLoader.from_generator(
    #     capacity=32,
    #     use_double_buffer=True,
    #     iterable=True,
    #     return_list=True,
    #     use_multiprocess=True)

    # train_loader.set_sample_list_generator(train_reader)

    for eop in range(epoch):
        resnet.train()
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label = label - 1
            label.stop_gradient = True
            # 前向传播
            out = resnet(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
            dy_out = avg_loss.numpy()
            avg_loss.backward()
            #
            optimizer.step()
            resnet.clear_gradients()
            if batch_id % 5 == 0:
                print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (
                eop, batch_id, dy_out, acc_top1, acc_top5))


# 启动训练
if __name__ == '__main__':
    train_resnet()
``` 

运行方式：
```python
python3 -m paddle.distributed.launch --gpus=0,1,2,3 train_fleet_dygraph.py
```  

## （3）两个程序的运行结果

- DDP程序运行结果
```python
Epoch 7/100
loss: tensor(4.3571, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.1223, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3427, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4527, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4157, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4089, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4046, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4396, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3414, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3834, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2856, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4135, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4047, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4164, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3926, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2654, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2372, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4528, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2135, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2939, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4517, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.1285, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4465, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5095, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3783, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4235, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5945, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4677, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2633, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2308, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4648, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3647, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2926, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3164, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3357, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5612, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4285, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.6838, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3026, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3654, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5110, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2723, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4866, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.6240, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4451, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4438, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3717, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2463, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2591, device='cuda:0', grad_fn=<NllLossBackward>)
train Loss: 4.3819 Acc: 0.0439
Epoch 8/100
loss: tensor(4.4381, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4635, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3493, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2414, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2735, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5272, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.1665, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4553, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3675, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5145, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3524, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4013, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4338, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2520, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.1550, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2074, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2859, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5531, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.6937, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3387, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4027, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3194, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5824, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3335, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4477, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2184, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2377, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4388, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2814, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4338, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4741, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3977, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5670, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2727, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5136, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2349, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2736, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4575, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2906, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2580, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5590, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2204, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.5499, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3762, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4189, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.3005, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.4099, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.2168, device='cuda:0', grad_fn=<NllLossBackward>)
loss: tensor(4.1337, device='cuda:0', grad_fn=<NllLossBackward>)
train Loss: 4.3692 Acc: 0.0447
```


- paddlepaddle运行结果，附了第7、8个epoch
```python
launch train in GPU mode!
INFO 2022-04-02 19:30:53,752 launch_utils.py:510] Local start 4 processes. First process distributed environment info (Only For Debug): 
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        0                      |
    |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:60409               |
    |                     PADDLE_TRAINERS_NUM                        4                      |
    |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:41442,127.0.0.1:51971,127.0.0.1:52550|
    |                     PADDLE_RANK_IN_NODE                        0                      |
    |                 PADDLE_LOCAL_DEVICE_IDS                        0                      |
    |                 PADDLE_WORLD_DEVICE_IDS                     0,1,2,3                   |
    |                     FLAGS_selected_gpus                        0                      |
    |             FLAGS_selected_accelerators                        0                      |
    +=======================================================================================+

INFO 2022-04-02 19:30:53,752 launch_utils.py:514] details abouts PADDLE_TRAINER_ENDPOINTS can be found in log/endpoints.log, and detail running logs maybe found in log/workerlog.0
launch proc_id:12642 idx:0
launch proc_id:12645 idx:1
launch proc_id:12648 idx:2
launch proc_id:12651 idx:3

Epoch 7, batch 0] loss: 4.30898, acc1: 0.00000, acc5: 0.12500
[Epoch 7, batch 5] loss: 4.07224, acc1: 0.06250, acc5: 0.18750
[Epoch 7, batch 10] loss: 4.15420, acc1: 0.12500, acc5: 0.25000
[Epoch 7, batch 15] loss: 4.47023, acc1: 0.06250, acc5: 0.31250
[Epoch 7, batch 20] loss: 4.43729, acc1: 0.00000, acc5: 0.06250
[Epoch 7, batch 25] loss: 4.22468, acc1: 0.00000, acc5: 0.12500
[Epoch 7, batch 30] loss: 3.64519, acc1: 0.18750, acc5: 0.50000
[Epoch 7, batch 35] loss: 4.45402, acc1: 0.06250, acc5: 0.12500
[Epoch 7, batch 40] loss: 4.19298, acc1: 0.06250, acc5: 0.12500
[Epoch 7, batch 45] loss: 3.99954, acc1: 0.00000, acc5: 0.37500
[Epoch 7, batch 50] loss: 4.40063, acc1: 0.06250, acc5: 0.25000
[Epoch 7, batch 55] loss: 4.24690, acc1: 0.00000, acc5: 0.25000
[Epoch 7, batch 60] loss: 4.49993, acc1: 0.00000, acc5: 0.25000
[Epoch 7, batch 65] loss: 4.41674, acc1: 0.00000, acc5: 0.06250
[Epoch 7, batch 70] loss: 4.08913, acc1: 0.00000, acc5: 0.31250
[Epoch 7, batch 75] loss: 4.19635, acc1: 0.00000, acc5: 0.12500
[Epoch 7, batch 80] loss: 3.75817, acc1: 0.12500, acc5: 0.43750
[Epoch 7, batch 85] loss: 4.29419, acc1: 0.06250, acc5: 0.25000
[Epoch 7, batch 90] loss: 3.99528, acc1: 0.00000, acc5: 0.12500
[Epoch 7, batch 95] loss: 4.32901, acc1: 0.06250, acc5: 0.37500
[Epoch 8, batch 0] loss: 4.41281, acc1: 0.00000, acc5: 0.12500
[Epoch 8, batch 5] loss: 4.10598, acc1: 0.12500, acc5: 0.25000
[Epoch 8, batch 10] loss: 4.27404, acc1: 0.00000, acc5: 0.18750
[Epoch 8, batch 15] loss: 3.97948, acc1: 0.06250, acc5: 0.06250
[Epoch 8, batch 20] loss: 4.49495, acc1: 0.00000, acc5: 0.06250
[Epoch 8, batch 25] loss: 4.07579, acc1: 0.00000, acc5: 0.25000
[Epoch 8, batch 30] loss: 3.63573, acc1: 0.12500, acc5: 0.37500
[Epoch 8, batch 35] loss: 3.79878, acc1: 0.18750, acc5: 0.31250
[Epoch 8, batch 40] loss: 4.38518, acc1: 0.06250, acc5: 0.25000
[Epoch 8, batch 45] loss: 4.08105, acc1: 0.12500, acc5: 0.25000
[Epoch 8, batch 50] loss: 4.33881, acc1: 0.12500, acc5: 0.18750
```  

# 五、文档质量
感觉文档中部分代码的版本较老，比如1.3动态图完整代码中：
```python
from resnet_dygraph import ResNet 
```  
上述操作可以使用paddle内置API进行替换
```python
from paddle.vision.models import ResNet
```  
flower数据集在曙光平台不能通过API直接加载得到，需要手动下载。加载鲜花数据集的API也需要进行更新。可以更新为
```python
train_dataset = paddle.vision.datasets.Flowers(mode='train', transform=transform)
```  
对于很多错误，在文档中没有一个很好的提示。可以对常见报错进行一个汇总。


# 六、报错查错（问题汇总）
1、paddle在曙光超算上运行出现libamdhip64.4,，需要指定使用rocm4.0.1环境运行，曙光上rocm-2.9环境不能运行。
```python
module rm compiler/rocm/2.9 
module load compiler/rocm/4.0.1
```  

![image](https://user-images.githubusercontent.com/102226413/164144235-ce808c51-1712-4417-b9bf-99da6362b3f0.png)


2、无法下载flower数据集，需要手动加载数据集
按照文档旧API无法在曙光平台以及移动九天平台加载数据集，需要手动下载数据集。
且数据集的保存地址为一个缓存空间，用户在使用的时候可能找不到数据集，如/public/home/username/.cache/paddle/dataset目录。
而pytorch的加载数据集API会吧数据集加载到当前目录，方便了使用者。

![image](https://user-images.githubusercontent.com/102226413/164143513-24236f90-975d-47f1-a1db-7e18e6c94c9c.png)

![image](https://user-images.githubusercontent.com/102226413/164144065-2fea8ac3-dcf5-48ac-a4c7-05cace99c611.png)

3、安装完paddle后运行该程序会缺少常用两个库：opencv-python以及scipy。
安装方式：
```python
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
```  
4、鲜花数据集的label索引是从1开始的，不是从0开始的，需要手工减1。
```python
img, label = data
label = label - 1
```

5、未解决问题（无法在曙光上使用paddle 的问题）
![image](https://user-images.githubusercontent.com/102226413/164143125-70d0e4ff-46d7-4461-8cb0-72c14e98b8e0.png)

![image](https://user-images.githubusercontent.com/102226413/164143166-cde2793b-eb06-43a3-92d1-bfa68c2f1558.png)
