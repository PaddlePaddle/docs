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

## 1、从PaddlePaddle的各个API目录来对比分析

首先，基础操作类、组网类、Loss类、工具类、视觉类这五大类从映射Pytorch上看，在单机训练中可以满足训练及预测所需的API使用类别。

其次，在单机模型训练中对比了一下主要API的使用

| 名称    | PaddlePaddle                  | Pytorch                        |
| ------- | ----------------------------- | ------------------------------ |
| layer   | nn.Layer                      | nn.Module                      |
| 各种层  | nn.layer2D(即paddle使用大写D) | nn.layer2d(即Pytorch使用小写d) |
| flatten | nn.Flatten                    | var.view(var.size(0), -1)      |
| concat  | paddle.concat                 | torch.cat                      |
| optim   | paddle.optimizer              | torch.optim                    |

## 2、从具体API的参数设计差异作对比分析

通过训练和测试Paddle的动态图单机模型，就个人体验而言，对模型中常用到的一些API作简要分析，如paddle.to_tensor，paddle.save,paddle.load,paddle.nn.Conv2D , paddle.nn.Linear, paddle.nn.CrossEntropyLoss , paddle.io.DataLoader

### 2.1 基础操作类API

```python
#paddle.to_tensor
paddle.to_tensor(data,
                 dtype=None,
                 place=None,
                 stop_gradient=True)
#torch.tensor
torch.tensor(data,
             dtype=None,
             device=None,
             requires_grad=False,
             pin_memory=False)
```

在paddle.to_tensor中，stop_gradient表示是否阻断梯度传导，PyTorch的requires_grad表示是否不阻断梯度传导。

在torch.tensor中，pin_memeory表示是否使用锁页内存，而PaddlePaddle却无此参数。

------

```python
#paddle.load
paddle.load(path, **configs)

#torch.load
torch.load(f,
           map_location=None,
           pickle_module=pickle,
           **pickle_load_args)
```

在torch.load中， pickle_module  表示用于unpickling元数据和对象的模块，PaddlePaddle无此参数。  map_location  表示加载模型的位置，PaddlePaddle无此参数。 

在加载内容上，PyTorch可以加载torch.Tensor、torch.nn.Module、优化器等多个类型的数据。
PaddlePaddle只能加载paddle.nn.Layer、优化器这两个类型的数据，这方面Pytorch更优一些。

------

```python
#paddle.save
paddle.save(obj, path, pickle_protocol=2)

#torch.save
torch.save(obj,
           f,
           pickle_module=pickle,
           pickle_protocol=2)
```

在paddle.save中， path表示存储的路径，这一点比 PyTorch 的f更为清晰一些。

在torch.save中， pickle_module  表示用于pickling元数据和对象的模块，PaddlePaddle无此参数。 

还有在存储内容上，跟paddle.load情况类似，PaddlePaddle只能存储paddle.nn.Layer、优化器这两个类型的数据，个人觉得这方面PaddlePaddle有待加强。

------

### 2.2 组网类API

```python
#paddle.nn.Conv2D
paddle.nn.Conv2D(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW')
#torch.nn.Conv2d
torch.nn.Conv2d(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros')
```

在paddle.nn.Conv2D中，PaddlePaddle支持NCHW和NHWC两种格式的输入（通过data_format设置）。而PyTorch只支持NCHW的输入，这一点PaddlePaddle更优一些。

------

```python
#paddle.nn.Linear
paddle.nn.Linear(in_features, out_features, weight_attr=None, bias_attr=None, name=None)
#torch.nn.Linear
torch.nn.Linear(in_features, out_features, bias=True)
```

在paddle.nn.Linear中，weight_attr/bias_attr默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，而PyTorch的bias默认为True，表示使用可更新的偏置参数。需要注意的是在PaddlePaddle中，当bias_attr设置为bool类型与PyTorch的作用一致。

------

### 2.3 Loss类API

```python
#paddle.nn.CrossEntropyLoss
paddle.nn.CrossEntropyLoss(weight=None,
                           ignore_index=-100,
                           reduction='mean',
                           soft_label=False,
                           axis=-1,
                           use_softmax=True,
                           name=None)
#torch.nn.CrossEntropyLoss
torch.nn.CrossEntropyLoss(weight=None,
                          size_average=None,
                          ignore_index=-100,
                          reduce=None,
                          reduction='mean')
```

在paddle.nn.CrossEntropyLoss中， use_softmax  表示在使用交叉熵之前是否计算softmax，PyTorch无此参数；soft_label指明label是否为软标签，PyTorch无此参数；而axis表示进行softmax计算的维度索引，PyTorch无此参数。 在这个API中，个人感觉PaddlePaddle的表现优于PyTorch。

------

### 2.4 工具类API

```python
#paddle.io.DataLoader
paddle.io.DataLoader(dataset,
                     feed_list=None,
                     places=None,
                     return_list=False,
                     batch_sampler=None,
                     batch_size=1,
                     shuffle=False,
                     drop_last=False,
                     collate_fn=None,
                     num_workers=0,
                     use_buffer_reader=True,
                     use_shared_memory=False,
                     timeout=0,
                     worker_init_fn=None)
#torch.utils.data.DataLoader
torch.utils.data.DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=None,
                            batch_sampler=None,
                            num_workers=0,
                            collate_fn=None,
                            pin_memory=False,
                            drop_last=False,
                            timeout=0,
                            worker_init_fn=None,
                            multiprocessing_context=None,
                            generator=None,
                            prefetch_factor=2,
                            persistent_workers=False)
```

在paddle.io.DataLoader中， feed_list 表示feed变量列表，PyTorch无此参数。  use_shared_memory  表示是否使用共享内存来提升子进程将数据放入进程间队列的速度，PyTorch无此参数。 

在torch.utils.data.DataLoader中，prefetch_factor  表示每个worker预先加载的数据数量，PaddlePaddle无此参数；还有就是PyTorch可通过设置sampler自定义数据采集器，PaddlePaddle只能自定义一个DataLoader来实现该功能，会有些繁琐。总的来说，这部分Pytorch的体验更好一些。

------

​		从整体的API使用上，感觉paddle升级后的 paddle.xxx  （例如：paddle.device  paddle.nn  paddle.vision ）比之前的 padddle.fluid.xxx 好用很多，还有就是新增加的高层API个人比较喜欢，一是对初学者比较友好、易用，二是对于开发者可以节省代码量，更简洁直观一些，在（六、动态图单机训练）中进行了代码展示和对比分析。

与Pytorch相比，基础API的结构和调用没有太大区别，但是在速度上，paddle的基础API会更快一点，如果是利用了paddle高层API，速度会快很多，在同样5次epoch的情况下，LeNet训练高层API用38s左右,基础API得用将近两分钟，所以用高层API能减少大约三分之二的训练时间。

总体来说，使用像paddle.Model、paddle.vision这样的高级API进行封装调用，使用体验比较好，个人感觉在以后深度学习模型普遍使用时，高层API会更受欢迎，也会成为模型训练测试中更为流行的一种方法。



# 四、Tensor 索引

在了解Paddle的Tensor索引和其切片规则以及逻辑相关函数重写规则等内容后，结合指南内容（ https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html#id1 ）和模型训练过程中的Tensor索引使用，共有以下几点体验总结：

**一、Paddle可以使用静态数组索引；不可以使用tensor索引。**
示例：广播 (broadcasting)
1.每个张量至少为一维张量

2.从后往前比较张量的形状，当前维度的大小要么相等，要么其中一个等于一，要么其中一个不存在

```python
import paddle
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 4))
# 两个张量 形状一致，可以广播
z = x + y
print(z.shape)
# [2, 3, 4]
 
x = paddle.ones((2, 3, 1, 5))
y = paddle.ones((3, 4, 1))
# 从后向前依次比较：
# 第一次：y的维度大小是1
# 第二次：x的维度大小是1
# 第三次：x和y的维度大小相等
# 第四次：y的维度不存在
# 所以 x和y是可以广播的
z = x + y
print(z.shape)
# [2, 3, 4, 5]
 
# 相反
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 6))
# 此时x和y是不可广播的，因为第一次比较 4不等于6
# z = x + y
# InvalidArgumentError: Broadcast dimension mismatch.
```

**二、两个张量进行广播语义后的结果张量的形状计算规则如下：**

1.如果两个张量的形状的长度不一致，那么需要在较小形状长度的矩阵向前添加1，直到两个张量的形状长度相等。

2.保证两个张量形状相等之后，每个维度上的结果维度就是当前维度上较大的那个。

```python
import paddle
 
x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 1))
z = x + y
print(z.shape)
# z的形状: [2,3,4]
 

x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 2))
# z = x + y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.
```

**三、Paddle 目前支持的Tensor索引规则：**

**Paddle 目前支持的Tensor索引状态：**

1、基于 0-n 的下标进⾏索引
2、如果下标为负数，则从尾部开始
3、通过冒号 : 分隔切⽚参数 start:stop:step 来进⾏切⽚操作，其中 start、stop、step 均可缺省

示例1：索引

```python
ndim_1_tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("最初的Tensor: ", ndim_1_tensor.numpy())
print("取⾸端元素:", ndim_1_tensor[0].numpy())
print("取末端元素:", ndim_1_tensor[-1].numpy())
print("取所有元素:", ndim_1_tensor[:].numpy())
print("取索引3之前的所有元素:", ndim_1_tensor[:3].numpy())
print("取从索引6开始的所有元素:", ndim_1_tensor[6:].numpy())
print("取从索引3开始到索引6之前的所有元素:", ndim_1_tensor[3:6].numpy())
print("间隔3取所有元素:", ndim_1_tensor[::3].numpy())
print("逆序取所有元素:", ndim_1_tensor[::-1].numpy())
```

部分运⾏结果如下：

```python
First element: [0]
Last element: [8]
All element: [0 1 2 3 4 5 6 7 8]
Before 3: [0 1 2]
From 6 to the end: [6 7 8]
From 3 to 6: [3 4 5]
Interval of 3: [0 3 6]
Reverse: [8 7 6 5 4 3 2 1 0]
```

**Paddle 目前不支持的Tensor索引状态：**

示例1：不能维度直接赋值

```python
#报错：
TypeError: 'paddle.fluid.libpaddle.VarBase' object does not support item assignment
#代码如下：    
# pytorch code
Pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    
# paddlepaddle code
pred_boxes = paddle.layers.concat([
    pred_ctr_x - 0.5 * pred_w,
    pred_ctr_y - 0.5 * pred_h,
    pred_ctr_x + 0.5 * pred_w,
    pred_ctr_y + 0.5 * pred_h
])


#维度报错:

too many indices (3) for tensor of dimension 2
#代码如下： 
# pytorch code
bbox_x[bind, :, np.newaxis ] 
# paddlepaddle code
paddle.layers.reshape(bbox_x[bind, :], [1, -1, 1])
```

示例2： tensor的值不能直接利用

报错：paddlepaddle中的value不能直接拿出来用。

```python
TypeError: The type of 'shape' in reshape must be list[int] or tuple(int) in
 Dygraph mode, but received <class 'list'>, which contains Variable.
#错误代码：其中stack_size, feat_size 为 tensor。

#代码如下： 
# paddlepaddle code
shift_x1 = paddle.layers.reshape(paddle.dygraph.to_variable(shift_x1), [1, stack_size, feat_size[1]])

```

改进加入

```python
# paddlepaddle code
stack_size = stack_size.numpy()
feat_size = feat_size.numpy()
```

**四、Tensor 索引整体体验**

感觉在通过索引或切片修改 Tensor 的整体过程有些冗余，稳定性也会下降。虽然使用指南里说明了修改会导致原值不会被保存，可能会给梯度计算引入风险 ，但是在这点上个人感觉Pytorch的体验要好于Paddle。

总的来说，在模型训练中利用Tensor加载数据集等操作上 Pytorch与 Paddle的体验并没有太大区别，但整体的感觉Pytorch的Tensor 索引更好一些，个人感觉Paddle在修改 Tensor的部分上可以增加一些文档说明。

**文档序号错误小提醒：**

 https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/update_cn.html#tensor 中的“使用Tensor概念表示数据”下的序号应为1、2、；文档中为两个1、1、。

# 五、NumPy兼容性分析及对比

NumPy在Paddle的体验，感觉和Pytorch的体验并无区别，但是在阅读使用文档时的体验感较好，内容叙述很详细 （文档链接：https://www.paddlepaddle.org.cn/tutorials/projectdetail/3466356 ）

**1、关于numpy API的重写**

在Paddle动态图单机训练中，所有与组网相关的 numpy 操作都必须用 paddle 的 API 重新实现 ，这一点个人认为需要注意，因为在习惯使用Pytorch代码逻辑时，转为PaddlePaddle容易出错，下面举例说明：

```python
#下述样例需要将 forward 中的所有的 numpy 操作都转为 Paddle API：
def forward(self, x):
    out = self.linear(x)  # [bs, 3]

    # 以下将 tensor 转为了 numpy 进行一系列操作
    x_data = x.numpy().astype('float32')  # [bs, 10]
    weight = np.random.randn([10,3])
    mask = paddle.to_tensor(x_data * weight)  # 此处又转回了 Tensor

    out = out * mask
    return out
```

注：由于在动态图模型代码中的 numpy 相关的操作不可以转为静态图，所以在动态图单机训练时候，只要与组网相关的 numpy 操作用 paddle 的 API 重新实现即可，所以在numpy API的重写部分，记住以上区别可以防止 Segment Fault 等错误的产生。

**2、关于Tensor 操作的支持**

在动态图单机训练中，感觉Paddle的Tensor高度兼容Numpy数组（array），发现增加了很多适用于深度学习任务的参数和方法，如反向计算梯度，更灵活的指定运行硬件 ，还有就是Paddle的Tensor可以与Numpy的数组方便的互转 ，比如以下代码展示：

```python
import paddle
import numpy as np

tensor_to_convert = paddle.to_tensor([1.,2.])

#通过 Tensor.numpy() 方法，将 Tensor 转化为 Numpy数组
tensor_to_convert.numpy()

#通过paddle.to_tensor() 方法，将 Numpy数组 转化为 Tensor
tensor_temp = paddle.to_tensor(np.array([1.0, 2.0]))
```

**3、numpy与tensor的转换补充**

numpy操作多样, 简单. 但网络前向只能是tensor类型, 各有优势, 所以需要相互转换补充.

```python
# convert Tensor x of torch to array y of  numpy: 
y = x.numpy();
 
# convert array x of  numpy to Tensor y of torch: 
y = torch.from_numpy(x)
 
# 先将数据转换成Tensor, 再使用CUDA函数来将Tensor移动到GPU上加速
如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
x_np = x.data.numpy()
 
# 改为：
 
x_np = x.data.cpu().numpy()
 
# 或者兼容上面两者的方式
x_np = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()

```

**整体体验：**

感觉对于刚使用Paddle的新手，这部分需要注意的就是 Paddle的Tensor虽然可以与Numpy的数组方便的互相转换 ，但是有两个场景优先使用Paddle的Tensor 比较好:

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

PaddlePaddle有丰富的api可以实现各种调参trick，像dropout，batchnormalization，groupnormalization，l2regularization, lr decay等等都可以很轻松地实现。
另外数据增强则推荐使用PIL库，尝试各种技巧不一定每次都能让模型准确度提升，毕竟训练神经网络是一个多参数配合的过程，只有练得多了才更容易找到最佳的方向。
根据查阅资料，现总结以下几点：
1、 cuDNN操作的选择
在 use_cudnn=True 时，框架底层调用的是cuDNN中的卷积操作。
通常cuDNN库提供的操作具有很好的性能表现，其性能明显优于Paddle原生的CUDA实现，比如 conv2d 。
但是cuDNN中有些操作的性能较差，比如： conv2d_transpose 在 batch_size=1 时、pool2d 在 global_pooling=True 时等，
这些情况下，cuDNN实现的性能差于Paddle的CUDA实现，建议手动设置 use_cudnn=False 。

2、使用融合功能的API
用户网络配置中使用融合功能的API，通常能取得更好的计算性能。
例如softmax_with_cross_entropy通常会比softmax cross_entropy分开用好

3、优化数据准备速度的方法
为降低训练的整体时间，建议用户使用异步数据读取的方式，并开启 use_double_buffer（默认开）。此外，用户可根据模型的实际情况设置数据队列的大小（capacity）。
如果数据准备的时间大于模型执行的时间，或者出现了数据队列为空的情况，这时候需要考虑对Python的用户reader进行加速。常用的方法为：使用Python多进程准备数据。
Python端的数据预处理，都是使用CPU完成。如果Paddle提供了相应功能的API，可将这部分预处理功能写到模型配置中，如此Paddle就可以使用GPU来完成该预处理功能，
这样也可以减轻CPU预处理数据的负担，提升总体训练速度。

4、显存优化策略
GC（Garbage Collection）的原理是在网络运行阶段及时释放无用变量的显存空间，达到节省显存的目的。GC适用于使用Executor，ParallelExecutor做模型训练/预测的场合。
由于原生的CUDA系统调用 cudaMalloc 和 cudaFree 均是同步操作，非常耗时。因此与许多框架类似，PaddlePaddle采用了显存预分配的策略加速显存分配。

5、Inplace策略
原理是Op的输出复用Op输入的显存空间。
由于目前设计上的一些问题，在开启Inplace策略后，必须保证后续exe.run中fetch_list的变量是persistable的。
fetch_list：结果获取表，训练时一般有loss等。
推荐的最佳显存优化策略为：
开启Inplace策略：设置 build_strategy.enable_inplace = True ，并设置fetch_list中的 var.persistable = True 。

PaddlePaddle在深度学习框架方面，已经覆盖了搜索、图像识别、语音语义识别理解、情感分析、机器翻译、用户画像推荐等多领域的业务和技术。
基于动态图实现的AlexNet代码如下:

```python
class ConvPoolLayer(nn.Layer):
  '''卷积+池化'''
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride,
                 padding,
                 stdv,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvPoolLayer, self).__init__()
        self.relu = ReLU() if act == "relu" else None

  self._conv = Conv2D(#返回一个由所有子层组成的列表。
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(
                name=name + "_weights", initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(
                name=name + "_offset", initializer=Uniform(-stdv, stdv)))
        self._pool = MaxPool2D(kernel_size=3, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv(inputs)
        if self.relu is not None:
            x = self.relu(x)
        x = self._pool(x)
        return x
```

PaddlePaddle搭建cnn网络以及进行模型训练预测，可以说PaddlePaddle搭建训练pipeline还是比较方便的。

# 八、报错汇总

1、 DataLoader报错问题 ：

```python
SystemError: (Fatal) Blocking queue is killed because the data reader raises an exception.
[Hint: Expected killed_ != true, but received killed_:1 == true:1.] (at /paddle/paddle/fluid/operators/reader/blocking_queue.h:158)
```

原因分析：由于PaddlePaddle和Pytorch两个框架在这部分并无区别，Paddle读取数据在这主要用到两个类：paddle.io.Dataset和paddle.io.DataLoader，所以查看源代码后发现在Dataset类中的__getitem__(self, idx)返回的数据不是numpy.ndarray类型

解决方案：

 在__getitem__（）函数里添加一行代码：image = paddle.to_tensor(image)

```python
# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        image = paddle.to_tensor(image) # 添加这行代码
        return image, label

    def __len__(self):
        return self.num_samples
```

注：还有一种情况是Dataset类的__getitem__(self, idx)返回的数据为字典（Dict） 类型也会报同样的错误，这时可把return改为return {'input': image, 'lb': label}



2、使用自己数据集的图像大小不合适而报错

```python
ERROR:root:DataLoader reader thread raised an exception!
Traceback (most recent call last):
File “/home/disk0/zw/workspace/PaddleOCR/test/load_data.py”, line 38, in
for idx,batch in enumerate(data_loader):
File “/home/disk0/wy/anaconda3/envs/paddle/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py”, line 788, in next
data = self.reader.read_next_var_list()
SystemError: (Fatal) Blocking queue is killed because the data reader raises an exception.
[Hint: Expected killed != true, but received killed_:1 == true:1.] (at /paddle/paddle/fluid/operators/reader/blocking_queue.h:158)
```

解决方案：由于自己数据集中有部分图片超过了默认shape的[3, 32, 320]，图片宽度大于了320，所以直接删除或调大shape尺寸即可

注:使用公开数据集时不会出现此问题



3、使用paddle.reshape时出现错误

```python
ValueError: (InvalidArgument) The 'shape' in ReshapeOp is invalid. The input tensor X'size must be equal to the capacity of 'shape'. But received X's shape = [64, 50, 4, 4], X's size = 51200, 'shape' is [1, 800], the capacity of 'shape' is 800.
  [Hint: Expected capacity == in_size, but received capacity:800 != in_size:51200.] (at C:\home\workspace\Paddle_release\paddle/fluid/operators/reshape_op.cc:224)
  [operator < reshape2 > error]
```

解决方案：在使用forward函数实现MNIST网络的执行逻辑时，忽略了self.pool_2_shape变量的大小，重新设置paddle.reshape为x = paddle.reshape(x, shape=[-1, self.pool_2_shape])即可



4、tensor的值不能直接利用

报错：paddlepaddle中的value不能直接拿出来用。

```python
TypeError: The type of 'shape' in reshape must be list[int] or tuple(int) in
 Dygraph mode, but received <class 'list'>, which contains Variable.
#错误代码：其中stack_size, feat_size 为 tensor。

#改进加入
# paddlepaddle code
stack_size = stack_size.numpy()
feat_size = feat_size.numpy()
```



5、Paddle加载数据集报错，无法下载MNIST数据集，需要手动进行下载，（使用了多台电脑测试，均会出现此情况）

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
