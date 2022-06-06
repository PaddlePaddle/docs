# 自动混合精度训练

一般情况下，训练深度学习模型时使用的数据类型为单精度（FP32）。2018年，百度与NVIDIA联合发表论文：[MIXED PRECISION TRAINING](https://arxiv.org/pdf/1710.03740.pdf)，提出了混合精度训练的方法。混合精度训练是指在训练过程中，同时使用单精度（FP32）和半精度（FP16），其目的是相较于使用单精度（FP32）训练模型，在保持精度持平的条件下，能够加速训练。本文将介绍如何使用飞桨框架，实现自动混合精度训练。  

## 一、概述：

### 1.1 半精度浮点类型 FP16

首先介绍半精度（FP16）。如图1所示，半精度（FP16）是一种相对较新的浮点类型，在计算机中使用2字节（16位）存储。在IEEE 754-2008标准中，它亦被称作binary16。与计算中常用的单精度（FP32）和双精度（FP64）类型相比，FP16更适于在精度要求不高的场景中使用。

<figure align="center">
    <img src="https://paddleweb-static.bj.bcebos.com/images/fp16.png" width="600" alt='missing'/>
    <figcaption><center>图 1. 半精度和单精度数据示意图</center></figcaption>
</figure>

### 1.2 NVIDIA GPU的FP16算力

在使用相同的超参数下，混合精度训练使用半精度浮点（FP16）和单精度（FP32）浮点即可达到与使用纯单精度训练相同的准确率，并可加速模型的训练速度，这主要得益于英伟达从Volta架构开始推出的Tensor Core技术。在使用FP16计算时具有如下特点：
- FP16可降低一半的内存带宽和存储需求，这使得在相同的硬件条件下研究人员可使用更大更复杂的模型以及更大的batch size大小。
- FP16可以充分利用英伟达Volta、Turing、Ampere架构GPU提供的Tensor Cores技术。在相同的GPU硬件上，Tensor Cores的FP16计算吞吐量是FP32的8倍。

通过``nvidia-smi``指令可帮助查看显卡架构信息。此外如果已开启amp训练，Paddle会自动帮助检测硬件环境是否符合上述硬件条件，如不符合，则将提供类似如下的警告信息：``UserWarning: AMP only support NVIDIA GPU with Compute Capability 7.0 or higher, current GPU is: Tesla K40m, with Compute Capability: 3.5.``。

## 二、使用飞桨框架实现自动混合精度

使用飞桨框架提供的API，能够在原始训练代码基础上快速开启自动混合精度训练（Automatic Mixed Precision，AMP），即在相关OP的计算中，根据一定的规则，自动选择FP16或FP32计算。

依据FP16在模型中的使用程度划分，飞桨的AMP分为两个等级：
- level = ’O1‘：采用黑白名单策略进行混合精度训练，黑名单中的OP将采用FP32计算，白名单中的OP将采用FP16计算，训练过程中框架会自动将白名单OP的输入参数数据类型从FP32转为FP16，使用FP16与FP32进行计算的OP列表可见该[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/Overview_cn.html)。对于不在黑白名单中的OP，框架会依据该OP的全部输入数据类型进行推断，当全部输入均为FP16时，OP将直接采用FP16计算，否则采用FP32计算。
- level = ’O2‘：该模式采用了比O1更为激进的策略，除了框架不支持FP16计算的OP，其他全部采用FP16计算，框架会预先将网络参数从FP32转换为FP16，相比O1，训练过程中无需做FP32转为FP16的操作，训练速度会有更明显的提升，但可能会存在精度问题，为此，框架提供了自定义黑名单，用户可通过该名单指定一些存在精度问题的OP执行FP32运算。

飞桨动态图与静态图均为用户提供了便捷的API用于开启上述混合精度训练，下面以具体的训练代码为例，来了解如何使用飞桨框架实现混合精度训练。

### 2.1 动态图混合精度训练

飞桨动态图提供了一系列便捷的API用于实现混合精度训练：``paddle.amp.auto_cast``、``paddle.amp.GradScaler``、``paddle.amp.decorate``。其中：

1）``paddle.amp.auto_cast``：用于创建混合精度训练的上下文环境，来支持动态图模式下执行的算子的自动混合精度策略

2）``paddle.amp.GradScaler``：GradScaler用于动态图模式下的"自动混合精度"的训练，可控制loss的缩放比例，有助于避免浮点数溢出的问题（注：可选，若使用FP16数据类型也可保证参数不会溢出则无需调用该接口）

3）``paddle.amp.decorate``：用于``O2``模式下，将神经网络参数数据类型改写为FP16（除BatchNorm和LayerNorm）（注：``O1``模式下，该接口无作用，无需调用）

<a name="2.1.1"></a>
#### 2.1.1 动态图FP32训练

1）构建一个简单的网络：用于对比使用FP32训练与使用混合精度训练的训练速度，为了充分体现混合精度训练所带来的性能提升，构建一个由九层 ``Linear`` 组成网络，每层``Linear``网络由``matmul``算子及``add``算子组成，``matmul``算子是可以得到加速的算子。

```python
import time
import paddle
import paddle.nn as nn
import numpy

paddle.seed(100)
numpy.random.seed(100)
place = paddle.CUDAPlace(0)

class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(input_size, output_size) for i in range(9)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i](x)
        return x
```

2）设置训练的相关参数及训练数据：这里为了能有效的看出混合精度训练对于训练速度的提升，将 ``input_size`` 与 ``output_size`` 的值设为较大的值，为了使用GPU提供的``Tensor Core``性能，还需将``batch_size``设置为 8 的倍数（基于混合精度训练的性能优化方法见：<a href="#三">三、混合精度训练性能优化</a>）。

```python
epochs = 2
input_size = 8192   # 设为较大的值
output_size = 8192  # 设为较大的值
batch_size = 2048    # batch_size 为8的倍数
nums_batch = 10

from paddle.io import Dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        data = numpy.random.random([input_size]).astype('float32')
        label = numpy.random.random([output_size]).astype('float32')
        return data, label

    def __len__(self):
        return self.num_samples

dataset = RandomDataset(nums_batch * batch_size)
loader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

```

注：如果该示例代码在您的机器上显示显存不足相关的错误，请尝试将``input_size``、``output_size``、``batch_size``调小。

3）使用动态图FP32训练：

```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)  # 定义模型
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

train_time = 0 # 总训练时长
for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time() # 开始训练时刻
        label._to(place)
        # 前向计算
        output = model(data)
        loss = mse(output, label)
        # 反向传播
        loss.backward()
        # 训练模型
        optimizer.step()
        optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time # 记录总训练时长

print("loss:", train_loss)
print("使用FP32模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))
```

    loss: [0.6486028]
    使用FP32模式耗时:0.529 sec


#### 2.1.2 动态图AMP-O1训练：

在飞桨框架中，使用AMP-O1训练训练，需要在FP32代码的基础上添加两处逻辑：

- 逻辑1：使用 ``paddle.amp.auto_cast`` 创建AMP上下文环境，在该上下文内，框架会根据框架预设的黑白名单，自动确定每个OP的输入数据类型（FP16或FP32）
- 逻辑2：可选，使用 ``paddle.amp.GradScaler`` 用于缩放 ``loss`` 比例，避免浮点数下溢（下溢：由于FP16的有效数据表示范围为[6.10×10−5,65504]相比FP32更小，模型中较小的参数的梯度值可能无法在FP16中表示）


```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)  # 定义模型
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

# 逻辑2：可选，定义 GradScaler，用于缩放loss比例，避免浮点数溢出
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

train_time = 0 # 总训练时长
for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time() # 开始训练时刻

        label._to(place)
        # 逻辑1：创建AMP-O1上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
            output = model(data)
            loss = mse(output, label)
        # 逻辑2：使用 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()
        # 训练模型
        scaler.step(optimizer)       # 更新参数
        scaler.update()              # 更新用于 loss 缩放的比例因子
        optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time # 记录总训练时长

print("loss:", train_loss)
print("使用AMP-O1模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))

上述代码中，在 ``paddle.amp.auto_cast`` 语境下的 ``model`` 及 ``mse`` 均以 AMP-O1 的逻辑执行，由于 ``elementwise_add`` 加入了白名单，因为 Linear 层的 ``matmul`` 算子及 ``add`` 算子均执行FP16计算。

```

    loss: [0.6486219]
    使用AMP-O1模式耗时:0.118 sec

- ``paddle.amp.GradScaler``使用介绍见[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/GradScaler_cn.html)
- ``paddle.amp.auto_cast``使用介绍见[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/auto_cast_cn.html)

#### 2.1.3 动态图AMP-O2训练：

O2模式采用了比O1更为激进的策略，除了框架不支持FP16计算的OP，其他全部采用FP16计算，需要在训练前将网络参数从FP32转为FP16，在FP32代码的基础上添加三处逻辑：

- 逻辑1：使用 ``paddle.amp.decorate`` 将网络参数从FP32转换为FP16
- 逻辑2：使用 ``paddle.amp.auto_cast`` 创建AMP上下文环境，在该上下文内，框架会将所有支持FP16的OP都采用FP16进行计算（自定义的黑名单除外），其他OP采用FP32进行计算
- 逻辑3：可选，使用 ``paddle.amp.GradScaler`` 用于缩放 ``loss`` 比例，避免浮点数下溢（下溢：由于FP16的有效数据表示范围为[6.10×10−5,65504]相比FP32更小，模型中较小的参数的梯度值可能无法在FP16中表示）

```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)  # 定义模型
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

# 逻辑1：在level=’O2‘模式下，将网络参数从FP32转换为FP16
model = paddle.amp.decorate(models=model, level='O2')

# 逻辑3：可选，定义 GradScaler，用于缩放loss比例，避免浮点数溢出
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

train_time = 0 # 总训练时长

for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time()

        label._to(place)
        # 逻辑2：创建AMP上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast(level='O2'):
            output = model(data)
            loss = mse(output, label)
        # Step4：使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()
        # 训练模型
        scaler.step(optimizer)       # 更新参数
        scaler.update()              # 更新用于 loss 缩放的比例因子
        optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time

print("loss=", train_loss)
print("使用AMP-O2模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```
上述代码中，通过``paddle.amp.decorate``修饰，``model`` 中全部的网络参数将从FP32转换为FP16，在 ``paddle.amp.auto_cast`` 语境下的 ``model`` 及 ``mse`` 均以 AMP-O2 的逻辑执行。

    loss= [0.6743]
    使用AMP-O2模式耗时:0.102 sec

- ``paddle.amp.decorate``使用介绍见[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/decorate_cn.html)

动态图FP32及AMP训练的精度速度对比如下表所示：

|test | FP32 | AMP-O1 | AMP-O2 |
|:---:|:---:|:---:|:---:|
|耗时 | 0.529s | 0.118s | 0.102s |
|loss | 0.6486028 | 0.6486219 | 0.6743 |

从上表统计结果可以看出，使用自动混合精度训练: O1模式训练速度提升约为4.5倍，O2模式训练速度提升约为5.2倍。如需更多使用混合精度训练的示例，请参考飞桨模型库： [paddlepaddle/models](https://github.com/PaddlePaddle/models)。

注：受机器环境影响，上述示例代码的训练耗时统计可能存在差异，该影响主要包括：GPU利用率、CPU利用率的等，测试机器配置如下：

|Device | MEM Clocks | SM Clocks | Running with CPU Clocks |
|:---:|:---:|:---:|:---:|
|Tesla V100 SXM2 16GB |  877 MHz   | 1530 MHz |   1000 - 2400 MHz  |

### 2.2 静态图混合精度训练

飞桨静态图提供了一系列便捷的API用于实现混合精度训练：``paddle.static.amp.decorate``、``paddle.static.amp.fp16_guard``。

#### 2.2.1 静态图FP32训练

采用与2.1.1节动态图训练相同的网络结构：<a href="#2.1.1">2.1.1 动态图FP32训练</a>，静态图网络初始化如下：

```python
paddle.enable_static()
place = paddle.CUDAPlace(0)
main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()

model = SimpleNet(input_size, output_size)
mse_loss = paddle.nn.MSELoss()

```

静态图训练代码如下：

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())
optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

train_time = 0 # 总训练时长
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time() # 开始训练时刻

        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)

        train_time += time.time() - start_time # 记录总训练时长

print("loss:", train_loss)
print("使用FP32模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6486028], dtype=float32)]
    使用FP32模式耗时:0.531 sec

#### 2.2.2 静态图AMP-O1训练

静态图通过``paddle.static.amp.decorate``对优化器进行封装、通过`paddle.static.amp.CustomOpLists`定义黑白名单，即可开启混合精度训练，示例代码如下：

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# 1) 通过 `CustomOpLists` 自定义黑白名单
amp_list = paddle.static.amp.CustomOpLists(custom_white_list=['elementwise_add'])

# 2）通过 `decorate` 对优化器进行封装：
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    amp_lists=amp_list,
    init_loss_scaling=128.0,
    use_dynamic_loss_scaling=True)

optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

train_time = 0 # 总训练时长
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time() # 开始训练时刻

        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)

        train_time += time.time() - start_time # 记录总训练时长

print("loss:", train_loss)
print("使用AMP-O1模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6486222], dtype=float32)]
    使用AMP-O1模式耗时:0.117 sec

`paddle.static.amp.CustomOpLists`用于自定义黑白名单，黑名单op执行FP32 kernel、白名单op执行FP16 kernel。

#### 2.2.3 静态图AMP-O2训练

静态图开启AMP-O2有两种方式：

- 使用``paddle.static.amp.fp16_guard``控制FP16应用的范围，在该语境下的所有op将执行FP16计算，其他op执行FP32计算；

- 不使用``paddle.static.amp.fp16_guard``控制FP16应用的范围，网络的全部op执行FP16计算（除去自定义黑名单的op、不支持FP16计算的op）

1）设置``paddle.static.amp.decorate``的参数``use_pure_fp16``为 True，同时设置参数``use_fp16_guard``为 False

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# 1）通过 `decorate` 对优化器进行封装：
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    init_loss_scaling=128.0,
    use_dynamic_loss_scaling=True,
    use_pure_fp16=True,
    use_fp16_guard=False)

optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

# 2) 利用 `amp_init` 将网络的 FP32 参数转换 FP16 参数.
optimizer.amp_init(place, scope=paddle.static.global_scope())

train_time = 0 # 总训练时长
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time() # 开始训练时刻

        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)

        train_time += time.time() - start_time # 记录总训练时长

print("loss:", train_loss)
print("使用AMP-O2模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6743], dtype=float16)]
    使用AMP-O2模式耗时:0.098 sec

注：在AMP-O2模式下，网络参数将从FP32转为FP16，输入数据需要相应输入FP16类型数据，因此需要将``class RandomDataset``中初始化的数据类型设置为``float16``。

2）设置``paddle.static.amp.decorate``的参数``use_pure_fp16``为 True，同时设置参数``use_fp16_guard``为True，通过``paddle.static.amp.fp16_guard``控制使用FP16的计算范围

在模型定义的代码中加入`fp16_guard`控制部分网络执行在FP16下：

```python
class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(input_size, output_size) for i in range(9)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            if i > 0:
                with paddle.static.amp.fp16_guard():
                    x = self.linears[i](x)
            else:
                x = self.linears[i](x)
        return x
```

该模式下的训练代码如下：

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# 1）通过 `decorate` 对优化器进行封装：
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    init_loss_scaling=128.0,
    use_dynamic_loss_scaling=True,
    use_pure_fp16=True,
    use_fp16_guard=True)

optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

# 2) 利用 `amp_init` 将网络的 FP32 参数转换 FP16 参数.
optimizer.amp_init(place, scope=paddle.static.global_scope())

train_time = 0 # 总训练时长
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time() # 开始训练时刻

        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)

        train_time += time.time() - start_time # 记录总训练时长

print("loss:", train_loss)
print("使用AMP-O2模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6691731], dtype=float32)]
    使用AMP-O2模式耗时:0.140 sec


静态图FP32及AMP训练的精度速度对比如下表所示：

|test | FP32 | AMP-O1 | AMP-O2 |
|:---:|:---:|:---:|:---:|
|耗时 | 0.531s | 0.117s | 0.098s |
|loss | 0.6486028 | 0.6486222 | 0.6743 |

从上表统计结果可以看出，使用自动混合精度训练: O1模式训练速度提升约为4.5倍，O2模式训练速度提升约为5.4倍。如需更多使用混合精度训练的示例，请参考飞桨模型库： [paddlepaddle/models](https://github.com/PaddlePaddle/models)。

<a name="三"></a>
## 三、混合精度训练使用技巧及约束

飞桨AMP提升模型训练性能的根本原因是：利用 Tensor Core 来加速 FP16 下的``matmul``和``conv``运算，为了获得最佳的加速效果，Tensor Core 对矩阵乘和卷积运算有一定的使用约束，约束如下：

### 3.1 矩阵乘使用建议

通用矩阵乘 (GEMM) 定义为：``C = A * B + C``，其中：
- A 维度为：M x K
- B 维度为：K x N
- C 维度为：M x N

矩阵乘使用建议如下：
- 根据Tensor Core使用建议，当矩阵维数 M、N、K 是8（A100架构GPU为16）的倍数时（FP16数据下），性能最优。

### 3.2 卷积计算使用建议

卷积计算定义为：``NKPQ = NCHW * KCRS``，其中：
- N 代表：batch size
- K 代表：输出数据的通道数
- P 代表：输出数据的高度
- Q 代表：输出数据的宽度
- C 代表：输入数据的通道数
- H 代表：输入数据的高度
- W 代表：输入数据的宽度
- R 代表：滤波器的高度
- S 代表：滤波器的宽度

卷积计算使用建议如下：
- 输入/输出数据的通道数（C/K）可以被8整除（FP16），（cudnn7.6.3及以上的版本，如果不是8的倍数将会被自动填充）
- 对于网络第一层，通道数设置为4可以获得最佳的运算性能（NVIDIA为网络的第一层卷积提供了特殊实现，使用4通道性能更优）
- 设置内存中的张量布局为NHWC格式（如果输入NCHW格式，Tesor Core会自动转换为NHWC，当输入输出数值较大的时候，这种转置的开销往往更大）

## 四、进阶用法
### 4.1 动态图下使用梯度累加
梯度累加是指在模型训练过程中，训练一个batch的数据得到梯度后，不立即用该梯度更新模型参数，而是继续下一个batch数据的训练，得到梯度后继续循环，多次循环后梯度不断累加，直至达到一定次数后，用累加的梯度更新参数，这样可以起到变相扩大 batch_size 的作用。

动态图是天然支持Gradient Merge。即，只要不调用 ``clear_grad`` 方法，动态图的梯度会一直累积，在自动混合精度训练中，梯度累加的使用方式如下：

```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)  # 定义模型
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

accumulate_batchs_num = 10 # 梯度累加中 batch 的数量

# 定义 GradScaler，用于缩放loss比例，避免浮点数溢出
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

train_time = 0 # 总训练时长
for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time() # 开始训练时刻
        label._to(place)
        # 创建AMP上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast(level='O1'):
            output = model(data)
            loss = mse(output, label)
        # 使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()
        # 当累计的 batch 为 accumulate_batchs_num 时，更新模型参数
        if (i + 1) % accumulate_batchs_num == 0:
            # 训练模型
            scaler.step(optimizer)       # 更新参数
            scaler.update()              # 更新用于 loss 缩放的比例因子
            optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time # 记录总训练时长

print("loss:", train_loss)
print("使用AMP-O1模式耗时:{:.3f} sec".format(train_time/(epochs*nums_batch)))
```
上面的例子中，每经过 accumulate_batchs_num 次训练步骤，进行1次参数更新。

    loss: [0.6602017]
    使用AMP-O1模式耗时:0.113 sec
