# 自动混合精度训练

一般情况下，训练深度学习模型时使用的数据类型为单精度（FP32）。2018年，百度与NVIDIA联合发表论文：[MIXED PRECISION TRAINING](https://arxiv.org/pdf/1710.03740.pdf)，提出了混合精度训练的方法。混合精度训练是指在训练过程中，同时使用单精度（FP32）和半精度（FP16），其目的是相较于使用单精度（FP32）训练模型，在保持精度持平的条件下，能够加速训练。本文将介绍如何使用飞桨框架，实现自动混合精度训练。  

## 一、半精度浮点类型 FP16

首先介绍半精度（FP16）。如图1所示，半精度（FP16）是一种相对较新的浮点类型，在计算机中使用2字节（16位）存储。在IEEE 754-2008标准中，它亦被称作binary16。与计算中常用的单精度（FP32）和双精度（FP64）类型相比，FP16更适于在精度要求不高的场景中使用。

<figure align="center">
    <img src="https://paddleweb-static.bj.bcebos.com/images/fp16.png" width="600" alt='missing'/>
    <figcaption><center>图 1. 半精度和单精度数据示意图</center></figcaption>
</figure>

## 二、NVIDIA GPU的FP16算力

在使用相同的超参数下，混合精度训练使用半精度浮点（FP16）和单精度（FP32）浮点即可达到与使用纯单精度训练相同的准确率，并可加速模型的训练速度。这主要得益于英伟达推出的Volta及Turing架构GPU在使用FP16计算时具有如下特点：
- FP16可降低一半的内存带宽和存储需求，这使得在相同的硬件条件下研究人员可使用更大更复杂的模型以及更大的batch size大小。
- FP16可以充分利用英伟达Volta、Turing、Ampere架构GPU提供的Tensor Cores技术。在相同的GPU硬件上，Tensor Cores的FP16计算吞吐量是FP32的8倍。

## 三、使用飞桨框架实现自动混合精度

使用飞桨框架提供的API，能够在原始训练代码基础上快速开启自动混合精度训练（Automatic Mixed Precision，AMP），即在相关OP的计算中，根据一定的规则，自动选择FP16或FP32计算。

依据FP16在模型中的使用程度划分，飞桨的AMP分为两个等级：
- level = ’O1‘：采用黑白名单策略进行混合精度训练，黑名单中的OP将采用FP32计算，白名单中的OP将采用FP16计算，训练过程中框架会自动将白名单OP的输入参数数据类型从FP32转为FP16，使用FP16与FP32进行计算的OP列表可见该[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/Overview_cn.html)。
- level = ’O2‘：该模式采用了比O1更为激进的策略，除了框架不支持FP16计算的OP，其他全部采用FP16计算，框架会预先将网络参数从FP32转换为FP16，相比O1，训练过程中无需做FP32转为FP16的操作，训练速度会有更明显的提升，但可能会存在精度问题，为此，框架提供了自定义黑名单，用户可通过该名单指定一些存在精度问题的OP执行FP32运算。

飞桨动态图与静态图均为用户提供了便捷的API用于开启混合精度训练，下面以具体的训练代码为例，来了解如何使用飞桨框架实现混合精度训练。

### 3.1 动态图混合精度训练

飞桨动态图提供了一系列便捷的API用于实现混合精度训练：``paddle.amp.GradScaler``、``paddle.amp.auto_cast``、``paddle.amp.decorate``。

首先定义辅助函数，用来计算训练时间。

```python
import time

# 开始时间
start_time = None

def start_timer():
    # 获取开始时间
    global start_time
    start_time = time.time()

def end_timer_and_print(msg):
    # 打印信息并输出训练时间
    end_time = time.time()
    print("\n" + msg)
    print("共计耗时 = {:.3f} sec".format(end_time - start_time))
```

<a name="3.1.1"></a>
#### 3.1.1 动态图FP32训练

1）构建一个简单的网络：用于对比使用普通方法进行训练与使用混合精度训练的训练速度。该网络由三层 ``Linear`` 组成，其中前两层 ``Linear`` 后接 ``ReLU`` 激活函数。

```python
import paddle
import paddle.nn as nn
import numpy

paddle.seed(100)
numpy.random.seed(100)

class SimpleNet(nn.Layer):

    def __init__(self, input_size, output_size):
        
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size, output_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size, output_size)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x
```

2）设置训练的相关参数及训练数据：这里为了能有效的看出混合精度训练对于训练速度的提升，将 ``input_size`` 与 ``output_size`` 的值设为较大的值，为了使用GPU 提供的``Tensor Core`` 性能，还需将 ``batch_size`` 设置为 8 的倍数（基于混合精度训练的性能优化方法见：<a href="#四">四、混合精度训练性能优化</a>）。

```python
epochs = 5
input_size = 4096   # 设为较大的值
output_size = 4096  # 设为较大的值
batch_size = 512    # batch_size 为8的倍数
nums_batch = 50

datas = [paddle.to_tensor(numpy.random.random(size=(batch_size, input_size)).astype('float32')) for _ in range(nums_batch)]
labels = [paddle.to_tensor(numpy.random.random(size=(batch_size, input_size)).astype('float32')) for _ in range(nums_batch)]

mse = paddle.nn.MSELoss()

```

3）使用动态图FP32训练：

```python
model = SimpleNet(input_size, output_size)  # 定义模型
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

start_timer() # 获取训练开始时间

for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (data, label) in enumerate(batchs):
        # 前向计算
        output = model(data)
        loss = mse(output, label)

        # 反向传播
        loss.backward()

        # 训练模型
        optimizer.step()
        optimizer.clear_grad()

print(loss)
end_timer_and_print("使用FP32模式耗时:") # 获取结束时间并打印相关信息
```

    Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
        [0.40839708])

    使用FP32模式耗时:
    共计耗时 = 2.925 sec


#### 3.1.2 动态图AMP-O1训练：

在飞桨框架中，使用AMP-O1训练训练，需要在FP32代码的基础上改动三处：

- Step1： 定义 ``paddle.amp.GradScaler`` ，用于缩放 ``loss`` 比例，避免浮点数下溢
- Step2： 使用 ``paddle.amp.auto_cast`` 创建AMP上下文环境，在该上下文内，框架会根据框架预设的黑白名单，自动确定每个OP的输入数据类型（FP16或FP32）
- Step3： 在训练代码中使用Step1中定义的 ``GradScaler`` 完成 ``loss`` 的缩放，用缩放后的 ``loss`` 进行反向传播，完成训练

```python
model = SimpleNet(input_size, output_size)  # 定义模型
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

# Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

start_timer() # 获取训练开始时间

for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (data, label) in enumerate(batchs):

        # Step2：创建AMP-O1上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast(level='O1'):
            output = model(data)
            loss = mse(output, label)

        # Step3：使用Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()

        # 训练模型
        scaler.step(optimizer)       # 更新参数
        scaler.update()              # 更新用于 loss 缩放的比例因子
        optimizer.clear_grad()

print(loss)
end_timer_and_print("使用AMP-O1模式耗时:")
```

    Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
        [0.40840322])

    使用AMP-O1模式耗时:
    共计耗时 = 1.208 sec

- ``paddle.amp.GradScaler``使用介绍见[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/GradScaler_cn.html)
- ``paddle.amp.auto_cast``使用介绍见[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/auto_cast_cn.html)

#### 3.1.3 动态图AMP-O2训练：

O2模式采用了比O1更为激进的策略，除了框架不支持FP16计算的OP，其他全部采用FP16计算，需要在训练前将网络参数从FP32转为FP16，在FP32代码的基础上改动四处：

- Step1： 定义 ``paddle.amp.GradScaler`` ，用于缩放 ``loss`` 比例，避免浮点数下溢
- Step2： 使用 ``paddle.amp.decorate`` 将网络参数从FP32转换为FP16
- Step3： 使用 ``paddle.amp.auto_cast`` 创建AMP上下文环境，在该上下文内，框架会将所有支持FP16的OP都采用FP16进行计算（自定义的黑名单除外），其他OP采用FP32进行计算
- Step4： 在训练代码中使用Step1中定义的 ``GradScaler`` 完成 ``loss`` 的缩放，用缩放后的 ``loss`` 进行反向传播，完成训练


```python
model = SimpleNet(input_size, output_size)  # 定义模型
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

# Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

# Step2：在level=’O2‘模式下，将网络参数从FP32转换为FP16
model = paddle.amp.decorate(models=model, level='O2')

start_timer() # 获取训练开始时间

for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (data, label) in enumerate(batchs):

        # Step3：创建AMP上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast(level='O2'):
            output = model(data)
            loss = mse(output, label)

        # Step4：使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()

        # 训练模型
        scaler.step(optimizer)       # 更新参数
        scaler.update()              # 更新用于 loss 缩放的比例因子
        optimizer.clear_grad()

print(loss)
end_timer_and_print("使用AMP-O2模式耗时:")
```

    Tensor(shape=[1], dtype=float16, place=CUDAPlace(0), stop_gradient=False,
        [0.41528320])

    使用AMP-O2模式耗时:
    共计耗时 = 0.833 sec

- ``paddle.amp.decorate``使用介绍见[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/decorate_cn.html)

从上面的示例中可以看出，使用自动混合精度训练，O1模式共计耗时约 2.852s，O2模式共计耗时约 1.911s，而普通的训练方式则耗时 6.736s，O1模式训练速度提升约为 2.4倍，O2模式训练速度提升约为 3.5倍。如需更多使用混合精度训练的示例，请参考飞桨模型库： [paddlepaddle/models](https://github.com/PaddlePaddle/models)。

 **注：**受机器环境影响，上述示例代码的训练耗时统计可能存在差异，该影响主要包括：GPU利用率、CPU利用率的等。

### 3.2 静态图混合精度训练

飞桨静态图提供了一系列便捷的API用于实现混合精度训练：``paddle.static.amp.decorate``、``paddle.static.amp.fp16_guard``。

#### 3.2.1 静态图FP32训练

采用与3.1.1节动态图训练相同的网络结构：<a href="#3.1.1">3.1.1 动态图FP32训练</a>），静态图网络初始化如下：

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

datas = [numpy.random.random(size=(batch_size, input_size)).astype('float32') for _ in range(nums_batch)]
labels = [numpy.random.random(size=(batch_size, input_size)).astype('float32') for _ in range(nums_batch)]
start_timer() # 获取训练开始时间
for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (train_data, traiin_label) in enumerate(batchs):
        loss_data = exe.run(main_program, feed={data.name: train_data, label.name: traiin_label }, fetch_list=[loss.name])

print(loss_data)
end_timer_and_print("使用FP32模式耗时:") # 获取结束时间并打印相关信息

```

    [array([0.40839708], dtype=float32)]

    使用FP32模式耗时:
    共计耗时 = 4.717 sec

#### 3.2.2 静态图AMP-O1训练

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

datas = [numpy.random.random(size=(batch_size, input_size)).astype('float32') for _ in range(nums_batch)]
labels = [numpy.random.random(size=(batch_size, input_size)).astype('float32') for _ in range(nums_batch)]
start_timer() # 获取训练开始时间
for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (train_data, traiin_label) in enumerate(batchs):
        loss_data = exe.run(main_program, feed={data.name: train_data, label.name: traiin_label }, fetch_list=[loss.name])

print(loss_data)
end_timer_and_print("使用AMP-O1模式耗时:") # 获取结束时间并打印相关信息

```

    [array([0.40841], dtype=float32)]

    使用AMP-O1模式耗时:
    共计耗时 = 3.064 sec

`paddle.static.amp.CustomOpLists`用于自定义黑白名单，黑名单op执行FP32 kernel、白名单op执行FP16 kernel。

#### 3.2.3 静态图AMP-O2训练

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

datas = [numpy.random.random(size=(batch_size, input_size)).astype('float16') for _ in range(nums_batch)]
labels = [numpy.random.random(size=(batch_size, input_size)).astype('float16') for _ in range(nums_batch)]
start_timer() # 获取训练开始时间
for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (train_data, traiin_label) in enumerate(batchs):
        loss_data = exe.run(main_program, feed={data.name: train_data, label.name: traiin_label }, fetch_list=[loss.name])

print(loss_data)
end_timer_and_print("使用AMP-O2模式耗时:") # 获取结束时间并打印相关信息

```

    [array([0.4153], dtype=float16)]

    使用AMP-O2模式耗时:
    共计耗时 = 2.222 sec

2）设置``paddle.static.amp.decorate``的参数``use_pure_fp16``为 True，同时设置参数``use_fp16_guard``为True，通过``paddle.static.amp.fp16_guard``控制使用FP16的计算范围

在模型定义的代码中加入`fp16_guard`控制部分网络执行在FP16下：

```python
class SimpleNet(nn.Layer):

    def __init__(self, input_size, output_size):
        
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size, output_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        # 控制FP16使用范围
        with paddle.static.amp.fp16_guard():
            x = self.relu1(x)
            x = self.linear2(x)
            x = self.relu2(x)
            x = self.linear3(x)

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

datas = [numpy.random.random(size=(batch_size, input_size)).astype('float32') for _ in range(nums_batch)]
labels = [numpy.random.random(size=(batch_size, input_size)).astype('float32') for _ in range(nums_batch)]
start_timer() # 获取训练开始时间
for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (train_data, traiin_label) in enumerate(batchs):
        loss_data = exe.run(main_program, feed={data.name: train_data, label.name: traiin_label }, fetch_list=[loss.name])

print(loss_data)
end_timer_and_print("使用AMP-O2模式耗时:") # 获取结束时间并打印相关信息

```

    [array([0.4127627], dtype=float32)]

    使用AMP-O2模式耗时:
    共计耗时 = 3.407 sec

<a name="四"></a>
## 四、混合精度训练性能优化

飞桨AMP提升模型训练性能的根本原因是：利用 Tensor Core 来加速 FP16 下的``matmul``和``conv``运算，为了获得最佳的加速效果，Tensor Core 对矩阵乘和卷积运算有一定的使用约束，约束如下：

### 4.1 矩阵乘使用建议

通用矩阵乘 (GEMM) 定义为：``C = A * B + C``，其中：
- A 维度为：M x K
- B 维度为：K x N
- C 维度为：M x N

矩阵乘使用建议如下：
- 根据Tensor Core使用建议，当矩阵维数 M、N、K 是8（A100架构GPU为16）的倍数时（FP16数据下），性能最优。

### 4.2 卷积计算使用建议

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

## 五、进阶用法
### 5.1 动态图下使用梯度累加
梯度累加是指在模型训练过程中，训练一个batch的数据得到梯度后，不立即用该梯度更新模型参数，而是继续下一个batch数据的训练，得到梯度后继续循环，多次循环后梯度不断累加，直至达到一定次数后，用累加的梯度更新参数，这样可以起到变相扩大 batch_size 的作用。

在自动混合精度训练中，也支持梯度累加，使用方式如下：


```python
model = SimpleNet(input_size, output_size)  # 定义模型

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

accumulate_batchs_num = 10 # 梯度累加中 batch 的数量

# 定义 GradScaler
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

start_timer() # 获取训练开始时间

for epoch in range(epochs):
    batchs = zip(datas, labels)
    for i, (data, label) in enumerate(batchs):

        # 创建AMP上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast():
            output = model(data)
            loss = mse(output, label)

        # 使用 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()

        # 当累计的 batch 为 accumulate_batchs_num 时，更新模型参数
        if (i + 1) % accumulate_batchs_num == 0:

            # 训练模型
            scaler.step(optimizer)       # 更新参数
            scaler.update()              # 更新用于 loss 缩放的比例因子
            optimizer.clear_grad()

print(loss)
end_timer_and_print("使用AMP-O1模式耗时:")
```

    Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
        [0.40864223])

    使用AMP-O1模式耗时:
    共计耗时 = 0.970 sec
