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
- FP16可以充分利用英伟达Volta及Turing架构GPU提供的Tensor Cores技术。在相同的GPU硬件上，Tensor Cores的FP16计算吞吐量是FP32的8倍。

## 三、使用飞桨框架实现自动混合精度
使用飞桨框架提供的API，``paddle.amp.auto_cast`` 和 ``paddle.amp.GradScaler`` 能够实现自动混合精度训练（Automatic Mixed Precision，AMP），即在相关OP的计算中，自动选择FP16或FP32计算。开启AMP模式后，使用FP16与FP32进行计算的OP列表可见该[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/Overview_cn.html)。下面来看一个具体的例子，来了解混合精度训练。

### 3.1 辅助函数
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

### 3.2 构建一个简单的网络

构建一个简单的网络，用于对比使用普通方法进行训练与使用混合精度训练的训练速度。该网络由三层 ``Linear`` 组成，其中前两层 ``Linear`` 后接 ``ReLU`` 激活函数。


```python
import paddle
import paddle.nn as nn

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

设置训练的相关参数，这里为了能有效的看出混合精度训练对于训练速度的提升，将 ``input_size`` 与 ``output_size`` 的值设为较大的值，为了使用GPU 提供的``Tensor Core`` 性能，还需将 ``batch_size`` 设置为 8 的倍数。


```python
epochs = 5
input_size = 4096   # 设为较大的值
output_size = 4096  # 设为较大的值
batch_size = 512    # batch_size 为8的倍数
nums_batch = 50

train_data = [paddle.randn((batch_size, input_size)) for _ in range(nums_batch)]
labels = [paddle.randn((batch_size, output_size)) for _ in range(nums_batch)]

loss_fc = nn.MSELoss()
```

### 3.3 使用默认的训练方式进行训练


```python
model = SimpleNet(input_size, output_size)  # 定义模型

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

start_timer() # 获取训练开始时间

for epoch in range(epochs):
    for data, label in zip(train_data, labels):

        output = model(data)
        loss = loss_fc(output, label)

        # 反向传播
        loss.backward()

        # 训练模型
        optimizer.step()
        optimizer.clear_grad()

end_timer_and_print("默认耗时:") # 获取结束时间并打印相关信息
```


    默认耗时:
    共计耗时 = 2.650 sec


### 3.4 使用AMP训练模型

在飞桨框架中，使用自动混合精度训练，需要进行三个步骤：

- Step1： 定义 ``GradScaler`` ，用于缩放 ``loss`` 比例，避免浮点数溢出
- Step2： 使用 ``auto_cast`` 用于创建AMP上下文环境，该上下文中自动会确定每个OP的输入数据类型（FP16或FP32）
- Step3： 使用 Step1中定义的 ``GradScaler`` 完成 ``loss`` 的缩放，用缩放后的 ``loss`` 进行反向传播，完成训练


```python
model = SimpleNet(input_size, output_size)  # 定义模型

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

# Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

start_timer() # 获取训练开始时间

for epoch in range(epochs):
    for data, label in zip(train_data, labels):

        # Step2：创建AMP上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast():
            output = model(data)
            loss = loss_fc(output, label)

        # Step3：使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()

        # 训练模型
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()

print(loss)
end_timer_and_print("使用AMP模式耗时:")
```

    Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [1.24434137])

    使用AMP模式耗时:
    共计耗时 = 1.291 sec


## 四、总结
从上面的示例中可以看出，使用自动混合精度训练，共计耗时约 1.291s，而普通的训练方式则耗时 2.650s，训练速度提升约为 1.88倍。如需更多使用混合精度训练的示例，请参考飞桨模型库： [paddlepaddle/models](https://github.com/PaddlePaddle/models)。
