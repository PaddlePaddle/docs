# 附录: 飞桨框架 2.x

2021 年 3 月飞桨推出了 2.0 版本，通过提供动态图、高层 API 等可谓是飞桨的“全面更新换代”。从 2.0 至今，飞桨框架按照一年 2 个大版本的节奏做持续更新，深耕技术提升体验。在迁移过程中，可以参考本文快速了解飞桨 2.x 版本。

## 一、动态图

### 1.1 推荐优先使用动态图模式

飞桨 2.x 版本把动态图作为默认模式（如果还想使用静态图，可通过调用`paddle.enable_static`切换）。

```python
import paddle
```

### 1.2 使用 Tensor 概念表示数据

静态图模式下，由于组网时使用的数据不能实时访问，飞桨用`Variable`来表示数据。
动态图下，从直观性等角度考虑，将数据表示概念统一为`Tensor`。动态图下`Tensor`的创建主要有两种方法：

1. 通过调用`paddle.to_tensor`函数，将`python scalar/list`，或者`numpy.ndarray`数据转换为 Paddle 的`Tensor`。具体使用方法，请查看官网的 API 文档。

    ```python
    import paddle
    import numpy as np

    paddle.to_tensor(1)
    paddle.to_tensor((1.1, 2.2))
    paddle.to_tensor(np.random.randn(3, 4))
    ```

2. 通过调用 `paddle.zeros, paddle.ones, paddle.full, paddle.arange, paddle.rand, paddle.randn, paddle.randint, paddle.normal, paddle.uniform` 等函数，创建并返回 Tensor。

## 二、API

### 2.1 API 目录结构

为了 API 组织更加简洁和清晰，飞桨对目录的组织进行了系统的条理化优化。同时还增加了高层 API，可以高低搭配使用。

**2.x 版本的 API 整体目录结构如下**：

| 目录                     | 功能和包含的 API                                              |
| ------------------------ | ------------------------------------------------------------ |
| paddle.*                 | paddle 根目录下保留了常用 API 的别名，包括：paddle.tensor, paddle.framework, paddle.device 目录下的所有 API |
| paddle.tensor            | Tensor 操作相关的 API，包括 创建 zeros, 矩阵运算 matmul, 变换 concat, 计算 add, 查找 argmax 等 |
| paddle.framework         | 框架通用 API 和动态图模式的 API，包括 no_grad 、 save 、 load 等。 |
| paddle.device            | 设备管理相关 API，包括 set_device, get_device 等。            |
| paddle.linalg            | 线性代数相关 API，包括 det, svd 等。                          |
| paddle.fft               | 快速傅里叶变换的相关 API，包括 fft, fft2 等。                 |
| paddle.amp               | 自动混合精度策略，包括 auto_cast 、 GradScaler 等。          |
| paddle.autograd          | 自动求导相关 API，包括 backward、PyLayer 等。                 |
| paddle.callbacks         | 日志回调类，包括 ModelCheckpoint 、 ProgBarLogger 等。       |
| paddle.distributed       | 分布式相关基础 API。                                          |
| paddle.distributed.fleet | 分布式相关高层 API。                                          |
| paddle.hub               | 模型拓展相关的 API，包括 list、load、help 等。                |
| paddle.io                | 数据输入输出相关 API，包括 Dataset, DataLoader 等。           |
| paddle.jit               | 动态图转静态图相关 API，包括 to_static、 ProgramTranslator、TracedLayer 等。 |
| paddle.metric            | 评估指标计算相关的 API，包括 Accuracy, Auc 等。                |
| paddle.nn                | 组网相关的 API，包括 Linear 、卷积 Conv2D 、 循环神经网络 RNN 、损失函数 CrossEntropyLoss 、 激活函数 ReLU 等。 |
| paddle.onnx              | paddle 转换为 onnx 协议相关 API，包括 export 等。                |
| paddle.optimizer         | 优化算法相关 API，包括 SGD，Adagrad, Adam 等。                |
| paddle.optimizer.lr      | 学习率衰减相关 API，包括 NoamDecay 、 StepDecay 、 PiecewiseDecay 等。 |
| paddle.regularizer       | 正则化相关 API，包括 L1Decay、L2Decay 等。                    |
| paddle.static            | 静态图下基础框架相关 API，包括 Variable, Program, Executor 等  |
| paddle.static.nn         | 静态图下组网专用 API，包括 全连接层 fc 、控制流 while_loop/cond 。 |
| paddle.text              | NLP 领域 API，包括 NLP 领域相关的数据集， 如 Imdb 、 Movielens 。 |
| paddle.utils             | 工具类相关 API，包括 CppExtension、CUDAExtension 等。         |
| paddle.vision            | 视觉领域 API，包括 数据集 Cifar10 、数据处理 ColorJitter、常用基础网络结构 ResNet 等。 |
| paddle.sparse            | 稀疏领域的 API。                                              |



### 2.3 常用 API 名称变化

- 加、减、乘、除使用全称，不使用简称
- 对于当前逐元素操作，不加 elementwise 前缀
- 对于按照某一轴操作，不加 reduce 前缀
- Conv, Pool, Dropout, BatchNorm, Pad 组网类 API 根据输入数据类型增加 1D, 2D, 3D 后缀

## 三、开发流程

### 3.1 数据处理

数据处理推荐使用**paddle.io 目录下的 Dataset，Sampler, BatchSampler, DataLoader 接口**，不推荐 reader 类接口。一些常用的数据集已经在 paddle.vision.datasets 和 paddle.text.datasets 目录实现，具体参考 API 文档。

```python
from paddle.io import Dataset

class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()

        if mode == 'train':
            self.data = [
                ['traindata1', 'label1'],
                ['traindata2', 'label2'],
                ['traindata3', 'label3'],
                ['traindata4', 'label4'],
            ]
        else:
            self.data = [
                ['testdata1', 'label1'],
                ['testdata2', 'label2'],
                ['testdata3', 'label3'],
                ['testdata4', 'label4'],
            ]

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定 index 时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = self.data[index][0]
        label = self.data[index][1]

        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)

# 测试定义的数据集
train_dataset = MyDataset(mode='train')
val_dataset = MyDataset(mode='test')

print('=============train dataset=============')
for data, label in train_dataset:
    print(data, label)

print('=============evaluation dataset=============')
for data, label in val_dataset:
    print(data, label)
```

### 3.2 组网方式

#### 3.2.1 Sequential 组网

针对顺序的线性网络结构可以直接使用 Sequential 来快速完成组网，可以减少类的定义等代码编写。

```python
import paddle

# Sequential 形式组网
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
```

#### 3.2.2 SubClass 组网

针对一些比较复杂的网络结构，就可以使用 Layer 子类定义的方式来进行模型代码编写，在`__init__`构造函数中进行组网 Layer 的声明，在`forward`中使用声明的 Layer 变量进行前向计算。子类组网方式也可以实现 sublayer 的复用，针对相同的 layer 可以在构造函数中一次性定义，在`forward 中多次调用。

```python
import paddle

# Layer 类继承方式组网
class Mnist(paddle.nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()

        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(784, 512)
        self.linear_2 = paddle.nn.Linear(512, 10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        y = self.flatten(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)

        return y

mnist = Mnist()
```

### 3.3 模型训练

#### 3.3.1 使用高层 API

飞桨提供`paddle.Model`高层 API，大部分任务可以使用此 API 用于简化训练、评估、预测类代码开发。注意区别 Model 和 Net 概念，Net 是指继承 paddle.nn.Layer 的网络结构；而 Model 是指持有一个 Net 对象，同时指定损失函数、优化算法、评估指标的可训练、评估、预测的实例。具体参考高层 API 的代码示例。

```python
import paddle
from paddle.vision.transforms import ToTensor

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()

# Mnist 继承 paddle.nn.Layer 属于 Net，model 包含了训练功能
model = paddle.Model(lenet)

# 设置训练模型所需的 optimizer, loss, metric
model.prepare(
    paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy()
    )

# 启动训练
model.fit(train_dataset, epochs=2, batch_size=64, log_freq=200)

# 启动评估
model.evaluate(test_dataset, log_freq=20, batch_size=64)
```

#### 3.3.2 使用基础 API

```python
import paddle
from paddle.vision.transforms import ToTensor

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()
loss_fn = paddle.nn.CrossEntropyLoss()

# 加载训练集 batch_size 设为 64
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train():
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=lenet.parameters())
    # 用 Adam 作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = lenet(x_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss = loss_fn(predicts, y_data)
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            adam.step()
            adam.clear_grad()

# 启动训练
train()
```

### 3.4 单机多卡启动

飞桨提供`paddle.distributed.spawn`函数来启动单机多卡训练，同时原有的`paddle.distributed.launch`的方式依然保留。

#### 3.4.1 方式 1: launch 启动

##### 3.4.1.1 高层 API 场景

当调用`paddle.Model`高层来实现训练时，想要启动单机多卡训练非常简单，代码不需要做任何修改，只需要在启动时增加一下参数`-m paddle.distributed.launch`。

```bash
# 单机单卡启动，默认使用第 0 号卡
$ python train.py

# 单机多卡启动，默认使用当前可见的所有卡
$ python -m paddle.distributed.launch train.py

# 单机多卡启动，设置当前使用的第 0 号和第 1 号卡
$ python -m paddle.distributed.launch --selected_gpus='0,1' train.py

# 单机多卡启动，设置当前使用第 0 号和第 1 号卡
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m paddle.distributed.launch train.py
```

##### 3.4.1.2 基础 API 场景

如果使用基础 API 实现训练，想要启动单机多卡训练，需要对单机单卡的代码进行 3 处修改，具体如下：

```python
import paddle
from paddle.vision.transforms import ToTensor

# 第 1 处改动，导入分布式训练所需要的包
import paddle.distributed as dist

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()
loss_fn = paddle.nn.CrossEntropyLoss()

# 加载训练集 batch_size 设为 64
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train(model):
    # 第 2 处改动，初始化并行环境
    dist.init_parallel_env()

    # 第 3 处改动，增加 paddle.DataParallel 封装
    lenet = paddle.DataParallel(model)
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=lenet.parameters())
    # 用 Adam 作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = lenet(x_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss = loss_fn(predicts, y_data)
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            adam.step()
            adam.clear_grad()

# 启动训练
train(lenet)
```

修改完后保存文件，然后使用跟高层 API 相同的启动方式即可

**注意：** 单卡训练不支持调用 `init_parallel_env`，请使用以下几种方式进行分布式训练。

```bash
# 单机多卡启动，默认使用当前可见的所有卡
$ python -m paddle.distributed.launch train.py

# 单机多卡启动，设置当前使用的第 0 号和第 1 号卡
$ python -m paddle.distributed.launch --selected_gpus '0,1' train.py

# 单机多卡启动，设置当前使用第 0 号和第 1 号卡
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m paddle.distributed.launch train.py
```

#### 3.4.2 方式 2: spawn 启动

launch 方式启动训练，以文件为单位启动多进程，需要在启动时调用 `paddle.distributed.launch` ，对于进程的管理要求较高。飞桨框架 提供 `spawn` 启动方式，可以更好地控制进程，在日志打印、训练退出时更友好。使用示例如下：

```python
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))

def train(print_result=False):

    # 1. 初始化并行训练环境
    dist.init_parallel_env()

    # 2. 创建并行训练 Layer 和 Optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(
        learning_rate=0.001, parameters=dp_layer.parameters())

    # 3. 运行网络
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    if print_result is True:
        print("loss:", loss.numpy())

    loss.backward()

    adam.step()
    adam.clear_grad()

# 使用方式 1：仅传入训练函数
# 适用场景：训练函数不需要任何参数，并且需要使用所有当前可见的 GPU 设备并行训练
if __name__ == '__main__':
    dist.spawn(train)

# 使用方式 2：传入训练函数和参数
# 适用场景：训练函数需要一些参数，并且需要使用所有当前可见的 GPU 设备并行训练
if __name__ == '__main__':
    dist.spawn(train, args=(True,))

# 使用方式 3：传入训练函数、参数并指定并行进程数
# 适用场景：训练函数需要一些参数，并且仅需要使用部分可见的 GPU 设备并行训练，例如：
# 当前机器有 8 张 GPU 卡 {0,1,2,3,4,5,6,7}，此时会使用前两张卡 {0,1}；
# 或者当前机器通过配置环境变量 CUDA_VISIBLE_DEVICES=4,5,6,7，仅使 4 张
# GPU 卡可见，此时会使用可见的前两张卡 {4,5}
if __name__ == '__main__':
    dist.spawn(train, args=(True,), nprocs=2)

# 使用方式 4：传入训练函数、参数、指定进程数并指定当前使用的卡号
# 使用场景：训练函数需要一些参数，并且仅需要使用部分可见的 GPU 设备并行训练，但是
# 可能由于权限问题，无权配置当前机器的环境变量，例如：当前机器有 8 张 GPU 卡
# {0,1,2,3,4,5,6,7}，但你无权配置 CUDA_VISIBLE_DEVICES，此时可以通过
# 指定参数 selected_gpus 选择希望使用的卡，例如 selected_gpus='4,5'，
# 可以指定使用第 4 号卡和第 5 号卡
if __name__ == '__main__':
    dist.spawn(train, nprocs=2, selected_gpus='4,5')
```

### 3.5 模型保存

Paddle 保存的模型有两种格式，一种是训练格式，保存模型参数和优化器相关的状态，可用于恢复训练；一种是预测格式，保存预测的静态图网络结构以及参数，用于预测部署。

#### 3.5.1 高层 API 场景

高层 API 下用于预测部署的模型保存方法为：

```python
model = paddle.Model(Mnist())
# 预测格式，保存的模型可用于预测部署
model.save('mnist', training=False)
# 保存后可以得到预测部署所需要的模型
```

#### 3.5.2 基础 API 场景

动态图训练的模型，可以通过动静转换功能，转换为可部署的静态图模型，具体做法如下：

```python
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    # 第 1 处改动
    # 通过 InputSpec 指定输入数据的形状，None 表示可变长
    # 通过 to_static 装饰器将动态图转换为静态图 Program
    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out


net = SimpleNet()

# 第 2 处改动
# 保存静态图模型，可用于预测部署
paddle.jit.save(net, './simple_net')
```

### 3.6 推理

推理库 Paddle Inference 的 API 做了升级，简化了写法，以及去掉了历史上冗余的概念。API 的变化为纯增，原有 API 保持不变，但推荐新的 API 体系，旧 API 在后续版本会逐步删除。

#### 3.6.1 C++ API

重要变化：

- 命名空间从 `paddle` 变更为 `paddle_infer`
- `PaddleTensor`, `PaddleBuf` 等被废弃，`ZeroCopyTensor` 变为默认 Tensor 类型，并更名为 `Tensor`
- 新增 `PredictorPool` 工具类简化多线程 predictor 的创建，后续也会增加更多周边工具
- `CreatePredictor` (原 `CreatePaddlePredictor`) 的返回值由 `unique_ptr` 变为 `shared_ptr` 以避免 Clone 后析构顺序出错的问题

API 变更

| 原有命名                     | 现有命名                     | 行为变化                      |
| ---------------------------- | ---------------------------- | ----------------------------- |
| 头文件 `paddle_infer.h`      | 无变化                       | 包含旧接口，保持向后兼容      |
| 无                           | `paddle_inference_api.h`     | 新 API，可以与旧接口并存      |
| `CreatePaddlePredictor`      | `CreatePredictor`            | 返回值变为 shared_ptr         |
| `ZeroCopyTensor`             | `Tensor`                     | 无                            |
| `AnalysisConfig`             | `Config`                     | 无                            |
| `TensorRTConfig`             | 废弃                         |                               |
| `PaddleTensor`+ `PaddleBuf`  | 废弃                         |                               |
| `Predictor::GetInputTensor`  | `Predictor::GetInputHandle`  | 无                            |
| `Predictor::GetOutputTensor` | `Predictor::GetOutputHandle` | 无                            |
|                              | `PredictorPool`              | 简化创建多个 predictor 的支持 |

使用新 C++ API 的流程与之前完全一致，只有命名变化

```cpp
#include "paddle_infernce_api.h"
using namespace paddle_infer;

Config config;
config.SetModel("xxx_model_dir");

auto predictor = CreatePredictor(config);

// Get the handles for the inputs and outputs of the model
auto input0 = predictor->GetInputHandle("X");
auto output0 = predictor->GetOutputHandle("Out");

for (...) {
  // Assign data to input0
  MyServiceSetData(input0);

  predictor->Run();

  // get data from the output0 handle
  MyServiceGetData(output0);
}
```

#### 3.6.2 Python API

Python API 的变更与 C++ 基本对应。
