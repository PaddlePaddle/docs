# 飞桨框架2.0beta升级指南

## 升级概要
本版本是2.0版的公测版，相对1.8版本有重大升级，涉及开发方面的重要变化如下：

 - 动态图功能完善，动态图模下数据表示概念为Tensor，推荐使用动态图模式；
 - API目录体系调整，API的命名和别名进行了统一规范化，虽然兼容老版API，但请使用新API体系开发；
 - 数据处理、组网方式、模型训练、多卡启动、模型保存和推理等开发流程都有了对应优化，请对应查看说明；

以上变化请仔细阅读本指南。对于已有模型的升级，我们还提供了2.0转换工具（见附录）提供更自动化的辅助。
其他一些功能增加方面诸如动态图对量化训练、混合精度的支持、动静转换等方面不在本指南列出，具体可查看[Release Note](https://github.com/PaddlePaddle/Paddle/releases/tag/v2.0.0-beta0#)或对应文档。

## 一、动态图

### 推荐优先使用动态图模式
飞桨2.0版本将会把动态图作为默认模式。2.0-beta版本虽然还未做默认切换，但推荐大家优先使用动态图模式，需要在程序开始时调用`paddle.disable_static`切换到动态图。2.0-rc版本后默认模式将切换为动态图，此行代码可删除。（2.0-rc版本后如果还想使用静态图，可通过调用`paddle.enable_static`切换）。

```python
import paddle

# 2.0-beta版本需要调用下面代码，切换到动态图模式
# 2.0-rc版本可以删除这一行
paddle.disable_static()
```

### 使用Tensor概念表示数据
静态图模式下，由于组网时使用的数据不能实时访问，Paddle用Variable来表示数据。
动态图下，从直观性等角度考虑，将数据表示概念统一为Tensor。动态图下Tensor的创建主要有两种方法：

1. 通过调用paddle.to_tensor函数，将python scalar/list，或者numpy.ndarray数据转换为Paddle的Tensor。具体使用方法，请查看官网的API文档。

```python
import paddle

paddle.disable_static()
paddle.to_tensor(1)
paddle.to_tensor((1.1, 2.2))
paddle.to_tensor(np.random.randn(3, 4))
```

2. 通过调用 `paddle.zeros, paddle.ones, paddle.full, paddle.arange, paddle.rand, paddle.randn, paddle.randint, paddle.normal, paddle.uniform` 等函数，创建并返回Tensor。

## 二、API
### API目录结构

为了API组织更加简洁和清晰，将原来padddle.fluid.xxx的目录体系全新升级为paddle.xxx，并对子目录的组织进行了系统的条理化优化。同时还增加了高层API，可以高低搭配使用。paddle.fluid目录下暂时保留了1.8版本API，主要是兼容性考虑，未来会被删除。
**基于2.0的开发任务，请使用paddle目录下的API，不要再使用paddle.fluid目录下的API。** 如果发现Paddle目录下有API缺失的情况，推荐使用基础API进行组合实现；您也可以通过在 [github](https://github.com/paddlepaddle/paddle) 上提issue的方式向我们反馈。

**2.0-beta版本的API 整体目录结构如下**：

| 目录 | 功能和包含的API |
| :--- | --------------- |
| paddle.*          | paddle根目录下保留了常用API的别名，当前包括：paddle.tensor和paddle.framework目录下的所有API |
| paddle.tensor     | 跟tensor操作相关的API，比如：创建zeros, 矩阵运算matmul, 变换concat, 计算add, 查找argmax等 |
| paddle.nn         | 跟组网相关的API，比如：Linear，卷积，LSTM，损失函数,激活函数等 |
| paddle.static.nn  | 静态图下组网专用API，比如：输入占位符data, 全连接层fc, 控制流while_loop/cond |
| paddle.static | 静态图下基础框架相关API，比如：Variable, Program, Executor等 |
| paddle.framework  | 框架通用API和动态图模式的API，比如：to_tensor, no_grad等 |
| paddle.optimizer  | 优化算法相关API，比如：SGD，Adagrad, Adam等                  |
| paddle.optimizer.lr_scheduler  | 学习率衰减相关API                  |
| paddle.metric     | 评估指标计算相关的API，比如：accuracy, auc等             |
| paddle.io         | 数据输入输出相关API，比如：Dataset, DataLoader等 |
| paddle.device     | 设备管理相关API，比如：CPUPlace， CUDAPlace等                |
| paddle.distributed      | 分布式相关基础API                                                |
| paddle.distributed.fleet      | 分布式相关高层API                                         |
| paddle.vision     | 视觉领域API，比如，数据集，数据处理，常用基础网络结构，比如resnet             |
| paddle.text       | NLP领域API, 比如，数据集，数据处理，常用网络结构，比如Transformer |

### API别名规则

- 为了方便用户使用，API会在不同的路径下建立别名：
	- 所有framework, tensor目录下的API，均在paddle根目录建立别名；除少数特殊API外，其他API在paddle根目录下均没有别名。
	- paddle.nn目录下除functional目录以外的所有API，在paddle.nn目录下均有别名；functional目录中的API，在paddle.nn目录下均没有别名。
- **推荐用户优先使用较短的路径的别名**，比如`paddle.add -> paddle.tensor.add`，推荐优先使用`paddle.add`
- 以下为一些特殊的别名关系，推荐使用左边的API名称：
  - paddle.sigmoid -> paddle.tensor.sigmoid -> paddle.nn.functional.sigmoid
  - paddle.tanh -> paddle.tensor.tanh -> paddle.nn.functional.tanh
  - paddle.remainder -> paddle.mod -> paddle.floor_mod
  - paddle.divide -> paddle.true_divide
  - paddle.rand -> paddle.uniform
  - paddle.randn -> paddle.standard_normal
  - Optimizer.clear_grad -> Optimizer.clear_gradients
  - Optimizer.set_state_dict -> Optimizer.set_dict
  - Optimizer.get_lr -> Optimizer.current_step_lr
  - Layer.clear_grad -> Layer.clear_gradients
  - Layer.set_state_dict -> Layer.set_dict

### 常用API名称变化

- 加、减、乘、除使用全称，不使用简称
- 对于当前逐元素操作，不加elementwise前缀
- 对于按照某一轴操作，不加reduce前缀
- Conv, Pool, Dropout, BatchNorm, Pad组网类API根据输入数据类型增加1d, 2d, 3d后缀

  | Paddle 1.8  API名称  | Paddle 2.0-beta 对应的名称|
  | --------------- | ------------------------ |
  | paddle.fluid.layers.elementwise_add | paddle.add               |
  | paddle.fluid.layers.elementwise_sub | paddle.subtract           |
  | paddle.fluid.layers.elementwise_mul | paddle.multiply          |
  | paddle.fluid.layers.elementwise_div | paddle.divide |
  | paddle.fluid.layers.elementwise_max | paddle.maximum             |
  | paddle.fluid.layers.elementwise_min | paddle.minimum |
  | paddle.fluid.layers.reduce_sum | paddle.sum |
  | paddle.fluid.layers.reduce_prod | paddle.prod |
  | paddle.fluid.layers.reduce_max | paddle.max        |
  | paddle.fluid.layers.reduce_min | paddle.min        |
  | paddle.fluid.layers.reduce_all | paddle.all        |
  | paddle.fluid.layers.reduce_any | paddle.any        |
  | paddle.fluid.dygraph.Conv2D | paddle.nn.Conv2d |
  | paddle.fluid.dygraph.Conv2DTranspose | paddle.nn.ConvTranspose2d |
  | paddle.fluid.dygraph.Pool2D | paddle.nn.MaxPool2d, paddle.nn.AvgPool2d |

## 三、开发流程
### 数据处理
数据处理推荐使用**paddle.io目录下的Dataset，Sampler, BatchSampler, DataLoader接口**，不推荐reader类接口。一些常用的数据集已经在paddle.vision.datasets和paddle.text.datasets目录实现，具体参考API文档。

```python
from paddle.io import Dataset

class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
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
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
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

### 组网方式
#### Sequential 组网

针对顺序的线性网络结构我们可以直接使用Sequential来快速完成组网，可以减少类的定义等代码编写。

```python
import paddle
paddle.disable_static()

# Sequential形式组网
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
```

#### SubClass组网

 针对一些比较复杂的网络结构，就可以使用Layer子类定义的方式来进行模型代码编写，在`__init__`构造函数中进行组网Layer的声明，在`forward`中使用声明的Layer变量进行前向计算。子类组网方式也可以实现sublayer的复用，针对相同的layer可以在构造函数中一次性定义，在forward中多次调用。

```python
import paddle
paddle.disable_static()

# Layer类继承方式组网
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

### 模型训练

#### 使用高层API

增加了paddle.Model高层API，大部分任务可以使用此API用于简化训练、评估、预测类代码开发。注意区别Model和Net概念，Net是指继承paddle.nn.Layer的网络结构；而Model是指持有一个Net实例，同时指定损失函数、优化算法、评估指标的可训练、评估、预测的实例。具体参考高层API的代码示例。

```python
import paddle
paddle.disable_static()

train_dataset = paddle.vision.datasets.MNIST(mode='train')
test_dataset = paddle.vision.datasets.MNIST(mode='test')
lenet = paddle.vision.models.LeNet()

# Mnist继承paddle.nn.Layer属于Net，model包含了训练功能
model = paddle.Model(lenet)

# 设置训练模型所需的optimizer, loss, metric
model.prepare(
    paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy(topk=(1, 2))
    )

# 启动训练
model.fit(train_dataset, epochs=2, batch_size=64, log_freq=200)

# 启动评估
model.evaluate(test_dataset, log_freq=20, batch_size=64)
```

#### 使用基础API

```python
import paddle

paddle.disable_static()
train_dataset = paddle.vision.datasets.MNIST(mode='train')
test_dataset = paddle.vision.datasets.MNIST(mode='test')
lenet = paddle.vision.models.LeNet()

# 加载训练集 batch_size 设为 64
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=64, shuffle=True)

def train():
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=lenet.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            predicts = lenet(x_data)
            loss = paddle.nn.functional.cross_entropy(predicts, y_data, reduction='mean')
            acc = paddle.metric.accuracy(predicts, y_data, k=1)
            avg_acc = paddle.mean(acc)
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), avg_acc.numpy()))
            adam.step()
            adam.clear_grad()
# 启动训练
train()
```

### 单机多卡启动
2.0增加paddle.distributed.spawn函数来启动单机多卡训练，同时原有的paddle.distributed.launch的方式依然保留。

#### 方式1、launch启动

##### 高层API场景

当调用paddle.Model高层来实现训练时，想要启动单机多卡训练非常简单，代码不需要做任何修改，只需要在启动时增加一下参数`-m paddle.distributed.launch`。

```bash
# 单机单卡启动，默认使用第0号卡
$ python train.py

# 单机多卡启动，默认使用当前可见的所有卡
$ python -m paddle.distributed.launch train.py

# 单机多卡启动，设置当前使用的第0号和第1号卡
$ python -m paddle.distributed.launch --selected_gpus='0,1' train.py

# 单机多卡启动，设置当前使用第0号和第1号卡
$ export CUDA_VISIABLE_DEVICES='0,1'
$ python -m paddle.distributed.launch train.py
```

##### 基础API场景

如果使用基础API实现训练，想要启动单机多卡训练，需要对单机单卡的代码进行4处修改，具体如下：

```python
import paddle
import paddle.distributed as dist

paddle.disable_static()
train_dataset = paddle.vision.datasets.MNIST(mode='train')
test_dataset = paddle.vision.datasets.MNIST(mode='test')

# 加载训练集 batch_size 设为 64
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=64, shuffle=True)

def train():
    # 第1处改动，初始化并行环境
    dist.init_parallel_env()

    # 第2处改动，增加paddle.DataParallel封装
    net = paddle.DataParallel(LeNet())
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = net(x_data)           acc = paddle.metric.accuracy(predicts, y_data, k=2)
            avg_acc = paddle.mean(acc)
            loss = paddle.nn.functional.cross_entropy(predicts, y_data)
            avg_loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, avg_loss.numpy(), avg_acc.numpy()))
            adam.step()
            adam.clear_grad()

# 启动训练  
train()
```

修改完后保存文件，然后使用跟高层API相同的启动方式即可

```bash
# 单机单卡启动，默认使用第0号卡
$ python train.py

# 单机多卡启动，默认使用当前可见的所有卡
$ python -m paddle.distributed.launch train.py

# 单机多卡启动，设置当前使用的第0号和第1号卡
$ python -m paddle.distributed.launch --selected_gpus '0,1' train.py

# 单机多卡启动，设置当前使用第0号和第1号卡
$ export CUDA_VISIABLE_DEVICES='0,1'
$ python -m paddle.distributed.launch train.py
```

#### 方式2、spawn启动

launch方式启动训练，以文件为单位启动多进程，需要用户在启动时调用`paddle.distributed.launch`，对于进程的管理要求较高。2.0版本增加了spawn启动方式，可以更好地控制进程，在日志打印、训练退出时更友好。

```bash
# 启动train多进程训练，默认使用所有可见的GPU卡
if __name__ == '__main__':
    dist.spawn(train)

# 启动train函数2个进程训练，默认使用当前可见的前2张卡
if __name__ == '__main__':
    dist.spawn(train, nprocs=2)

# 启动train函数2个进程训练，默认使用第4号和第5号卡
if __name__ == '__main__':
    dist.spawn(train, nprocs=2, selelcted_gpus='4,5')
```

### 模型保存
Paddle保存的模型有两种格式，一种是训练格式，保存模型参数和优化器相关的状态，可用于恢复训练；一种是预测格式，保存预测的静态图网络结构以及参数，用于预测部署。
#### 高层API场景

高层API下用于预测部署的模型保存方法为：

```python
model = paddle.Model(Mnist())
# 预测格式，保存的模型可用于预测部署
model.save('mnist', training=False)
# 保存后可以得到预测部署所需要的模型
```

#### 基础API场景

动态图训练的模型，可以通过动静转换功能，转换为可部署的静态图模型，具体做法如下：

```python
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    # 第1处改动
    # 通过InputSpec指定输入数据的形状，None表示可变长
    # 通过to_static装饰器将动态图转换为静态图Program
    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

paddle.disable_static()

net = SimpleNet()

# 第2处改动
# 保存静态图模型，可用于预测部署
paddle.jit.save(net, './simple_net')
```
### 推理
推理库Paddle Inference的API做了升级，简化了写法，以及去掉了历史上冗余的概念。API的变化为纯增，原有API保持不变，但推荐新的API体系，旧API在后续版本会逐步删除。

#### C++ API

重要变化：

- 命名空间从 `paddle` 变更为 `paddle_infer`
- `PaddleTensor`, `PaddleBuf` 等被废弃，`ZeroCopyTensor` 变为默认 Tensor 类型，并更名为 `Tensor`
- 新增 `PredictorPool` 工具类简化多线程 predictor 的创建，后续也会增加更多周边工具
- `CreatePredictor` (原 `CreatePaddlePredictor`) 的返回值由 `unique_ptr` 变为 `shared_ptr` 以避免 Clone 后析构顺序出错的问题

API 变更

| 原有命名                     | 现有命名                     | 行为变化                      |
| ---------------------------- | ---------------------------- | ----------------------------- |
| 头文件 `paddle_infer.h`      | 无变化                       | 包含旧接口，保持向后兼容      |
| 无                           | `paddle_inference_api.h`     | 新API，可以与旧接口并存       |
| `CreatePaddlePredictor`      | `CreatePredictor`            | 返回值变为 shared_ptr         |
| `ZeroCopyTensor`             | `Tensor`                     | 无                            |
| `AnalysisConfig`             | `Config`                     | 无                            |
| `TensorRTConfig`             | 废弃                         |                               |
| `PaddleTensor` + `PaddleBuf` | 废弃                         |                               |
| `Predictor::GetInputTensor`  | `Predictor::GetInputHandle`  | 无                            |
| `Predictor::GetOutputTensor` | `Predictor::GetOutputHandle` | 无                            |
|                              | `PredictorPool`              | 简化创建多个 predictor 的支持 |

使用新 C++ API 的流程与之前完全一致，只有命名变化

```c++
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

#### Python API

Python API 的变更与 C++ 基本对应，会在2.0RC版发布。


## 附录
### 2.0转换工具
为了降级代码升级的成本，我们提供了转换工具，可以帮助将Paddle 1.8版本开发的代码，升级为2.0-beta的API。由于相比于Paddle 1.8版本，2.0-beta版本的API进行了大量的升级，包括API名称，参数名称，行为等。转换工具当前还不能覆盖所有的API升级；对于无法转换的API，转换工具会报错，提示用户手动升级。

https://github.com/PaddlePaddle/paddle1to2

对于转换工具没有覆盖的API，请查看官网的API文档，手动升级代码的API。

### 2.0文档教程
以下提供了2.0版本的一些示例教程：

您可以在官网[应用实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/tutorial/index_cn.html)栏目内进行在线浏览，也可以下载在这里提供的源代码:
https://github.com/PaddlePaddle/book/tree/develop/paddle2.0_docs

### 2.0API升级列表
- [Release Note](https://github.com/PaddlePaddle/Paddle/releases/tag/v2.0.0-beta0#)
- [API新增列表](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0beta-New-API-List)
- [API升级列表](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0beta-Upgraded-API-List)
