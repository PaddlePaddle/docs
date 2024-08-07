# 使用样例


## 一、概述

### 1.1 动态图和静态图简介

在深度学习模型构建上，飞桨框架支持动态图编程和静态图编程两种方式，其代码编写和执行方式均存在差异。

+ 动态图编程： 采用 Python 的编程风格，解析式地执行每一行网络代码，并同时返回计算结果。在 [模型开发](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/index_cn.html) 章节中，介绍的主要是动态图编程方式。

+ 静态图编程： 采用先编译后执行的方式。需先在代码中预定义完整的神经网络结构，飞桨框架会将神经网络描述为 Program 的数据结构，并对 Program 进行编译优化，再调用执行器获得计算结果。

动态图编程体验更佳、更易调试，但是因为采用 Python 实时执行的方式，开销较大，在性能方面与 C++ 有一定差距；静态图调试难度大，但是将前端 Python 编写的神经网络预定义为 Program 描述，转到 C++ 端重新解析执行，脱离了 Python 依赖，往往执行性能更佳，并且预先拥有完整网络结构也更利于全局优化。

想了解动态图和静态图的详细对比介绍，可参见 [动态图和静态图的差异](https://www.paddlepaddle.org.cn/tutorials/projectdetail/4047189)。

### 1.2 动态图转静态图应用场景

飞桨框架在设计时，考虑同时兼顾动态图的高易用性和静态图的高性能优势，采用『动静统一』的方案：

+ 在模型开发和训练时，推荐采用动态图编程。 可获得更好的编程体验、更易用的接口、更友好的调试交互机制。

+ 在模型推理部署时，推荐采用**动态图转静态图（以下简称：动转静）**。平滑衔接将训好的动态图模型自动保存为静态图模型，可获得更好的模型运行性能。

以上是飞桨框架推荐的使用方法，即『训练调优用动态图，推理部署用动转静』。但是在某些对模型训练性能有更高要求的场景，也可以使用动转静训练，即在动态图组网代码中添加一行装饰器 ``@to_static`` ，便可在底层转为性能更优的静态图模式下训练。

**本文即介绍动转静模块的相关用法：**

+ 动转静训练：将动态图编程的模型，转换为静态图模型 （[@to_static](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/to_static_cn.html#to-static)）进行训练。

+ 动转静模型保存和加载：既支持动转静训练的模型保存和加载，也支持将动态图训练好的模型，直接保存为静态图模型文件（[paddle.jit.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/save_cn.html#save)），然后用于推理部署。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/jit/images/overall.png" style="zoom:80%"/>
</figure>


> 注：飞桨框架 2.0 及以上版本默认的编程模式是动态图模式，包括使用高层 API 编程和基础的 API 编程。如果想切换到静态图模式编程，可以在程序的开始执行 ``enable_static()`` 函数。如果程序已经使用动态图的模式编写了，想转成静态图模式训练或者保存模型用于部署，可以使用装饰器 ``@to_static`` 。


## 二、动转静训练

**在飞桨框架中，通常情况下使用动态图训练，即可满足大部分场景需求。** 飞桨经过多个版本的持续优化，动态图模型训练的性能已经可以和静态图媲美。如果在某些场景下确实需要使用静态图模式训练，则可以使用**动转静训练功能，即仍然采用更易用的动态图编程，添加少量代码，便可在底层转为静态图训练。**

实际操作只需在待转化的函数前添加一个装饰器 [@paddle.jit.to_static](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/to_static_cn.html#to-static)，框架便通过解析 Python 代码（抽象语法树，简称：AST）等方法自动完成动静转换。具体原理可参见 [转换原理](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/principle_cn.html) 章节。

### 2.1 动转静训练应用场景

在如下场景时可以考虑使用动转静进行模型训练，带来的性能提升效果较明显：

+ **如果发现模型训练 CPU 向 GPU 调度不充分的情况下。**

  如下是模型训练时执行单个 step 的 timeline 示意图，框架通过 CPU 调度底层 Kernel 计算，在某些情况下，如果 CPU 调度时间过长，会导致 GPU 利用率不高（可终端执行 watch -n 1 nvidia-smi 观察）。

  <figure align="center">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/jit/images/timeline_base.png" style="zoom:70%" />
  </figure>
  动态图和静态图在 CPU 调度层面存在差异：

  + 动态图训练时，CPU 调度时间涉及 Python 到 C++ 的交互（Python 前端代码调起底层 C++ OP）和 C++ 代码调度；

  + 静态图训练时，是统一编译 C++ 后执行，CPU 调度时间没有 Python 到 C++ 的交互时间，只有 C++ 代码调度，因此比动态图调度时间短。

  因此如果发现是 CPU 调度时间过长，导致的 GPU 利用率低的情况，便可以采用动转静训练提升性能。从应用层面看，如果模型任务本身的 Kernel 计算时间很长，相对来说调度到 Kernel 拉起造成的影响不大，这种情况一般用动态图训练即可，比如 Bert 等模型，反之如 HRNet 等模型则可以观察 GPU 利用率来决定是否使用动转静训练。

+ **如果想要进一步对计算图优化，以提升模型训练性能的情况下。**

  相对于动态图按一行行代码解释执行，动转静后飞桨能够获取模型的整张计算图，即拥有了全局视野，因此可以借助算子融合等技术对计算图进行局部改写，替换为更高效的计算单元，我们称之为“图优化”。

  如下是应用了算子融合策略后，模型训练时执行单个 step 的 timeline 示意图。相对于图 2，飞桨框架获取了整张计算图，按照一定规则匹配到 OP3 和 OP4 可以融合为 Fuse_OP，因此可以减少 GPU 的空闲时间，提升执行效率。

<figure align="center">
  <img src="https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/jit/images/timeline_d2s.png?raw=true" style="zoom:70%" />
</figure>

  调用 ``@paddle.jit.to_static`` 进行动转静时可以使用 build_strategy 参数开启 ``fuse_elewise_add_act_op`` 、``enable_addto`` 等优化策略来对计算图进行优化，更多策略开关可以参考 [BuildStrategy](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/BuildStrategy_cn.html#buildstrategy) 接口文档。图优化策略的使用样例请参考：[4.1 动转静训练计算图优化策略](#41)。


### 2.2 动转静训练样例

结合一个简单的网络训练示例，介绍动转静训练的方法，主要有两种方式：

#### 2.2.1 方式一：使用 ``@paddle.jit.to_static`` 装饰器

下面的示例代码中展示了如何使用动转静训练 LinearNet 网络，在前向计算 forward 函数前添加一个装饰器，即可以将动态图网络转为静态图网络。对于其他动态图下的训练代码无需进行改动即可进行动转静的训练。

```python
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static       # <----在前向计算 forward 函数前添加一个装饰器
    def forward(self, x):
        return self._linear(x)

def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

# create network
layer = LinearNet()
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
train(layer, loader, loss_fn, adam)
```


#### 2.2.2 方式二：使用 ``@paddle.jit.to_static`` 函数

除了装饰器的方式外，也可以在组网后，增加一行 ``layer = paddle.jit.to_static(layer)`` 就可以将动态图网络转为静态图网络。

```python
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)

def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

# create network
layer = LinearNet()
layer = paddle.jit.to_static(layer) # <----通过函数式调用 paddle.jit.to_static(layer) 一键实现动转静
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
train(layer, loader, loss_fn, adam)
```

+ 方式一在组网代码中 ``forward()`` 函数定义处装饰，意指将其转为静态图执行模式；

+ 方式二在组网代码后，对整个网络 ``Layer`` 对象调用了 ``paddle.jit.to_static()`` 方法，默认也是对 ``Layer.forward()`` 进行动静转换；

+ 两种方式均实现了动转静训练，最终实现的效果是一样。


**值得注意的是：请确保被装饰的 ``Layer.forward`` 方法中仅实现预测功能，避免将训练所需的 loss 计算逻辑写入 forward 方法。**

Layer 更准确的语义是描述一个具有预测功能的模型对象，接收输入的样本数据，输出预测的结果，而 loss 计算是仅属于模型训练中的概念。将 loss 计算的实现放到 ``Layer.forward`` 方法中，会使 Layer 在不同场景下概念有所差别，并且增大 Layer 使用的复杂性，这不是良好的编码行为，同时也会在最终保存预测模型时引入剪枝的复杂性，因此建议保持 Layer 实现的简洁性。

错误示例如下：

```python
import paddle
import paddle.nn as nn

IMAGE_SIZE = 784
CLASS_NUM = 10

class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static
    def forward(self, x, label=None):
        out = self._linear(x)
        # 不规范写法，forward 中包括对 loss 进行计算
        if label:
            loss = nn.functional.cross_entropy(out, label)
            avg_loss = nn.functional.mean(loss)
            return out, avg_loss
        else:
            return out
```

正确示例如下：

```python
import paddle
import paddle.nn as nn

IMAGE_SIZE = 784
CLASS_NUM = 10

class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
    # 规范写法，forward 中仅实现预测功能
    @paddle.jit.to_static
    def forward(self, x):
        return self._linear(x)
```


### 2.3 动转静训练执行流程

动转静训练的基本执行流程如下图：

只对``Layer``的``sub_layer2``子图进行动转静训练。动态图执行在进行正向传播的时，动态图是根据每一层的执行即时动态变化的，并自动记录对应的反向传播所需的信息，在调用``loss.backward()``时，进行反向传播计算梯度，优化器更新模型参数。对于动态图训练的正向传播和反向传播是分离的两个操作。对于静态子图训练，``sub_layer2``子图动转静，确定图结构，获取整个子图的信息进行优化，创建 [Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/Executor_cn.html#cn-api-paddle-static-executor) 执行器对整个静态子图进行训练。在动转静训练时，执行整个静态子图的``run_program_op``进行动转静训练。


<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/jit/images/to_static_train.png" style="zoom:50%" />
</figure>


## 三、动转静模型保存和加载

采用动态图模式将模型训练好后，如果需要将模型的结构和参数持久化保存到磁盘文件中，用于后续推理部署，则可以使用 [paddle.jit.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/save_cn.html#save) 实现动转静模型保存。保存的模型可使用 [paddle.jit.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/load_cn.html#load) 载入用于验证推理效果。


### 3.1 保存和加载机制介绍

如下图所示，动态图模式下，模型结构指的是 Python 前端组网代码；模型参数指的是 `model.state_dict()` 中存放的权重数据。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/jit/images/export.png" style="zoom:100%" />
</figure>

使用 ``paddle.jit.save`` 保存模型，通常是在后台执行了两个步骤：

1. 先执行了动转静。当然如果前面已经执行了动转静训练，则跳过这一步。在处理逻辑上，主要包含两个主要模块：
    + 模型结构层面：将动态图模型中被 ``@paddle.jit.to_static`` 装饰的函数转化为完整的静态图 Program。

    + 模型参数层面：将动态图模型中的参数（Parameters 和 Buffers ）转为 ``Persistable=True``  的静态图模型参数 Variable。

2. 再将静态图模型和参数导出为磁盘文件。Program 和 Variable 都可以直接序列化导出为磁盘文件，与前端代码完全解耦，导出的文件包括：
    + 后缀为 ``.pdmodel`` 的模型结构文件；

    + 后缀为 ``.pdiparams`` 的模型参数文件；

    + 后缀为 ``.pdiparams.info`` 的和参数状态有关的额外信息文件。

类似的，使用 ``paddle.jit.load`` 加载模型，即将上述三个文件加载为静态图模型的 Program 和 Variable，可用于执行静态图模式下训练调优或验证推理效果。

无论是动态图训练，还是动转静训练，都支持通过 ``paddle.jit.save`` 自动转为静态图模型，只是使用过程中有一些配置差异，下面通过示例介绍。

### 3.2 动转静训练后模型保存和加载样例

#### 3.2.1 模型保存样例

接前文动转静训练的示例代码，训练完成后，使用 ``paddle.jit.save`` 对模型和参数进行存储：

```python
# 如果保存模型用于推理部署，则需切换 eval()模式
# layer.eval()
# 使用 paddle.jit.save 保存训练好的静态图模型
path = "example.model/linear"
paddle.jit.save(layer, path)
```


> 注：由于类似 Dropout 、LayerNorm 等接口在 train() 和 eval() 状态的行为存在较大的差异，在模型导出前，请务必确认模型已切换到正确的模式，否则导出的模型在预测阶段可能出现输出结果不符合预期的情况。
> + 如果保存模型用于推理部署，需切换到 eval() 模式；
> + 如果保存模型用于后续训练调优，则切换到 train() 模式；


执行上述代码样例后，会在当前目录下生成三个文件，即代表成功导出静态图模型：

```
linear.pdiparams        // 存放模型中所有的权重数据
linear.pdmodel         // 存放模型的网络结构
linear.pdiparams.info   // 存放和参数状态有关的额外信息
```

导出的模型可用于在云、边、端不同的硬件环境中部署，可以支持不同语言环境部署，如 C++、Java、Python 等。飞桨提供了服务器端部署的 Paddle Inference、移动端/IoT 端部署的 Paddle Lite、服务化部署的 Paddle Serving 等，以实现模型的快速部署上线。具体介绍可参见 [推理部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/infer/index_cn.html) 章节。


#### 3.2.2 模型加载样例

动转静训练保存模型后，如果需要再加载用于训练调优或验证推理效果，可以选择使用 [paddle.jit.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/load_cn.html#load) 或 [paddle.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/load_cn.html#load) API。

+ **使用 ``paddle.jit.load`` 加载**：该方式可以载入模型结构和参数，传入数据即可训练或推理。
+ **使用 ``paddle.load`` 加载**：如果已有组网代码，则只传入模型参数也可再训练，因此也可以选择该方式加载。

##### 3.2.2.1 使用 ``paddle.jit.load`` 加载

使用 ``paddle.jit.load`` 载入，载入后得到的是一个 Layer 的派生类对象 TranslatedLayer ， TranslatedLayer 具有 Layer 的通用特征，可以进行模型调优。

> 注意：使用 ``paddle.jit.load`` 载入模型，如果要用于训练调优，在 ``paddle.jit.save`` 的时候不能切换成 eval() 模式进行保存。另外，为了规避变量名字冲突，载入之后会重命名变量。

示例代码如下：

```python
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# 载入 paddle.jit.save 保存的模型
path = "example.model/linear"
loaded_layer = paddle.jit.load(path)
```

载入模型及参数后进行预测，示例如下（接上文示例）：

```python
# 执行预测
loaded_layer.eval()
x = paddle.randn([1, IMAGE_SIZE], 'float32')
pred = loaded_layer(x)
```

载入模型及参数后进行调优（fine-tune），示例如下（接上文示例）：

```python
# 定义一个随机数据集
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

# 对载入后的模型进行训练调优
loaded_layer.train()
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
train(loaded_layer, loader, loss_fn, adam)
# 训练调优后再次保存
paddle.jit.save(loaded_layer, "fine-tune.model/linear", input_spec=[x])
```

##### 3.2.2.2 使用 ``paddle.load`` 加载

``paddle.jit.save`` 同时保存了模型和参数，如果已有组网代码，只需要从存储结果中载入模型的参数，则可以使用 ``paddle.load`` 接口载入，返回所存储模型的 ``state_dict`` ，并使用 ``set_state_dict`` 方法将模型参数与 Layer 关联。示例如下：

```python
import paddle
import paddle.nn as nn

IMAGE_SIZE = 784
CLASS_NUM = 10

# 网络定义
class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static
    def forward(self, x):
        return self._linear(x)

# 创建一个网络
layer = LinearNet()

# 载入 paddle.jit.save 保存好的参数
path = "example.model/linear"
state_dict = paddle.load(path)

# 将加载后的参数赋给 layer 并进行预测
layer.set_state_dict(state_dict, use_structured_name=False)
layer.eval()
x = paddle.randn([1, IMAGE_SIZE], 'float32')
pred = layer(x)
```

### 3.3 动态图训练后模型保存和加载样例

结合一个简单的网络训练示例，介绍动态图训练后，转为静态图模型保存的方法。先用动态图模式训练一个模型：

```python
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# 定义一个随机数数据集
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples
# 定义神经网络
class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)
# 定义训练过程
def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

# 构建神经网络
layer = LinearNet()
# 设置损失函数
loss_fn = nn.CrossEntropyLoss()
# 设置优化器
adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# 构建 DataLoader 数据读取器
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# 开始训练
train(layer, loader, loss_fn, adam)
```

#### 3.3.1 模型保存样例

动态图模型训练完成后，保存为静态图模型用于推理部署，主要包括三个步骤：

1. **切换 ``eval()`` 模式：** 类似 Dropout 、LayerNorm 等接口在 train() 和 eval() 的行为存在较大的差异，在模型导出前，请务必确认模型已切换到正确的模式，否则导出的模型在预测阶段可能出现输出结果不符合预期的情况。用于推理部署切换到 eval()模式，用于后续训练调优则切换到 train()模式。

2. **构造 ``InputSpec`` 信息：** ``InputSpec`` 用于表示模型输入数据的 shape、dtype、name 信息，是辅助动静转换的必要描述信息。这是由于静态图模型在调用执行器前并不执行实际操作，因此也并不读入实际数据，需要设置 “占位符” 表示输入数据。详细请参见 [InputSpec 的用法介绍](#35) 。

3. **调用 ``save`` 接口：** 调用 ``paddle.jit.save`` 接口，若传入的参数是类实例（如示例中的 Layer 类），则框架会默认对神经网络中定义的前向计算 forward 函数进行 ``@paddle.jit.to_static`` 装饰（执行动转静），并导出其对应的模型文件和参数文件。


    ```python
    from paddle.static import InputSpec
    # 1.切换 eval()模式
    layer.eval()
    # 2. 构造 InputSpec 信息
    input_spec = InputSpec([None, 784], 'float32', 'x')
    # 3.调用 paddle.jit.save 接口转为静态图模型
    path = "example.dy_model/linear"
    paddle.jit.save(
        layer=layer,
        path=path,
        input_spec=[input_spec])
    ```

执行上述代码样例后，会在当前目录下生成三个文件，即代表成功导出可用于推理部署的静态图模型：

```
linear.pdiparams        // 存放模型中所有的权重数据
linear.pdmodel         // 存放模型的网络结构
linear.pdiparams.info   // 存放和参数状态有关的额外信息
```


#### 3.3.2 模型加载样例

动态图训练保存模型后，模型加载通常就是用于验证推理效果，使用 ``paddle.jit.load`` 载入。

示例代码如下：

```python
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# 载入 paddle.jit.save 保存的模型
path = "example.dy_model/linear"
loaded_layer = paddle.jit.load(path)
```

载入模型及参数后进行预测，示例如下：

```python
# 执行预测
loaded_layer.eval()
x = paddle.randn([1, IMAGE_SIZE], 'float32')
pred = loaded_layer(x)
```

### 3.4 模型保存和加载注意事项

1. 由于调用 ``paddle.jit.save`` 接口实际隐含了动转静的动作，因此动态图训练后可以不需要通过加装饰器 ``@paddle.jit.to_static`` 或 ``paddle.jit.to_static()`` 函数的方式将动态图转为静态图，直接调用 ``paddle.jit.save`` 接口即可。

2. 动态图训练场景下，使用 ``paddle.jit.save`` 保存模型时，必须指定 Layer 的 ``InputSpec`` ，Layer 对象 forward 方法的每一个参数均需要对应的 ``InputSpec`` 描述，不能省略。

3. 动转静训练场景下，通常不用指定 ``InputSpec`` ，这是因为飞桨框架会自动根据训练时实际输入 Tensor 的 shape、dtype 等信息创建对应的 ``InputSpec`` ，不需要再指定。但是若模型的输入数据是动态输入（如输入数据的 shape 有一维度是可变的，该维度需设置为 None 或 -1），这种情况下则需要在 ``@paddle.jit.to_static`` 接口中指明模型的 ``InputSpec`` ，示例如下：

    ```python
    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    # 网络定义
    class LinearNet(nn.Layer):
        def __init__(self):
            super().__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
        # 输入数据是动态输入（shape 中有一个维度是可变的），因此需要添加 InputSpec
        @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
        def forward(self, x):
            return self._linear(x)
    ```

4. 前文中大多都是用 ``paddle.jit.save`` 保存的 ``Layer.forward`` 类实例，保存内容包括模型结构和参数。当保存单独的一个函数时， ``paddle.jit.save`` 只会保存这个函数对应的静态图模型结构 Program ，不会保存和这个函数相关的参数。如果必须保存参数，请使用 Layer 类封装这个函数。 示例代码如下：

    ```python
    # 定义一个函数
    def fun(inputs):
        return paddle.tanh(inputs)

    path = 'func/model'
    inps = paddle.rand([3, 6])
    origin = fun(inps)

    # 将函数对应的 Program 结构进行保存
    paddle.jit.save(
        fun,
        path,
        input_spec=[
            InputSpec(
                shape=[None, 6], dtype='float32', name='x'),
        ])

    # 载入保存后的 fun 并执行
    load_func = paddle.jit.load(path)
    load_result = load_func(inps)
    ```

5. 动转静训练时，如果想保存多个函数，则需要用 ``@paddle.jit.to_static`` 装饰每一个待保存的函数。示例代码如下：

    ```python
    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    class LinearNet(nn.Layer):
        def __init__(self):
            super().__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
            self._linear_2 = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        # 装饰 forward 方法，InputSpec 指定为 None
        @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
        def forward(self, x):
            return self._linear(x)

        # 装饰需要保存的非 forward 方法,InputSpec 指定为 None
        @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
        def another_forward(self, x):
            return self._linear_2(x)

    inps = paddle.randn([1, IMAGE_SIZE])
    layer = LinearNet()
    before_0 = layer.another_forward(inps)
    before_1 = layer(inps)
    # 模型保存
    path = "example.model/linear"
    paddle.jit.save(layer, path)
    ```

    > 注：只有在 forward 之外还需要保存其他函数时才用这个特性，如果仅装饰非 forward 函数，而 forward 本身函数没有被装饰，是不符合规范的。当保存多个函数时， ``InputSpec`` 信息需要在各个函数的 ``@paddle.jit.to_static`` 里分别指定，并且 ``input_spec`` 参数必须为 None，因为此时 save 接口 input_spec 参数无法知道它应该配置给哪个函数。

  + 该场景下保存的模型命名规则如下：

    + forward 的模型名字为：**模型名+后缀** ，其他函数的模型名字为：**模型名+函数名+后缀** 。每个函数有各自的 pdmodel 和 pdiparams 的文件，所有函数共用 `pdiparams.info` 。上述示例代码将在 `example.model` 文件夹下产生 5 个文件： ``linear.another_forward.pdiparams`` 、 ``linear.pdiparams`` 、 ``linear.pdmodel`` 、 ``linear.another_forward.pdmodel`` 、``linear.pdiparams.info`` 。


### 3.5 ``InputSpec`` 的用法介绍
<span id='35'></span>

``InputSpec`` 用于表示模型输入数据的 shape、dtype、name 信息，是辅助动静转换的必要描述信息。

在静态图模式下，飞桨框架会将神经网络描述为 Program 的数据结构，并对 Program 进行编译优化，再调用执行器获得计算结果。可以看到静态图模式下运行，在调用执行器前并不执行实际操作（这个阶段一般称为“组网阶段”或者“编译阶段”），因此也并不读入实际数据，所以在静态图中还需要一种特殊的变量来表示输入数据，一般称为“占位符”，动转静提供了 ``InputSpec`` 接口配置该“占位符”，用于表示输入数据的描述信息。

另外在无法确定输入数据维度时，可以通过 ``InputSpec`` 将相应维度指定为 None 来表示动态 shape（如输入的 batch_size 维度）。

#### 3.5.1 构造 ``InputSpec``

**（1）方式一：直接构造**

``InputSpec`` 接口在 ``paddle.static`` 目录下， 只有 shape 是必须参数， dtype 和 name 可以缺省，默认取值分别为 float32 和 None 。使用样例如下：

```python
from paddle.static import InputSpec

x = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')

print(x)      # InputSpec(shape=(-1, 784), dtype=VarType.FP32, name=x)
print(label)  # InputSpec(shape=(-1, 1), dtype=VarType.INT64, name=label)
```

**（2）方式二：由 Tensor 构造**

可以借助 ``InputSpec.from_tensor`` 方法，从一个 Tensor 直接创建 ``InputSpec`` 对象，其拥有与源 Tensor 相同的 shape 和 dtype 。 使用样例如下：

```python
import numpy as np
import paddle
from paddle.static import InputSpec

x = paddle.to_tensor(np.ones([2, 2], np.float32))
x_spec = InputSpec.from_tensor(x, name='x')
print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)
```


> 注：若未在 from_tensor 中指定新的 name，则默认使用与源 Tensor 相同的 name。


**（3）方式三：由 Example Tensor 列表构造**

可以通过训练时的输入数据构造，使用 forward 训练时的示例输入得到输入数据的 shape、dtype 等信息，比如使用数据加载器 DataLoader 得到的输入 ``image`` 来构造，示例如下：

```python
import paddle
from paddle.static import InputSpec
# 省略动态图训练代码
# 保存时将输入数据传入 input_spec 参数
paddle.jit.save(
    layer=layer,
    path=path,
    input_spec=[image])
```


**（4）方式四：由 numpy.ndarray 构造**

也可以借助 ``InputSpec.from_numpy`` 方法，从一个 ``Numpy.ndarray`` 直接创建 InputSpec 对象，其拥有与源 ndarray 相同的 shape 和 dtype 。使用样例如下：

```python
import numpy as np
from paddle.static import InputSpec

x = np.ones([2, 2], np.float32)
x_spec = InputSpec.from_numpy(x, name='x')
print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)
```

> 注：若未在 from_numpy 中指定新的 name，则默认使用 None 。


#### 3.5.2 基本用法


**（1）方式一： 在 ``@to_static`` 装饰器中调用**

如下是一个简单的使用样例：

```python
import paddle
from paddle.nn import Layer
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)
    # 在装饰器中调用 InputSpec
    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()

# save static graph mode for inference directly
paddle.jit.save(net, './simple_net')
```

在上述的样例中， ``@to_static`` 装饰器中的 ``input_spec`` 为一个 InputSpec 对象组成的列表，用于依次指定参数 x 和 y 对应的 Tensor 签名信息。在实例化 SimpleNet 后，可以直接调用 ``paddle.jit.save`` 保存静态图模型，不需要执行任何其他的代码。

> 注：
> 1. input_spec 参数中不仅支持 InputSpec 对象，也支持 int 、 float 等常见 Python 原生类型。
> 2. 若指定 input_spec 参数，则需为被装饰函数的所有必选参数都添加对应的 InputSpec 对象，如上述样例中，不支持仅指定 x 的签名信息。
> 3. 若被装饰函数中包括非 Tensor 参数，推荐函数的非 Tensor 参数设置默认值，如 ``forward(self, x, use_bn=False)``


**（2）方式二：在 ``to_static`` 函数中调用**

若期望在动态图下训练模型，在训练完成后保存预测模型，并指定预测时需要的签名信息，则可以选择在保存模型时，直接调用 to_static 函数。使用样例如下：

```python
class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()

# train process (Pseudo code)
for epoch_id in range(10):
    train_step(net, train_reader)
# 在 paddle.jit.to_static 函数中调用 InputSpec
net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])

# save static graph model for inference directly
paddle.jit.save(net, './simple_net')
```

如上述样例代码中，在完成训练后，可以借助 ``to_static(net, input_spec=...)`` 形式对模型实例进行处理。飞桨框架会根据 input_spec 信息对 forward 函数进行递归的动转静处理，得到完整的静态图，且包括当前训练好的参数数据。


**（3）方式三：通过 list 和 dict 推导**

上述两个样例中，被装饰的 forward 函数的参数均为 Tensor 。这种情况下，参数个数必须与 InputSpec 个数相同。但当被装饰的函数参数为 list 或 dict 类型时，input_spec 需要与函数参数保持相同的嵌套结构。

当函数的参数为 list 类型时，input_spec 列表中对应元素的位置，也必须是包含相同元素的 InputSpec 列表。使用样例如下：

```python

class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static(input_spec=[[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')]])
    def forward(self, inputs):
        x, y = inputs[0], inputs[1]
        out = self.linear(x)
        out = out + y
        return out
```

其中 input_spec 参数是长度为 1 的 list ，对应 forward 函数的 inputs 参数。 input_spec[0] 包含了两个 ``InputSpec`` 对象，对应于参数 inputs 的两个 Tensor 签名信息。

当函数的参数为 dict 类型时， input_spec 列表中对应元素的位置，也必须是包含相同键（key）的 InputSpec 列表。使用样例如下：

```python
class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), {'x': InputSpec(shape=[3], name='bias_info')}])
    def forward(self, x, bias_info):
        x_bias = bias_info['x']
        out = self.linear(x)
        out = out + x_bias
        return out
```

其中 input_spec 参数是长度为 2 的 list ，对应 forward 函数的 x 和 bias_info 两个参数。 input_spec 的最后一个元素是包含键名为 x 的 ``InputSpec`` 对象的 dict ，对应参数 bias_info 的 Tensor 签名信息。


**（4）方式四：指定非 Tensor 参数类型**

若被装饰函数的参数列表除了 Tensor 类型，还包含其他如 Int、 String 等非 Tensor 类型时，推荐在函数中使用 kwargs 形式定义非 Tensor 参数，如下述样例中的 ``use_act`` 参数。

```python
class SimpleNet(Layer):
    def __init__(self, ):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)
        self.relu = paddle.nn.ReLU()

    def forward(self, x, use_act=False):
        out = self.linear(x)
        if use_act:
            out = self.relu(out)
        return out

net = SimpleNet()
# 方式一：save inference model with use_act=False
net = to_static(input_spec=[InputSpec(shape=[None, 10], name='x')])
paddle.jit.save(net, path='./simple_net')

# 方式二：save inference model with use_act=True
net = to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), True])
paddle.jit.save(net, path='./simple_net')
```

在上述样例中，假设 step 为奇数时， ``use_act`` 取值为 False ； step 为偶数时， ``use_act`` 取值为 True 。动转静支持非 Tensor 参数在训练时取不同的值，且保证了取值不同的训练过程都可以更新模型的网络参数，行为与动态图一致。

在借助 ``paddle.jit.save`` 保存预测模型时，动转静会根据 input_spec 和 kwargs 的默认值保存推理模型和网络参数。建议将 ``kwargs`` 参数默认值设置为预测时的取值。



## 四、动转静更多用法

### 4.1 动转静训练计算图优化策略
<span id='41'></span>

如下是一个 ResNet50 模型动转静训练时，通过在 ``to_static`` 函数中配置 ``build_strategy`` 参数来开启算子融合 ``fuse_elewise_add_act_ops`` 和 ``enable_addto`` 图优化策略的使用样例。不同的模型可应用的优化策略不同，比如算子融合策略一般与模型中用到的 API 有关系：

+ 若存在 elementwise_add 后跟 relu 等激活函数，则可以尝试开启 ``fuse_elewise_add_act_ops``

+ 若存在 relu 后跟 depthwise_conv2 函数，则可以尝试开启 ``fuse_relu_depthwise_conv``

+ 若存在较多的 conv2dAPI 的调用，则可以尝试开启 ``enable_addto`` ，更多策略开关可以参考 [BuildStrategy](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/BuildStrategy_cn.html#buildstrategy) 接口文档。


    ```python
    import os
    import paddle
    import numpy as np
    import paddle.optimizer as opt
    from paddle import nn
    from paddle.vision.models import resnet50

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 224
    CLASS_NUM = 1000

    # 定义一个随机数数据集
    class RandomDataset(paddle.io.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([3, IMAGE_SIZE, IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, CLASS_NUM, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    # 定义训练过程
    def train(layer, loader, loss_fn, opt):
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (images, labels) in enumerate(loader()):
                out = layer(images)
                loss = loss_fn(out, labels)
                loss.backward()
                opt.step()
                opt.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy())))

    def get_build_strategy():
        build_strategy = paddle.static.BuildStrategy()
        # addto 策略常搭配 FLAGS_max_inplace_grad_add 变量使用
        build_strategy.enable_addto = True
        os.environ['FLAGS_max_inplace_grad_add'] = "8"
        build_strategy.fuse_elewise_add_act_ops = True
        return build_strategy

    # 构建神经网络
    model = resnet50()
    # 动转静，并设置计算图优化策略
    model = paddle.jit.to_static(model, build_strategy=get_build_strategy())
    # 设置损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 设置优化器
    adam = opt.Adam(learning_rate=0.001, parameters=model.parameters())

    # 构建 DataLoader 数据读取器
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # 开始训练
    train(model, loader, loss_fn, adam)
    ```

### 4.2 动转静训练开启 AMP

**自动混合精度（Automatic Mixed Precision，AMP）** 训练的方法，可在模型训练时，自动为算子选择合适的数据计算精度（float32 或 float16 / bfloat16），在保持训练精度（accuracy）不损失的条件下，能够加速训练。

如下是动态图开启 AMP 训练后，执行单个 step 的 timeline 示意图。相对于 FP32 训练，开启 AMP 后，每个 GPU 的 Kernel 计算效率进一步提升，耗时更短（图中蓝框更窄了），但对 CPU 端的调度性能要求也更高了。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/jit/images/timeline_d2s_amp.png" style="zoom:70%" />
</figure>


由于动转静后，转为静态图训练可以有效地减少 Python 与 C++ 端交互的开销，减少框架调度时间，提升训练性能。经验证，在 Bert Base 和 Transformer Base 模型 AMP 训练任务上测试，动转静训练后性能有 20% 左右的提升。因此在 AMP 训练任务上，推荐尝试开启动转静实现训练加速。开启 AMP 的具体用法请参见 [自动混合精度训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/performance_improving/amp_cn.html) 章节。


## 五、总结

飞桨框架动转静模块的设计，主要是为了兼顾动态图的易用性和静态图的高性能。有了动转静模块，开发者并不需要同时深入掌握动态图和静态图的编程方法，只需要掌握动态图的使用，只需添加一行装饰器 ``@paddle.jit.to_static`` ，即可转为静态图训练，飞桨框架在后台帮助完成了动转静，前台仍然保持动态图的使用习惯即可。总结动转静的主要用法：

+ 动转静训练：即动态图编码，转静态图训练。只需通过 ``@paddle.jit.to_static`` 装饰模型 Layer 类实例的 forward 函数即可实现。
+ 动转静模型保存和加载：既支持动转静训练后模型保存和加载，也支持动态图编码和训练，直接转为静态图模型文件，用于模型推理部署。直接使用 ``paddle.jit.save`` 保存即可，飞桨框架自动完成了动转静和保存操作。
