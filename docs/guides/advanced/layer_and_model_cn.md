# 使用 paddle.nn.Layer 自定义网络

为了更灵活地构建特定场景的专属深度学习模型，飞桨提供了 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 系列接口，以便用户轻松地定义专属的深度学习模型。

为了充分利用它们，并根据实际需求进行量身定制，需要真正理解它们到底在做什么。为了加深这种理解，我们将首先在 MNIST 数据集上训练基本的神经网络，不使用这些模型的任何特征，同时采用最基本的飞桨 Tensor 功能进行模型开发。然后，我们将逐步从 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 中添加一个特征，展示如何使用飞桨的 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html)系列接口进行模型、层与参数的设计，来开发一个用户专属的深度学习模型。

在具体操作之前，让我们先了解与之相关的基本概念。

## 一、概念介绍

**1. 模型**

模型的核心功能是将一组输入变量经过一系列计算，映射到另一组输出变量，通常为带参数的函数，该函数代表一种算法。深度学习的目标就是学习一组最优的参数使得模型的预测最“准确”。在飞桨框架中，模型包括以下两方面内容：

- 一系列层的组合，用于输入到输出的映射（前向计算）
- 一些参数变量，在训练过程中实时更新

**2. 层**

飞桨大多数模型由一系列层组成。层是模型的基础逻辑执行单元。层包含以下两方面内容：

- 一个或多个具体的算子，用于完成相应的计算
- 计算所需的变量，以临时变量或参数的形式作为层的成员存在

**3. paddle.nn.Layer**

从零开始构建变量、算子，并组建层以及模型，是一个很复杂的过程，难免出现很多冗余代码，因此飞桨提供了基础数据类型 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) ，方便开发者继承并扩展。

paddle.nn.Layer 是飞桨定义的一个非常重要的类，是飞桨所有神经网络模块的基类， 它代表所有可以用层表示的网络结构，包含网络各层的定义及前向计算方法。除此之外，飞桨还基于 Layer 定义了各种常用的层，比如卷积，池化，Padding，激活，Normalization，循环神经网络，Transformer 相关，线性，Dropout，Embedding，Loss，Vision，Clip，公共层等等（paddle.nn 包中的各个类均继承 paddle.nn.Layer 这个基类），详情请参考[组网相关的 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html)。



> 说明：
> 本教程基于[基于手写数字识别（MNIST）任务](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/quick_start_cn.html)作为样板代码进行说明，通过本节的学习，用户将进一步掌握使用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 改进模型、层与参数的方法。

以下内容假定你已经完成了飞桨的安装以及熟悉了一些基本的飞桨操作。

## 二、数据处理

### 2.1 加载 Mnist 数据集

相信根据前面的内容，你已经知道如何使用 [paddle.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Dataset_cn.html) 和 [paddle.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html) 处理想要的数据了，如果你还有问题可以参考[数据集定义与加载](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/data_load_cn.html)文档，这里采用前面讲到的方法使用 Mnist 数据集。

```python
import paddle
import math
from paddle.vision.transforms import Compose, Normalize

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
```



### 2.2 对数据集进行预处理

为演示方便先从这个训练集中取出一条数据，简单测试下后面搭建的网络，同时为了方便训练对该数据进行形状的变换。

当然，实际过程中需要通过一个循环不断获取 train_dataset 中的数据，不间断的进行训练。

```python
train_data0 = train_dataset[0]
x_data = paddle.to_tensor(train_data0[0])
x_data = paddle.flatten(x_data, start_axis=1)
print("x_data's shape is:", x_data.shape)
```

```python
x_data's shape is: [1, 784]
```

## 三、搭建一个完整的深度学习网络

接下来仅利用飞桨最基本的 Tensor 功能快速完成一个深度学习网络的搭建。

### 3.1 参数初始化

首先， 需要通过 [paddle.randn](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randn_cn.html) 函数或者 [paddle.zeros](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/zeros_cn.html) 函数来创建随机数填充或者全零填充的一个参数（Weight）（模型训练中会被更新的部分），和一个偏置项（Bias）。

注意：这里可通过 Xavier (XavierInitializer 的别名) 初始化方式初始化参数，即对产生的随机数除以 sqrt（n）(n 是第零维的大小)。

```python
weight = paddle.randn([784, 10]) * (1/math.sqrt(784))
weight.stop_gradient=False
bias = paddle.zeros(shape=[10])
bias.stop_gradient=False
```

参数初始化完成后，就可以开始准备神经网络了。

### 3.2 准备网络结构

网络结构是深度学习模型关键要素之一，相当于模型的假设空间，即实现模型从输入到输出的映射过程（前向计算）。

本文利用一个矩阵乘法和一个加法（飞桨的加法可以自己完成 broadcast，就像 numpy 一样好用）来实现一个简单的 Linear 网络。通常，还需要一些激活函数（例如 log_softmax）来保证网络的非线性。请注意，虽然飞桨提供了大量已实现好的激活函数，你也可以利用简单的 Python 代码来完成自己的激活函数。这些 Python 代码都将会被飞桨识别从而变成高效的设备代码被执行。

```python
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(x):
    return log_softmax(paddle.matmul(x, weight) + bias)
```

### 3.3 前向计算

通常训练都是针对一个 batch 进行的，可以从之前的数据中按照 batch_size=64 取出一部分进行一轮的前向执行。由于本轮利用随机初始化的参数进行前向计算，那么计算的结果也就和一个随机的网络差不多。

```python
batch_size = 64
train_batch_data_x = []
train_batch_data_y = []
for i in range(batch_size):
    train_batch_data_x.append(train_dataset[i][0])
    train_batch_data_y.append(train_dataset[i][1])

x_batch_data = paddle.to_tensor(train_batch_data_x)
x_batch_data = paddle.flatten(x_batch_data, start_axis=1)
print("x_batch_data's shape is:", x_batch_data.shape)

y = model(x_batch_data)

print("y[0]: {} \ny.shape: {}".format(y[0], y.shape))
```

```python
x_data's shape is: [1, 784]
x_batch_data's shape is: [64, 784]
y[0]: Tensor(shape=[10], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [-1.20662355, -4.20237827, -2.47686505, -0.78191900, -5.13888979,
        -3.07260418, -2.94610834, -4.91643810, -3.71131158, -4.85082626])
y.shape: [64, 10]
```

### 3.4 反向传播

这里我们会发现，y 的信息中包含一项 StopGradient=False。这意味着我们可以通过 y 来进行 BP（反向传播），同时可以定义自己的损失函数。以一个简单的 nll_loss 来演示，写法上如同写一段简单的 Python 代码。

```python
loss_func = paddle.nn.functional.nll_loss

y_batch_data = paddle.to_tensor(train_batch_data_y)
y_batch_data = paddle.flatten(y_batch_data, start_axis=1)
print("y_batch_data's shape is:", y_batch_data.shape)
y_standard = y_batch_data[0:batch_size]
loss = loss_func(y, y_standard)
print("loss: ", loss)
```

```python
loss:  Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [2.85819387])
```

### 3.5 计算 ACC 观察模型收敛情况

同样，也可以实现一个简单的计算 accuracy 的方法来验证模型收敛情况。

```python
def accuracy(out, y):
    preds = paddle.argmax(out, axis=-1, keepdim=True)
    return (preds == y).cast("float32").mean()

accuracy = accuracy(y, y_standard)
print("accuracy:", accuracy)
```

```python
accuracy: Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [0.09375000])
```

### 3.6 使用自动微分功能计算网络的梯度并更新参数

接下来我们将利用飞桨的自动微分功能计算网络的梯度，并且利用该梯度和参数完成一轮参数更新（需要注意的是，在更新参数的阶段我们并不希望进行微分的逻辑，只需要使用 [paddle.no_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/no_grad_cn.html) 禁用相关功能即可）。

```python
loss.backward()

@paddle.no_grad()
def OptimizeNetwork(lr=0.5):
    weight.set_value(weight - lr * weight.grad)
    bias.set_value(bias - lr * bias.grad)
    weight.clear_gradient()
    bias.clear_gradient()
print("weight: ", weight)
print("bias: ", bias)
OptimizeNetwork()
print("weight after optimize: ", weight)
print("bias after optimize: ", bias)
```

```python
weight:  Tensor(shape=[784, 10], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[-0.02580861,  0.03132926,  0.07240372, ...,  0.05494612,
         -0.03443871, -0.00228449],
        [-0.01263286, -0.03029860,  0.04301141, ...,  0.02060869,
         -0.00263721, -0.01837303],
        [ 0.02355293, -0.06277876, -0.03418431, ...,  0.03847973,
          0.02322033,  0.08055742],
        ...,
        [-0.02945464,  0.00892299, -0.07298648, ...,  0.04788664,
          0.03856503,  0.07544740],
        [ 0.06136639, -0.00014994,  0.00933051, ..., -0.00939863,
          0.06214209, -0.01135642],
        [-0.01522523, -0.04802566,  0.01832000, ...,  0.01538999,
          0.04224478,  0.01449125]])
bias:  Tensor(shape=[10], dtype=float32, place=Place(cpu), stop_gradient=False,
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
weight after optimize:  Tensor(shape=[784, 10], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[-0.05760278,  0.03702446,  0.06256686, ...,  0.13622762,
         -0.01372341, -0.04273041],
        [-0.04442703, -0.02460339,  0.03317455, ...,  0.10189019,
          0.01807809, -0.05881895],
        [-0.00824124, -0.05708356, -0.04402117, ...,  0.11976123,
          0.04393563,  0.04011151],
        ...,
        [-0.06124880,  0.01461819, -0.08282334, ...,  0.12916814,
          0.05928034,  0.03500149],
        [ 0.02957222,  0.00554527, -0.00050635, ...,  0.07188287,
          0.08285740, -0.05180233],
        [-0.04701940, -0.04233045,  0.00848314, ...,  0.09667149,
          0.06296009, -0.02595467]])
bias after optimize:  Tensor(shape=[10], dtype=float32, place=Place(cpu), stop_gradient=False,
       [ 0.03179417, -0.00569520,  0.00983686, -0.02128297,  0.00566411,
         0.02163870,  0.01959525, -0.08128151, -0.02071531,  0.04044591])
```

至此，就完成了一个简单的训练过程。我们会发现，需要定义大量的计算逻辑来完成这个组网过程，过程是很繁杂的。好在飞桨已经提供了大量封装好的 API，可以更简单的定义常见的网络结构，下面介绍具体的用法。



## 四、使用 paddle.nn.Layer 构建深度学习网络

[paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 是飞桨定义的一个类，它代表所有可以用层表示的网络结构。对本文前面这个例子来说，我们需要构建线性网络的参数 weight，bias，以及矩阵乘法，加法，log_softmax 也可以看成是一个层。换句话说 ，我们可以把任意的网络结构看成是一个层，层是网络结构的一个封装。

### 4.1 使用 Layer 改造线性层

首先，可以定义自己的线性层：

```python
class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter([784,10])
        self.bias = self.create_parameter([10], is_bias=True, default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, inputs):
        return paddle.nn.functional.log_softmax(paddle.matmul(inputs, self.weight) + self.bias)
```

和直接使用 python 代码不同，我们可以借助飞桨提供的 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 类实现一个基本的网络。我们可以通过继承的方式利用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 的各种工具。

那么通过继承 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 构建层有什么好处呢？

#### 4.1.1 子类调用父类的构造函数

首先，我们会发现，在这个继承的子类当中需要去调用一下父类的构造函数：

```python
    def __init__(self):
        super().__init__()
```

#### 4.1.2 完成一系列的初始化

这个时候飞桨会完成一系列初始化操作，其目的是为了记录所有定义在该层的状态，包括参数，call_back, 子层等信息。

```python
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter([784,10])
        self.bias = self.create_parameter([10], is_bias=True, default_initializer=paddle.nn.initializer.Constant(value=0.0))
```

### 4.2 访问并自动记录参数的更新过程

这里我们调用的 create_parameter 函数就来自于 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 类，这个函数帮助我们简单的创建并初始化参数。最简单的我们仅仅传入希望的参数形状即可（如 weight），这时候 create_parameter 会通过默认的方式初始化参数（默认是参数而不是 bias，使用 UniformRandom 来初始化参数，详情可以参考 create_parameter）；或者可以通过诸多参数来定义你自己希望的初始化参数的方式（如 bias），可以限定其初始化方式是全零的常数项（更多初始化方式可以参考 paddle.nn.initializer）。

完成参数初始化后，不同于我们直接使用 Python 时利用临时变量 weight 和 bias，这里可以利用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 自动将定义的参数记录下来，并且随时通过 self.named_parameters 访问。

```python
my_layer = MyLayer()
for name, param in my_layer.named_parameters():
    print("Parameters: {}, {}".format(name, param) )
```

```python
Parameters: weight, Parameter containing:
Tensor(shape=[784, 10], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [[-0.03399023, -0.02405306, -0.06372951, ..., -0.05039166,
          0.05060801,  0.05453540],
        [ 0.01788948, -0.06409007,  0.02617371, ...,  0.08341692,
         -0.01115795,  0.06199412],
        [-0.07155208,  0.01988612,  0.03681165, ..., -0.00741174,
          0.03892786,  0.03055505],
        ...,
        [-0.01735171, -0.05819885, -0.05768500, ...,  0.04783282,
          0.05039406, -0.04458937],
        [ 0.08272233,  0.02620430, -0.00838694, ...,  0.03075657,
         -0.05368494,  0.03899705],
        [-0.06041612, -0.05808754, -0.07175658, ..., -0.07276732,
          0.08097268, -0.00280717]])
Parameters: bias, Parameter containing:
Tensor(shape=[10], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

### 4.3 执行已定义的层

下面可以看看如何使用我们定义好的层。

#### 4.3.1 进入训练阶段并执行

首先, 我们通过构造函数构造了一个层，并且设置其执行模式为 train（训练）模式（通常你并不需要显式调用，因为默认是训练模式，这里仅仅为了演示），这样做是因为如 Dropout，BatchNorm 等计算，在训练和评估阶段的行为往往有区别，因此飞桨提供了方便的接口对整层设置该属性，如果层包含相关操作，可以通过这个设置改变他们在不同阶段的行为。

```python
my_layer = MyLayer()
my_layer.train()
# my_layer.eval()
y = my_layer(x_batch_data)
print("y[0]", y[0])
```

然后，可以将输入数据 x_batch_data 输入我们构建好的层对象，结果将被即时写在 y 当中。

```python
y[0] Tensor(shape=[10], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [-2.78626776, -2.75923157, -3.15698314, -2.98575473, -5.58894873,
        -5.03897095, -1.63698268, -0.70400816, -6.44660282, -2.51351619])
```

#### 4.3.2 计算 loss

同样调用 paddle.nn.functional.nll_loss 来计算 nll_loss。

```python
loss_func = paddle.nn.functional.nll_loss
y = my_layer(x_batch_data)
loss = loss_func(y, y_standard)
print("loss: ", loss)
```

#### 4.3.3 构建 SGD 优化器、参数传递及计算

与此同时，可以利用飞桨提供的 API 完成之前的操作。

例如，可以借助 [paddle.optimizer.SGD](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/SGD_cn.html) 构建一个优化器，并且通过 paddle.nn.Layer.parameters()获取所有需要优化的参数传入优化器，剩下的优化器计算事宜通过调用 opt.step()就可以交给飞桨来完成。

```python
my_layer = MyLayer()
opt = paddle.optimizer.SGD(learning_rate=0.5, parameters=my_layer.parameters())
loss_func = paddle.nn.functional.nll_loss
y = my_layer(x_batch_data)
loss = loss_func(y, y_standard)
print("loss: ", loss)
```

```python
loss.backward()
opt.step()
loss:  Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [2.76338077])
```

这样，我们就利用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 完成了网络的改造。可以发现，[paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 对大部分的网络场景提供了简单的网络状态控制和网络信息处理的方法。

### 4.4 使用 paddle.nn.Linear 改造预定义的层

此外，飞桨基于 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 构建了一系列层，这些层都可以通过简单的方式被复用在我们自定义网络中，上述例子中的 MyLayer 可以用飞桨定义的 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 来改造。

[paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 的改造主要包含替换线性层、调节参数初始化方式、改造前向传播及 softmax 等。

```python
class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(784, 10, bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0)))

    def forward(self, inputs):
        return paddle.nn.functional.log_softmax(self.linear(inputs))
```

可以看到，利用线性层替换了之前的矩阵乘法和加法（而这也正是线性层的定义）。只需要定义好自己的隐层大小，以及参数的初始化方式，就可以利用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 建立我们的线性层，此方式可节省自定义参数和运算的成本。

### 4.5 总结

至此，我们完成了如何用飞桨层的概念和 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 来完成一个简单的训练任务。可点击此[链接](https://aistudio.baidu.com/aistudio/projectdetail/4508657?contributionType=1)获取完整代码。

[paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 的功能远不止于此，利用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 还可以进行子层访问、层的成员变量操作、模型存储等操作，具体操作接下来会逐一介绍。



## 五、利用 paddle.nn.Layer 进行子层的访问

本节继续基于前面的手写数字识别任务，介绍如何使用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html) 进行子层的访问。

### 5.1 查看模型的所有层

如果想要访问或修改一个模型中定义的层，则可以调用**SubLayer**相关的接口。

以前面创建的简单模型为例,代码如下所示。

```python
mylayer = MyLayer()
print(mylayer.sublayers())

print("----------------------")

for item in mylayer.named_sublayers():
    print(item)
```

```python
[Linear(in_features=784, out_features=10, dtype=float32)]
----------------------
('linear', Linear(in_features=784, out_features=10, dtype=float32))
```

可以看到，通过调用 `mylayer.sublayers()` 接口，打印出了前述模型中持有的全部子层(这时模型中只有一个 [paddle.nn.Linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Linear_cn.html) 子层)。

而遍历 `mylayer.named_sublayers()` 时，每一轮循环会拿到一组 ( 子层名称('linear')，子层对象(paddle.nn.Linear) )的元组。

### 5.2 向模型添加一个子层

接下来如果想要进一步添加一个子层，则可以调用 `add_sublayer()` 接口。例如可以通过这个接口在前面做好的线性网络中再加入一个子层。

```python
my_layer = MyLayer()
fc = paddle.nn.Linear(10, 3)
my_layer.add_sublayer("fc", fc)
print(my_layer.sublayers())
```

```python
[Linear(in_features=784, out_features=10, dtype=float32), Linear(in_features=10, out_features=3, dtype=float32)]
```

可以看到 `my_layer.add_sublayer()` 向模型中添加了一个 10*3 的 [paddle.nn.Linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Linear_cn.html) 子层，这样模型中总共有两个 [paddle.nn.Linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Linear_cn.html) 的子层。

### 5.3 自定义函数并批量作用在所有子层

通过上述方法可以在模型中添加成千上万个子层。当模型中子层数量较多时，如何高效地对所有子层进行统一修改呢？Paddle 提供了 apply() 接口。通过这个接口，可以自定义一个函数，然后将该函数批量作用在所有子层上。

```python
def function(layer):
    print(layer)

my_layer.apply(function)
Linear(in_features=784, out_features=10, dtype=float32)
Linear(in_features=10, out_features=3, dtype=float32)
```

```python
MyLayer(
  (linear): Linear(in_features=784, out_features=10, dtype=float32)
  (fc): Linear(in_features=10, out_features=3, dtype=float32)
)
```

当前例子，定义了一个以 layer 作为参数的函数 function，用来打印传入的 layer 信息。通过调用 model.apply() 接口，将 function 作用在模型的所有子层中，输出信息打印 model 中所有子层的信息。

### 5.4 循环访问所有子层

另外一个批量访问子层的接口是 children() 或者 named_children() 。这两个接口通过 Iterator 的方式访问每个子层。

```python
my_layer = MyLayer()
fc = paddle.nn.Linear(10, 3)
my_layer.add_sublayer("fc", fc)
sublayer_iter = my_layer.children()
for sublayer in sublayer_iter:
    print(sublayer)
```

```python
Linear(in_features=784, out_features=10, dtype=float32)
Linear(in_features=10, out_features=3, dtype=float32)
```

可以看到，遍历 model.children() 时，每一轮循环都可以按照子层注册顺序拿到对应 paddle.nn.Layer 的对象。

## 六、修改 paddle.nn.Layer 层的成员变量

### 6.1 批量添加参数变量

和我们在前面演示的一样，你可以通过 create_parameter 来为当前层加入参数，这对于只有几个参数的层是比较简单的。但是，当我们需要很多参数的时候就比较麻烦了，尤其是希望使用一些 container 来处理这些参数，这时候就需要使用 add_parameter，让层感知需要增加的参数。

```python
class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        for i in range(10):
            self.add_parameter("param_" + str(i), self.create_parameter([784,10]))
    def forward(inputs):
        pass

my_layer = MyLayer()
for name, item in my_layer.named_parameters():
    print(name)
```

### 6.2 添加临时中间变量

刚刚的 Minst 的例子中，仅仅使用参数 weight，bias。参数变量往往需要参与梯度更新，但很多情况下只是需要一个临时变量甚至一个常量。比如在模型执行过程中想将一个中间变量保存下来，这时需要调用 create_tensor() 接口。

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.saved_tensor = self.create_tensor(name="saved_tensor0")
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(10, 100)

    def forward(self, input):
        y = self.flatten(input)
        # Save intermediate tensor
        paddle.assign(y, self.saved_tensor)
        y = self.fc(y)
        return y
```

这里调用 `self.create_tensor()` 创造一个临时变量，并将其记录在模型的 `self.saved_tensor` 中。在模型执行时，调用 [paddle.assign](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/assign_cn.html) 用该临时变量记录变量**y**的数值。

### 6.3 添加 Buffer 变量完成动转静

Buffer 的概念仅仅影响动态图向静态图的转换过程。在上一节中创建了一个临时变量用来临时存储中间变量的值。但这个临时变量在动态图向静态图转换的过程中并不会被记录在静态的计算图当中。如果希望该变量成为静态图的一部分，就需要进一步调用 register_buffers() 接口。

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        saved_tensor = self.create_tensor(name="saved_tensor0")
        self.register_buffer("saved_tensor", saved_tensor, persistable=True)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(10, 100)

    def forward(self, input):
        y = self.flatten(input)
        # Save intermediate tensor
        paddle.assign(y, self.saved_tensor)
        y = self.fc(y)
        return y
```

这样在动态图转静态图时 saved_tensor 就会被记录到静态图中。

对于模型中已经注册的 Buffer，可以通过 buffers() 或者 named_buffers() 进行访问。

```python
model = Model()
print(model.buffers())
for item in model.named_buffers():
    print(item)
```

```python
[Tensor(Not initialized)]
('saved_tensor', Tensor(Not initialized))
```

## 七、存储模型的参数

参考前面的操作完成 Layer 自定义和修改之后，可以参考以下操作进行保存。

首先调用 `state_dict()` 接口将模型中的参数以及永久变量存储到一个 Python 字典中，随后通过 [paddle.save()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/save_cn.html) 保存该字典。

state_dict 是一个简单的 Python 字典对象，将每一层与它的对应参数建立映射关系。可用于保存 Layer 或者 Optimizer。Layer.state_dict 可以保存训练过程中需要学习的权重和偏执系数，保存文件推荐使用后缀 `.pdparams` 。如果想要连同模型一起保存，则可以参考[paddle.jit.save()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html)



```plain
model = Model()
state_dict = model.state_dict()
paddle.save( state_dict, "paddle_dy.pdparams")
```

可以随时恢复：参数载入时，先从磁盘载入保存的 state_dict，然后通过 set_state_dict 方法配置到目标对象中。

```plain
model = Model()
state_dict = paddle.load("paddle_dy.pdparams")
model.set_state_dict(state_dict)
```



## 八、总结

至此，本文介绍了如何使用 paddle.nn.Layer 来辅助您构造深度学习网络模型，并展示了如何使用 paddle.nn.Layer 进行层的查看、修改。还可以根据自己的需要进一步探索 Layer 的更多用法。此外，如果在使用 paddle.nn.Layer 的过程中遇到任何问题及建议，欢迎在飞桨 Github 中进行提问和反馈。
