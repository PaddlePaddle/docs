# Paddle中的模型与层

模型是深度学习中的重要概念之一。模型的核心功能是将一组输入变量经过一系列计算，映射到另一组输出变量，该映射函数即代表一种深度学习算法。在**Paddle**框架中，模型包括以下两方面内容：

1. 一系列层的组合用于进行映射（前向执行）
2. 一些参数变量在训练过程中实时更新

本文档中，你将学习如何定义与使用**Paddle**模型，并了解模型与层的关系。

## 在**Paddle**中定义模型与层

在**Paddle**中，大多数模型由一系列层组成，层是模型的基础逻辑执行单元。层中持有两方面内容：一方面是计算所需的变量，以临时变量或参数的形式作为层的成员持有，另一方面则持有一个或多个具体的**Operator**来完成相应的计算。

从零开始构建变量、**Operator**，从而组建层、模型是一个很复杂的过程，并且当中难以避免的会出现很多冗余代码，因此**Paddle**提供了基础数据类型 ``paddle.nn.Layer`` ，来方便你快速的实现自己的层和模型。模型和层都可以基于 ``paddle.nn.Layer`` 扩充实现，因此也可以说模型只是一种特殊的层。下面将演示如何利用 ``paddle.nn.Layer`` 建立自己的模型：

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```

当前示例中，通过继承 ``paddle.nn.Layer`` 的方式构建了一个模型类型 ``Model`` ，模型中仅包含一个 ``paddle.nn.Flatten`` 层。模型执行时，输入变量**inputs**会被 ``paddle.nn.Flatten`` 层展平。

## 子层接口
如果想要访问或修改一个模型中定义的层，则可以调用**SubLayer**相关的接口。

以上文创建的简单模型为例, 如果想要查看模型中定义的所有子层：

```python
model = Model()
print(model.sublayers())

print("----------------------")

for item in model.named_sublayers():
    print(item)
```

```text
[Flatten()]
----------------------
('flatten', Flatten())
```

可以看到，通过调用 ``model.sublayers()`` 接口，打印出了前述模型中持有的全部子层(这时模型中只有一个 ``paddle.nn.Flatten`` 子层)。

而遍历 ``model.named_sublayers()`` 时，每一轮循环会拿到一组 ( 子层名称('flatten')，子层对象(paddle.nn.Flatten) )的元组。

接下来如果想要进一步添加一个子层，则可以调用 ``add_sublayer()`` 接口：

```python
fc = paddle.nn.Linear(10, 3)
model.add_sublayer("fc", fc)
print(model.sublayers())
```

```text
[Flatten(), Linear(in_features=10, out_features=3, dtype=float32)]
```

可以看到 ``model.add_sublayer()`` 向模型中添加了一个 ``paddle.nn.Linear`` 子层，这样模型中总共有 ``paddle.nn.Flatten`` 和 ``paddle.nn.Linear`` 两个子层了。


通过上述方法可以往模型中添加成千上万个子层，当模型中子层数量较多时，如何高效地对所有子层进行统一修改呢？**Paddle** 提供了 ``apply()`` 接口。通过这个接口，可以自定义一个函数，然后将该函数批量作用在所有子层上：

```python
def function(layer):
    print(layer)

model.apply(function)
```

```text
Flatten()
Linear(in_features=10, out_features=3, dtype=float32)

Model(
  (flatten): Flatten()
  (fc): Linear(in_features=10, out_features=3, dtype=float32)
)
```

当前例子中，定义了一个以**layer**作为参数的函数**function**，用来打印传入的**layer**信息。通过调用 ``model.apply()`` 接口，将**function**作用在模型的所有子层中，也因此输出信息中打印了**model**中所有子层的信息。


另外一个批量访问子层的接口是 ``children()`` 或者 ``named_children()`` 。这两个接口通过**Iterator**的方式访问每个子层：

```python
sublayer_iter = model.children()
for sublayer in sublayer_iter:
    print(sublayer)
```

```text
Flatten()
Linear(in_features=10, out_features=3, dtype=float32)
```

可以看到，遍历 ``model.children()`` 时，每一轮循环都可以按照子层注册顺序拿到对应 ``paddle.nn.Layer`` 的对象

## 层的变量成员

### 参数变量的添加与修改

有的时候希望向网络中添加一个参数作为输入。比如在使用图像风格转换模型时，会使用参数作为输入图像，在训练过程中不断更新该图像参数，最终拿到风格转换后的图像。

这时可以通过 ``create_parameter()`` 与 ``add_parameter()`` 组合，来创建并记录参数：

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
        img = self.create_parameter([1,3,256,256])
        self.add_parameter("img", img)
        self.flatten = paddle.nn.Flatten()

    def forward(self):
        y = self.flatten(self.img)
        return y
```

上述例子创建并向模型中添加了一个名字为"img"的参数。随后可以直接通过调用**model.img**来访问该参数。

对于已经添加的参数，可以通过 ``parameters()`` 或者 ``named_parameters()`` 来访问

```python
model = Model()
model.parameters()
print("----------------------------------------------------------------------------------")
for item in model.named_parameters():
    print(item)
```

```text
[Parameter containing:
Tensor(shape=[1, 3, 256, 256], dtype=float32, place=CPUPlace, stop_gradient=False,
       ...
----------------------------------------------------------------------------------
('img', Parameter containing:
Tensor(shape=[1, 3, 256, 256], dtype=float32, place=CPUPlace, stop_gradient=False,
       ...
```

可以看到，``model.parameters()`` 将模型中所有参数以数组的方式返回。

在实际的模型训练过程中，当调用反向图执行方法后，**Paddle**会计算出模型中每个参数的梯度并将其保存在相应的参数对象中。如果已经对该参数进行了梯度更新，或者出于一些原因不希望该梯度累加到下一轮训练，则可以调用 ``clear_gradients()`` 来清除这些梯度值。

```python
model = Model()
out = model()
out.backward()
model.clear_gradients()
```

### 非参数变量的添加
参数变量往往需要参与梯度更新，但很多情况下只是需要一个临时变量甚至一个常量。比如在模型执行过程中想将一个中间变量保存下来，这时需要调用 ``create_tensor()`` 接口：

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
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

这里调用 ``self.create_tensor()`` 创造了一个临时变量并将其记录在模型的 ``self.saved_tensor`` 中。在模型执行时调用 ``paddle.assign`` 用该临时变量记录变量**y**的数值。

### **Buffer** 变量的添加
**Buffer**的概念仅仅影响动态图向静态图的转换过程。在上一节中创建了一个临时变量用来临时存储中间变量的值。但这个临时变量在动态图向静态图转换的过程中并不会被记录在静态的计算图当中。如果希望该变量成为静态图的一部分，就需要进一步调用 ``register_buffers()`` 接口：

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
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

这样在动态图转静态图时**saved_tensor**就会被记录到静态图中。

对于模型中已经注册的**Buffer**，可以通过 ``buffers()`` 或者 ``named_buffers()`` 进行访问:

```python
model = Model()
print(model.buffers())
for item in model.named_buffers():
    print(item)
```

```text
[Tensor(Not initialized)]
('saved_tensor', Tensor(Not initialized))
```

可以看到 ``model.buffers()`` 以数组形式返回了模型中注册的所有**Buffer**

## 层的执行

经过一系列对模型的配置，假如已经准备好了一个**Paddle**模型如下：

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```

想要执行该模型，首先需要对执行模式进行设置

### 执行模式设置

模型的执行模式有两种，如果需要训练的话调用 ``train()`` ，如果只进行前向执行则调用 ``eval()``

```python
x = paddle.randn([10, 1], 'float32')
model = Model()
model.eval()  # set model to eval mode
out = model(x)
model.train()  # set model to train mode
out = model(x)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       ...
```

这里将模型的执行模式先后设置为**eval**和**train**。两种执行模式是互斥的，新的执行模式设置会覆盖原有的设置。

### 执行函数

模式设置完成后可以直接调用执行函数。可以直接调用forward()方法进行前向执行，也可以调用 ``__call__()`` ，从而执行在 ``forward()`` 当中定义的前向计算逻辑。

```python
model = Model()
x = paddle.randn([10, 1], 'float32')
out = model(x)
print(out)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       ...
```

这里直接调用 ``__call__()`` 方法调用模型的前向执行逻辑。

### 添加额外的执行逻辑

有时希望某些变量在进入层前首先进行一些预处理，这个功能可以通过注册**hook**来实现。**hook**是一个作用于变量的自定义函数，在模型执行时调用。对于注册在层上的**hook**函数，可以分为**pre_hook**和**post_hook**两种。**pre_hook**可以对层的输入变量进行处理，用函数的返回值作为新的变量参与层的计算。**post_hook**则可以对层的输出变量进行处理，将层的输出进行进一步处理后，用函数的返回值作为层计算的输出。

通过 ``register_forward_post_hook()`` 接口，我们可以注册一个**post_hook**：

```python
def forward_post_hook(layer, input, output):
    return 2*output

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_post_hook_handle = model.flatten.register_forward_post_hook(forward_post_hook)
out = model(x)
print(out)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[2.],
        [2.],
        ...
```

同样的也可以使用 ``register_forward_pre_hook()`` 来注册**pre_hook**：

```python
def forward_pre_hook(layer, input, output):
    return 2*output

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_pre_hook_handle = model.flatten.register_forward_pre_hook(forward_pre_hook)
out = model(x)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[2.],
        [2.],
        ...
```

## 模型数据保存

如果想要保存模型中参数而不存储模型本身，则可以首先调用 ``state_dict()`` 接口将模型中的参数以及永久变量存储到一个**Python**字典中，随后保存该字典。

```python
model = Model()
state_dict = model.state_dict()
paddle.save( state_dict, "paddle_dy.pdparams")
```

可以随时恢复：

```python
model = Model()
state_dict = paddle.load("paddle_dy.pdparams")
model.set_state_dict(state_dict)
```
如果想要连同模型一起保存，则可以参考[paddle.jit.save()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html)
