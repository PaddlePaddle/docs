
# 编程指南

目前飞桨（PaddlePaddle，以下简称Paddle）已经同时支持命令式编程模式(动态图)和声明式编程模式(静态图)两种编程方式，
本文主要侧重于介绍声明式编程模式的编程方法，关于命令式编程模式编程方法，请参考[命令式编程模式机制-DyGraph](../dygraph/DyGraph.html)。

阅读完本文档，您将了解在Paddle声明式编程模式编程方式中，如何表示和定义数据变量，以及如何完整的组建一个深度学习网络并进行训练。

## 数据的表示和定义

Paddle和其他主流框架一样，使用Tensor数据结构来承载数据，包括模型中的可学习参数（如网络权重、偏置等），
网络中每一层的输入输出数据，常量数据等。

Tensor可以简单理解成一个多维数组，一般而言可以有任意多的维度。
不同的Tensor可以具有自己的数据类型和形状，同一Tensor中每个元素的数据类型是一样的，
Tensor的形状就是Tensor的维度。关于Tensor的详细介绍请参阅：[Tensor](../tensor.html) 。

在Paddle中我们使用 `fluid.data` 来创建数据变量， `fluid.data` 需要指定Tensor的形状信息和数据类型，
当遇到无法确定的维度时，可以将相应维度指定为None，如下面的代码片段所示：

```python
import paddle.fluid as fluid

# 定义一个数据类型为int64的二维数据变量x，x第一维的维度为3，第二个维度未知，要在程序执行过程中才能确定，因此x的形状可以指定为[3, None]
x = fluid.data(name="x", shape=[3, None], dtype="int64")

# 大多数网络都会采用batch方式进行数据组织，batch大小在定义时不确定，因此batch所在维度（通常是第一维）可以指定为None
batched_x = fluid.data(name="batched_x", shape=[None, 3, None], dtype='int64')
```

除 `fluid.data` 之外，我们还可以使用 `fluid.layers.fill_constant` 来创建常量，
如下代码将创建一个维度为[3, 4], 数据类型为int64的Tensor，其中所有元素均为16（value参数所指定的值）。

```python
import paddle.fluid as fluid
data = fluid.layers.fill_constant(shape=[3, 4], value=16, dtype='int64')
```

以上例子中，我们只使用了一种数据类型"int64"，即有符号64位整数数据类型，更多Paddle目前支持的数据类型请查看：[支持的数据类型](../../../advanced_guide/data_preparing/feeding_data.html#fluid)。

需要注意的是，在声明式编程模式编程方式中，上述定义的Tensor并不具有值（即使创建常量的时候指定了value），
它们仅表示将要执行的操作，在网络执行时（训练或者预测）才会进行真正的赋值操作，
如您直接打印上例代码中的data将会得对其信息的描述：

```python
print data
```
输出结果:

```
name: "fill_constant_0.tmp_0"
type {
    type: LOD_TENSOR
    lod_tensor {
        tensor {
            data_type: INT64
            dims: 3
            dims: 4
        }
    }
}
persistable: false
```

在网络执行过程中，获取Tensor数值有两种方式：方式一是利用 `paddle.fluid.layers.Print` 创建一个打印操作，
打印正在访问的Tensor。方式二是将Variable添加在fetch_list中。

方式一的代码实现如下所示：

```python
import paddle.fluid as fluid

data = fluid.layers.fill_constant(shape=[3, 4], value=16, dtype='int64')
data = fluid.layers.Print(data, message="Print data:")

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

ret = exe.run()
```

运行时的输出结果：

```
1571742368    Print data:    The place is:CPUPlace
Tensor[fill_constant_0.tmp_0]
    shape: [3,4,]
    dtype: x
    data: 16,16,16,16,16,16,16,16,16,16,16,16,
```

方式二Fetch_list的详细过程会在后文展开描述。

## 数据读取

使用 `fluid.data` 创建数据变量之后，我们需要把网络执行所需要的数据读取到对应变量中，
具体的数据准备过程，请阅读[准备数据](../../../advanced_guide/data_preparing/index_cn.html)。

## 组建网络

在Paddle中，数据计算类API统一称为Operator（算子），简称OP，大多数OP在 `paddle.fluid.layers` 模块中提供。

例如用户可以利用 `paddle.fluid.layers.elementwise_add()` 实现两个输入Tensor的加法运算：

```python
# 定义变量
import paddle.fluid as fluid
a = fluid.data(name="a", shape=[None, 1], dtype='int64')
b = fluid.data(name="b", shape=[None, 1], dtype='int64')

# 组建网络（此处网络仅由一个操作构成，即elementwise_add）
result = fluid.layers.elementwise_add(a,b)

# 准备运行网络
cpu = fluid.CPUPlace() # 定义运算设备，这里选择在CPU下训练
exe = fluid.Executor(cpu) # 创建执行器
exe.run(fluid.default_startup_program()) # 网络参数初始化

# 读取输入数据
import numpy
data_1 = int(input("Please enter an integer: a="))
data_2 = int(input("Please enter an integer: b="))
x = numpy.array([[data_1]])
y = numpy.array([[data_2]])

# 运行网络
outs = exe.run(
    feed={'a':x, 'b':y}, # 将输入数据x, y分别赋值给变量a，b
    fetch_list=[result] # 通过fetch_list参数指定需要获取的变量结果
    )

# 输出计算结果
print "%d+%d=%d" % (data_1,data_2,outs[0][0])
```

输出结果：
```
Please enter an integer: a=7
Please enter an integer: b=3
7+3=10
```

本次运行时，输入a=7，b=3，得到outs=10。

您可以复制这段代码在本地执行，根据指示输入其他数值观察计算结果。

如果想获取网络执行过程中的a，b的具体值，可以将希望查看的变量添加在fetch_list中。

```python
...
# 运行网络
outs = exe.run(
    feed={'a':x, 'b':y}, # 将输入数据x, y分别赋值给变量a，b
    fetch_list=[a, b, result] # 通过fetch_list参数指定需要获取的变量结果
    )

# 输出计算结果
print outs
```

输出结果：
```
[array([[7]]), array([[3]]), array([[10]])]
```

## 组建更加复杂的网络

某些场景下，用户需要根据当前网络中的某些状态，来具体决定后续使用哪一种操作，或者重复执行某些操作。在命令式编程模式中，可以方便的使用Python的控制流语句（如for，if-else等）来进行条件判断，但是在声明式编程模式中，由于组网阶段并没有实际执行操作，也没有产生中间计算结果，因此无法使用Python的控制流语句来进行条件判断，为此声明式编程模式提供了多个控制流API来实现条件判断。这里以[fluid.layers.while_loop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/layers_cn/while_loop_cn.html)为例来说明如何在声明式编程模式中实现条件循环的操作。

while_loop API用于实现类似while/for的循环控制功能，使用一个callable的方法cond作为参数来表示循环的条件，只要cond的返回值为True，while_loop就会循环执行循环体body（也是一个callable的方法），直到 cond 的返回值为False。对于while_loop API的详细定义和具体说明请参考文档[fluid.layers.while_loop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/layers_cn/while_loop_cn.html)。

下面的例子中，使用while_loop API进行条件循环操作，其实现的功能相当于在python中实现如下代码：

```python
i = 0
ten = 10
while i < ten:
    i = i + 1
print('i =', i)
```

在声明式编程模式中使用while_loop API实现以上代码的逻辑：

```python
# 该代码要求安装飞桨1.7+版本

# 该示例代码展示整数循环+1，循环10次，输出计数结果
import paddle.fluid as fluid
import paddle.fluid.layers as layers

# 定义cond方法，作为while_loop的判断条件
def cond(i, ten):
    return i < ten

# 定义body方法，作为while_loop的执行体，只要cond返回值为True，while_loop就会一直调用该方法进行计算
# 由于在使用while_loop OP时，cond和body的参数都是由while_loop的loop_vars参数指定的，所以cond和body必须有相同数量的参数列表，因此body中虽然只需要i这个参数，但是仍然要保持参数列表个数为2，此处添加了一个dummy参数来进行"占位"
def body(i, dummy):
    # 计算过程是对输入参数i进行自增操作，即 i = i + 1
    i = i + 1
    return i, dummy

i = layers.fill_constant(shape=[1], dtype='int64', value=0) # 循环计数器
ten = layers.fill_constant(shape=[1], dtype='int64', value=10) # 循环次数
out, ten = layers.while_loop(cond=cond, body=body, loop_vars=[i, ten]) # while_loop的返回值是一个tensor列表，其长度，结构，类型与loop_vars相同

exe = fluid.Executor(fluid.CPUPlace())
res = exe.run(fluid.default_main_program(), feed={}, fetch_list=out)
print(res) #[array([10])]

```

限于篇幅，上面仅仅用一个最简单的例子来说明如何在声明式编程模式中实现循环操作。循环操作在很多应用中都有着重要作用，比如NLP中常用的Transformer模型，在解码（生成）阶段的Beam Search算法中，需要使用循环操作来进行候选的选取与生成，可以参考[Transformer](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/PaddleMT/transformer)模型的实现来进一步学习while_loop在复杂场景下的用法。

除while_loop之外，飞桨还提供fluid.layers.cond API来实现条件分支的操作，以及fluid.layers.switch_case和fluid.layers.case API来实现分支控制功能，具体用法请参考文档：[cond](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/layers_cn/cond_cn.html)，[switch_case](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/layers_cn/switch_case_cn.html)和[case](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/layers_cn/case_cn.html#case)

## 一个完整的网络示例

一个典型的模型通常包含4个部分，分别是：输入数据定义，搭建网络（模型前向计算逻辑），定义损失函数，以及选择优化算法。

下面我们通过一个非常简单的数据预测网络（线性回归），来完整的展示如何使用Paddle声明式编程模式方式完成一个深度学习模型的组建和训练。

问题描述：给定一组数据 $<X,Y>$，求解出函数 $f$，使得 $y=f(x)$，其中$X$,$Y$均为一维张量。最终网络可以依据输入$x$，准确预测出$y_{\_predict}$。

1. 定义数据

    假设输入数据X=[1 2 3 4]，Y=[2 4 6 8]，在网络中定义：

    ```python
    # 定义X数值
    train_data=numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
    # 定义期望预测的真实值y_true
    y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')
    ```

2. 搭建网络（定义前向计算逻辑）

    接下来需要定义预测值与输入的关系，本次使用一个简单的线性回归函数进行预测：

    ```python
    # 定义输入数据类型
    x = fluid.data(name="x", shape=[None, 1], dtype='float32')
    y = fluid.data(name="y", shape=[None, 1], dtype='float32')
    # 搭建全连接网络
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    ```

3. 添加损失函数

    完成模型搭建后，如何评估预测结果的好坏呢？我们通常在设计的网络中添加损失函数，以计算真实值与预测值的差。

    在本例中，损失函数采用[均方差函数](https://en.wikipedia.org/wiki/Mean_squared_error)：

    ```python
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    ```

4. 网络优化

    确定损失函数后，可以通过前向计算得到损失值，并根据损失值对网络参数进行更新，最简单的算法是随机梯度下降法：w=w−η⋅g，由 `fluid.optimizer.SGD` 实现：

    ```python
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost)

    ```

    让我们的网络训练100次，查看结果：

    ```python
    # 加载库
    import paddle.fluid as fluid
    import numpy

    # 定义输入数据
    train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
    y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')

    # 组建网络
    x = fluid.data(name="x",shape=[None, 1],dtype='float32')
    y = fluid.data(name="y",shape=[None, 1],dtype='float32')
    y_predict = fluid.layers.fc(input=x,size=1,act=None)

    # 定义损失函数
    cost = fluid.layers.square_error_cost(input=y_predict,label=y)
    avg_cost = fluid.layers.mean(cost)

    # 选择优化方法
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost)

    # 网络参数初始化
    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    # 开始训练，迭代100次
    for i in range(100):
        outs = exe.run(
            feed={'x':train_data, 'y':y_true},
            fetch_list=[y_predict, avg_cost])

    # 输出训练结果
    print outs
    ```

    输出结果:
    ```
    [array([[2.2075021],
            [4.1005487],
            [5.9935956],
            [7.8866425]], dtype=float32), array([0.01651453], dtype=float32)]
    ```

    可以看到100次迭代后，预测值已经非常接近真实值了，损失值也下降到了0.0165。

    恭喜您！已经成功完成了第一个简单网络的搭建，想尝试线性回归的进阶版——房价预测模型，请阅读：[线性回归](../../../user_guides/simple_case/fit_a_line/README.cn.html)。更多丰富的模型实例可以在[典型案例](../../../user_guides/index_cn.html)中找到。

<a name="what_next"></a>
## 进一步学习

如果您已经掌握了基本操作，可以进行下一阶段的学习了：

跟随这一教程将学习到如何对实际问题建模并使用Paddle构建模型：[配置简单的网络](../../coding_practice/configure_simple_model/index_cn.html)。

完成网络搭建后，可以开始在单机上训练您的网络了，详细步骤请参考[单机训练](../../coding_practice/single_node.html)。

除此之外，使用文档模块根据开发者的不同背景划分了三个学习阶段：[快速上手](../../index_cn.html)、[典型案例](../../../user_guides/index_cn.html)和[进阶指南](../../../advanced_guide/index_cn.html)。

如果您希望阅读更多场景下的应用案例，可以参考[典型案例](../../../user_guides/index_cn.html)。已经具备深度学习基础知识的用户，也可以从[进阶指南](../../../advanced_guide/index_cn.html)开始阅读。
