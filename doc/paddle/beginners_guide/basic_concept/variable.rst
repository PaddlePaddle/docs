.. _cn_user_guide_Variable:

=========
Variable
=========

飞桨（PaddlePaddle，以下简称Paddle）中的 :code:`Variable` 可以包含任何类型的值变量，提供的API中用到的类型是 :ref:`Tensor <cn_user_guide_tensor>` 。

后续的文档介绍中提到的 :code:`Variable` 基本等价于 :ref:`Tensor <cn_user_guide_tensor>` （特殊的地方会标注说明）。

在 Paddle 中存在三种 :code:`Variable`：

**1. 模型中的可学习参数**

模型中的可学习参数（包括网络权重、偏置等）生存期和整个训练任务一样长，会接受优化算法的更新，在 Paddle中以 Variable 的子类 Parameter 表示。

在Paddle中可以通过 :code:`fluid.layers.create_parameter` 来创建可学习参数：

.. code-block:: python

    w = fluid.layers.create_parameter(name="w",shape=[1],dtype='float32')



Paddle 为大部分常见的神经网络基本计算模块都提供了封装。以最简单的全连接模型为例，下面的代码片段会直接为全连接层创建连接权值（W）和偏置（ bias ）两个可学习参数，无需显式地调用 Parameter 相关接口来创建。

.. code-block:: python

    import paddle.fluid as fluid
    y = fluid.layers.fc(input=x, size=128, bias_attr=True)


**2. 占位 Variable**

在声明式编程模式(静态图)模式下，组网的时候通常不知道实际输入的信息，此刻需要一个占位的 :code:`Variable`，表示一个待提供输入的 :code:`Variable`

Paddle 中使用 :code:`fluid.data` 来接收输入数据， :code:`fluid.data` 需要提供输入 Tensor 的形状信息，当遇到无法确定的维度时，相应维度指定为 None ，如下面的代码片段所示：

.. code-block:: python

    import paddle.fluid as fluid

    #定义x的维度为[3,None]，其中我们只能确定x的第一的维度为3，第二个维度未知，要在程序执行过程中才能确定
    x = fluid.data(name="x", shape=[3,None], dtype="int64")

    #若图片的宽度和高度在运行时可变，将宽度和高度定义为None。
    #shape的三个维度含义分别是：batch_size, channel、图片的宽度、图片的高度
    b = fluid.data(name="image",shape=[None, 3,None,None],dtype="float32")


其中，dtype=“int64”表示有符号64位整数数据类型，更多Fluid目前支持的数据类型请查看： :ref:`Paddle目前支持的数据类型 <user_guide_use_numpy_array_as_train_data>` 。

**3. 常量 Variable**

Fluid 通过 :code:`fluid.layers.fill_constant` 来实现常量Variable，用户可以指定内部包含Tensor的形状，数据类型和常量值。代码实现如下所示：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')

