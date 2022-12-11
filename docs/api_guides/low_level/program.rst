.. _api_guide_Program:

#########
基础概念
#########

==================
Program
==================

:code:`Fluid` 中使用类似于编程语言的抽象语法树的形式描述用户的神经网络配置，用户对计算的描述都将写入一段 Program。Fluid 中的 Program 替代了传统框架中模型的概念，通过对顺序执行、条件选择和循环执行三种执行结构的支持，做到对任意复杂模型的描述。书写 :code:`Program` 的过程非常接近于写一段通用程序，如果您已经具有一定的编程经验，会很自然地将自己的知识迁移过来。


总得来说：

* 一个模型是一个 Fluid :code:`Program` ,一个模型可以含有多于一个 :code:`Program` ；

* :code:`Program` 由嵌套的 :code:`Block` 构成，:code:`Block` 的概念可以类比到 C++ 或是 Java 中的一对大括号，或是 Python 语言中的一个缩进块；

* :code:`Block` 中的计算由顺序执行、条件选择或者循环执行三种方式组合，构成复杂的计算逻辑；

* :code:`Block` 中包含对计算和计算对象的描述。计算的描述称之为 Operator；计算作用的对象（或者说 Operator 的输入和输出）被统一为 Tensor，在 Fluid 中，Tensor 用层级为 0 的 :ref:`Lod_Tensor  <cn_user_guide_lod_tensor>` 表示。



.. _api_guide_Block:

=========
Block
=========

:code:`Block` 是高级语言中变量作用域的概念，在编程语言中，Block 是一对大括号，其中包含局部变量定义和一系列指令或操作符。编程语言中的控制流结构 :code:`if-else` 和 :code:`for` 在深度学习中可以被等效为：

+-----------------+--------------------+
|    编程语言     |       Fluid        |
+=================+====================+
| for, while loop | RNN,WhileOP        |
+-----------------+--------------------+
| if-else, switch | IfElseOp, SwitchOp |
+-----------------+--------------------+
| 顺序执行        | 一系列 layers      |
+-----------------+--------------------+

如上文所说，Fluid 中的 :code:`Block` 描述了一组以顺序、选择或是循环执行的 Operator 以及 Operator 操作的对象：Tensor。




=============
Operator
=============

在 Fluid 中，所有对数据的操作都由 :code:`Operator` 表示，为了便于用户使用，在 Python 端，Fluid 中的 :code:`Operator` 被一步封装入 :code:`paddle.fluid.layers` ， :code:`paddle.fluid.nets` 等模块。

这是因为一些常见的对 Tensor 的操作可能是由更多基础操作构成，为了提高使用的便利性，框架内部对基础 Operator 进行了一些封装，包括创建 Operator 依赖可学习参数，可学习参数的初始化细节等，减少用户重复开发的成本。


更多内容可参考阅读 `Fluid 设计思想 <../../advanced_usage/design_idea/fluid_design_idea.html>`_

.. _api_guide_Variable:

=========
Variable
=========

Fluid 中的 :code:`Variable` 可以包含任何类型的值———在大多数情况下是一个 :ref:`Lod_Tensor <cn_user_guide_lod_tensor>` 。

模型中所有的可学习参数都以 :code:`Variable` 的形式保留在内存空间中，您在绝大多数情况下都不需要自己来创建网络中的可学习参数， Fluid 为几乎常见的神经网络基本计算模块都提供了封装。以最简单的全连接模型为例，调用 :code:`fluid.layers.fc` 会直接为全连接层创建连接权值( W )和偏置（ bias ）两个可学习参数，无需显示地调用 :code:`variable` 相关接口创建可学习参数。

.. _api_guide_Name:

=========
Name
=========

Fluid 中部分网络层里包含了 :code:`name` 参数，如 :ref:`cn_api_fluid_layers_fc` 。此 :code:`name` 一般用来作为网络层输出、权重的前缀标识，具体规则如下：

* 用于网络层输出的前缀标识。若网络层中指定了 :code:`name` 参数，Fluid 将以 ``name 值.tmp_数字`` 作为唯一标识对网络层输出进行命名；未指定 :code:`name` 参数时，则以 ``OP 名_数字.tmp_数字`` 的方式进行命名，其中的数字会自动递增，以区分同名 OP 下的不同网络层。

* 用于权重或偏置变量的前缀标识。若在网络层中通过 ``param_attr`` 和 ``bias_attr`` 创建了权重变量或偏置变量， 如 :ref:`cn_api_fluid_layers_embedding` 、 :ref:`cn_api_fluid_layers_fc` ，则 Fluid 会自动生成 ``前缀.w_数字`` 或 ``前缀.b_数字`` 的唯一标识对其进行命名，其中 ``前缀`` 为用户指定的 :code:`name` 或自动生成的 ``OP 名_数字`` 。若在 ``param_attr`` 和 ``bias_attr`` 中指定了 :code:`name` ，则用此 :code:`name` ，不再自动生成。细节请参考示例代码。

此外，在 :ref:`cn_api_fluid_ParamAttr` 中，可通过指定 :code:`name` 参数实现多个网络层的权重共享。

示例代码如下：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=1)
    emb = fluid.layers.embedding(input=x, size=(128, 100))  # embedding_0.w_0
    emb = fluid.layers.Print(emb) # Tensor[embedding_0.tmp_0]

    # default name
    fc_none = fluid.layers.fc(input=emb, size=1)  # fc_0.w_0, fc_0.b_0
    fc_none = fluid.layers.Print(fc_none)  # Tensor[fc_0.tmp_1]

    fc_none1 = fluid.layers.fc(input=emb, size=1)  # fc_1.w_0, fc_1.b_0
    fc_none1 = fluid.layers.Print(fc_none1)  # Tensor[fc_1.tmp_1]

    # name in ParamAttr
    w_param_attrs = fluid.ParamAttr(name="fc_weight", learning_rate=0.5, trainable=True)
    print(w_param_attrs.name)  # fc_weight

    # name == 'my_fc'
    my_fc1 = fluid.layers.fc(input=emb, size=1, name='my_fc', param_attr=w_param_attrs) # fc_weight, my_fc.b_0
    my_fc1 = fluid.layers.Print(my_fc1)  # Tensor[my_fc.tmp_1]

    my_fc2 = fluid.layers.fc(input=emb, size=1, name='my_fc', param_attr=w_param_attrs) # fc_weight, my_fc.b_1
    my_fc2 = fluid.layers.Print(my_fc2)  # Tensor[my_fc.tmp_3]

    place = fluid.CPUPlace()
    x_data = np.array([[1],[2],[3]]).astype("int64")
    x_lodTensor = fluid.create_lod_tensor(x_data, [[1, 2]], place)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    ret = exe.run(feed={'x': x_lodTensor}, fetch_list=[fc_none, fc_none1, my_fc1, my_fc2], return_numpy=False)


上述示例中， ``fc_none`` 和 ``fc_none1`` 均未指定 :code:`name` 参数，则以 ``OP 名_数字.tmp_数字`` 分别对该 OP 输出进行命名：``fc_0.tmp_1`` 和 ``fc_1.tmp_1`` ，其中 ``fc_0``  和 ``fc_1`` 中的数字自动递增以区分两个全连接层； ``my_fc1`` 和 ``my_fc2`` 均指定了 :code:`name` 参数，但取值相同，Fluid 以后缀 ``tmp_数字`` 进行区分，即 ``my_fc.tmp_1`` 和 ``my_fc.tmp_3`` 。

对于网络层中创建的变量， ``emb`` 层和 ``fc_none`` 、 ``fc_none1`` 层均默认以 ``OP 名_数字`` 为前缀对权重或偏置变量进行命名，如 ``embedding_0.w_0`` 、 ``fc_0.w_0`` 、 ``fc_0.b_0`` ，其前缀与 OP 输出的前缀一致。 ``my_fc1`` 层和 ``my_fc2`` 层则优先以 ``ParamAttr`` 中指定的 ``fc_weight`` 作为共享权重的名称。而偏置变量 ``my_fc.b_0`` 和 ``my_fc.b_1`` 则次优地以 :code:`name` 作为前缀标识。

在上述示例中，``my_fc1`` 和 ``my_fc2`` 两个全连接层通过构建 ``ParamAttr`` ，并指定 :code:`name` 参数，实现了网络层权重变量的共享机制。

.. _api_guide_ParamAttr:

=========
ParamAttr
=========

=========
相关 API
=========

* 用户配置的单个神经网络叫做 :ref:`cn_api_fluid_Program` 。值得注意的是，训练神经网
  络时，用户经常需要配置和操作多个 :code:`Program` 。比如参数初始化的
  :code:`Program` ， 训练用的 :code:`Program` ，测试用的
  :code:`Program` 等等。


* 用户还可以使用 :ref:`cn_api_fluid_program_guard` 配合 :code:`with` 语句，修改配置好的 :ref:`cn_api_fluid_default_startup_program` 和 :ref:`cn_api_fluid_default_main_program` 。

* 在 Fluid 中，Block 内部执行顺序由控制流决定，如 :ref:`cn_api_fluid_layers_IfElse` , :ref:`cn_api_fluid_layers_While`, :ref:`cn_api_fluid_layers_Switch` 等，更多内容可参考： :ref:`api_guide_control_flow`
