.. _api_guide_Program:

#########
基础概念
#########

==================
Program
==================

在飞桨中，Program 是一种静态图模型，类似于其他编程语言中的程序。静态图编程采用先编译后执行的方式。需先在代码中预定义完整的神经网络结构，飞桨框架会将神经网络描述为 Program 的数据结构，并对 Program 进行编译优化，再调用执行器获得计算结果。

* :code:`Program` 由嵌套的 :code:`Block` 构成，:code:`Block` 的概念可以类比到 C++ 或是 Java 中的一对大括号，或是 Python 语言中的一个缩进块；

* :code:`Block` 中的计算由顺序执行、条件选择或者循环执行三种方式组合，构成复杂的计算逻辑；

* :code:`Block` 中包含对计算和计算对象的描述。计算的描述称之为 :code:`Operator`；计算作用的对象（或者说 :code:`Operator` 的输入和输出）被统一为 :code:`Tensor`。


.. _api_guide_Block:

=========
Block
=========

:code:`Block` 是高级语言中变量作用域的概念，类似 C 语言或 Java 语言中的一对大括号，其中包含局部变量定义和一系列指令或操作符.

:code:`Block` 是计算图中用于表示计算逻辑的基本单元。它包含一系列操作（:code:`Operator`）和计算对象（:code:`Tensor`），支持顺序执行、条件选择和循环执行等控制结构，从而构建复杂的计算流程。

:code:`Block` 的主要特点：

* 计算描述： :code:`Block` 内部包含多个 :code:`Operator`，每个 Operator 表示一个计算操作，如加法、卷积等。

* 对象描述： :code:`Block` 中的计算对象统一为 :code:`Tensor`，表示多维数组或矩阵，是数据存储和传输的基本单元。

* 控制结构： :code:`Block` 支持顺序执行、条件选择和循环执行等控制结构，使得计算流程更加灵活和复杂。

在飞桨的计算图中，:code:`Block` 、:code:`Operator` 和 :code:`Tensor` 共同构成了计算流程的骨架。Block 提供了容器功能，组织和管理内部的 :code:`Operator` 和 :code:`Tensor`，从而实现高效的计算图构建和执行。



.. _api_guide_Operator:

=============
Operator
=============

在 Paddle 中，所有对数据的操作都由 :code:`Operator` 表示 每个 :code:`Operator` 执行特定的功能，如矩阵乘法、卷积、激活函数等，通过组合这些 :code:`Operator`，可以构建复杂的计算图，实现模型的前向传播和反向传播。



.. _api_guide_Variable:

=========
Variable
=========

Paddle 中的 :code:`Variable` 可以包含任何类型的值———在大多数情况下是一个 :ref:`Tensor <cn_user_guide_tensor>` 。

模型中所有的可学习参数都以 :code:`Variable` 的形式保留在内存空间中，您在绝大多数情况下都不需要自己来创建网络中的可学习参数， Paddle 为几乎常见的神经网络基本计算模块都提供了封装。以静态图中最简单的全连接模型为例，调用 :code:`paddle.static.nn.fc` 会直接为全连接层创建连接权值( W )和偏置（ bias ）两个可学习参数，无需显示地调用 :code:`variable` 相关接口创建可学习参数。

.. _api_guide_Name:

=========
Name
=========

Paddle 中部分网络层里包含了 :code:`name` 参数，如 :ref:`cn_api_static_nn_fc` 。此 :code:`name` 一般用来作为网络层输出、权重的前缀标识，具体规则如下：

* 用于网络层输出的前缀标识。若网络层中指定了 :code:`name` 参数，Paddle 将以 ``name 值.tmp_数字`` 作为唯一标识对网络层输出进行命名；未指定 :code:`name` 参数时，则以 ``OP 名_数字.tmp_数字`` 的方式进行命名，其中的数字会自动递增，以区分同名 OP 下的不同网络层。

* 用于权重或偏置变量的前缀标识。若在网络层中通过 ``param_attr`` 和 ``bias_attr`` 创建了权重变量或偏置变量， 如 :ref:`cn_api_nn_embedding` 、 :ref:`cn_api_static_nn_fc` ，则 Paddle 会自动生成 ``前缀.w_数字`` 或 ``前缀.b_数字`` 的唯一标识对其进行命名，其中 ``前缀`` 为用户指定的 :code:`name` 或自动生成的 ``OP 名_数字`` 。若在 ``param_attr`` 和 ``bias_attr`` 中指定了 :code:`name` ，则用此 :code:`name` ，不再自动生成。细节请参考示例代码。

此外，在 :ref:`cn_api_ParamAttr` 中，可通过指定 :code:`name` 参数实现多个网络层的权重共享。

示例代码如下：

.. code-block:: python

    import paddle
    import numpy as np

    embedding = paddle.nn.Embedding(num_embeddings=128, embedding_dim=100)
    emb = embedding(x)  # embedding_0.w_0
    print(emb) # Tensor[embedding_0.tmp_0]

    # default name
    fc = paddle.nn.Linear(in_features=100, out_features=1)
    fc_out = fc(emb)  # fc_0.w_0, fc_0.b_0
    print(fc_out)  # Tensor[fc_0.tmp_1]

    fc1 = paddle.nn.Linear(in_features=100, out_features=1)  # fc_1.w_0, fc_1.b_0
    fc1_out = fc1(emb)  # fc_1.w_0, fc_1.b_0
    print(fc1_out)  # Tensor[fc_1.tmp_1]

    # name in ParamAttr
    w_param_attrs = paddle.ParamAttr(name="fc_weight", learning_rate=0.5, trainable=True)
    print(w_param_attrs.name)  # fc_weight

    # name == 'my_fc'
    my_fc = paddle.nn.Linear(in_features=100, out_features=1, name='my_fc', weight_attr=w_param_attrs)
    my_fc_out = my_fc(emb) # fc_weight, my_fc.b_0
    print(my_fc_out)  # Tensor[my_fc.tmp_1]

    my_fc2 = paddle.nn.Linear(in_features=100, out_features=1, name='my_fc', weight_attr=w_param_attrs)
    my_fc2_out = my_fc2(emb) # fc_weight, my_fc.b_1
    print(my_fc2_out)  # Tensor[my_fc.tmp_3]

    place = paddle.CPUPlace()

    exe = paddle.static.Executor(place)

    exe.run(paddle.static.default_startup_program())

    ret = exe.run(feed={'x': x}, fetch_list=[fc_out, fc1_out, my_fc_out, my_fc2_out], return_numpy=False)


上述示例中， ``fc_none`` 和 ``fc_none1`` 均未指定 :code:`name` 参数，则以 ``OP 名_数字.tmp_数字`` 分别对该 OP 输出进行命名：``fc_0.tmp_1`` 和 ``fc_1.tmp_1`` ，其中 ``fc_0``  和 ``fc_1`` 中的数字自动递增以区分两个全连接层； ``my_fc1`` 和 ``my_fc2`` 均指定了 :code:`name` 参数，但取值相同，Paddle 以后缀 ``tmp_数字`` 进行区分，即 ``my_fc.tmp_1`` 和 ``my_fc.tmp_3`` 。

对于网络层中创建的变量， ``emb`` 层和 ``fc_none`` 、 ``fc_none1`` 层均默认以 ``OP 名_数字`` 为前缀对权重或偏置变量进行命名，如 ``embedding_0.w_0`` 、 ``fc_0.w_0`` 、 ``fc_0.b_0`` ，其前缀与 OP 输出的前缀一致。 ``my_fc1`` 层和 ``my_fc2`` 层则优先以 ``ParamAttr`` 中指定的 ``fc_weight`` 作为共享权重的名称。而偏置变量 ``my_fc.b_0`` 和 ``my_fc.b_1`` 则次优地以 :code:`name` 作为前缀标识。

在上述示例中，``my_fc1`` 和 ``my_fc2`` 两个全连接层通过构建 ``ParamAttr`` ，并指定 :code:`name` 参数，实现了网络层权重变量的共享机制。

.. _api_guide_ParamAttr:

=========
ParamAttr
=========

``ParamAttr`` 是用于设置模型参数（如权重和偏置）属性的配置类。通过 ``ParamAttr``，用户可以灵活地定义参数的初始化方式、正则化策略、梯度裁剪以及模型平均等特性。

实例代码如下：

.. code-block:: python
    import paddle
    from paddle import ParamAttr

    # 创建一个全连接层，设置权重和偏置的属性
    fc = paddle.nn.Linear(in_features=128, out_features=64,
                          weight_attr=ParamAttr(
                              name='fc_weight',
                              initializer=paddle.nn.initializer.XavierUniform(),
                              regularizer=paddle.regularizer.L2Decay(0.0001)
                          ),
                           bias_attr=ParamAttr(
                              name='fc_bias',
                              initializer=paddle.nn.initializer.Constant(0.0)
                          ))


在上述示例中： :code:`weight_attr` 和 :code:`bias_attr` 分别设置了权重和偏置的属性。:code:`name` 指定参数的名称。:code:`initializer` 设置参数的初始化方式。:code:`regularizer` 设置参数的正则化策略。

=========
相关 API
=========

* 用户配置的单个神经网络叫做 :ref:`cn_api_Program` 。值得注意的是，训练神经网络时，用户经常需要配置和操作多个 :code:`Program` 。比如参数初始化的:code:`Program` ， 训练用的 :code:`Program` ，测试用的:code:`Program` 等等。


* 用户还可以使用 :ref:`cn_api_program_guard` 配合 :code:`with` 语句，修改配置好的 :ref:`cn_api_default_startup_program` 和 :ref:`cn_api_default_main_program` 。
