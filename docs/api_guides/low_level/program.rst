.. _api_guide_Program:

#########
基础概念
#########

.. _api_guide_IR:

==================
IR
==================

:code:`Paddle` 通过一种 IR（Intermediate Representation，中间表示形式）来表示计算图，并在此基础上借助编译器的理念、技术和工具对神经网络进行自动优化和代码生成。

新 IR 通过 :code:`Operation` 、:code:`Region` 、:code:`Block` 三者的循环嵌套来表示结构化控制流。

一个 Op 会包含 0 个或多个 :code:`Region` , 一个 :code:`Region` 会包含 0 个或多个 :code:`Block` , 一个 :code:`Block` 里面包含了 0 个或多个:code:`Operation` 。 三者循环嵌套包含，用来描述复杂的模型结构。


==================
Program
==================

:code:`Program` 用来表示一个具体的模型。它包含两部分：计算图 和 权重 。 模型等价于一个有向无环图。 :code:`Operation` 为节点，:code:`Value` 为边


权重（ :code:`Weight` ）用来对模型的权重参数进行单独存储，:code:`Value `、:code:`Operation` 用来对计算图进行抽象。

:code:`Operation` 表示计算图中的节点。一个 :code:`Operation` 表示一个算子，它里面包含了零个或多个 :code:`Region` 。:code:`Region` 表示一个闭包，它里面包含了零个或多个 :code:`Block` 。:code:`Block` 表示一个符合 SSA 的基本块，里面包含了零个或多个 :code:`Operation` 。三者循环嵌套，可以实现任意复杂的语法结构。

:code:`Value` 表示计算图中的有向边，他用来将两个 :code:`Operation` 关联起来，描述了程序中的 UD 链 。

:code:`Program` 中 ``ModuleOp module_`` 存计算图 , ParameterMap ``parameters_`` 存权重。 ``ModuleOp`` 类中, 用 :code:`Block` 来存计算图中的内容。

.. _api_guide_Region:

=========
Region
=========

:code:`Region` 里面包含了一个:code:`Block`列表， 第一个:code:`Block` (如果存在的话)，称为该 :code:`Region` 的入口块。

与基本块不同，:code:`Region` 存在一个最显著的约束是：:code:`Region` 内定义的 :code:`Value` 只能在该 :code:`Region` 内部使用，:code:`Region` 的外面不允许使用。

当控制流进入一个 :code:`Region` ， 相当于创建了一个新的子 scope， 当控制流退出该 :code:`Region` 时，该子 scope 中定义的所有变量都可以回收。

控制流进入 :code:`Region` , 一定会首先进入该 :code:`Region` 的入口块。因此，:code:`Region` 的参数用入口块参数即可描述，不需要额外处理。

当 :code:`Region` 的一次执行结束，控制流由子 :code:`Block` 返回到该 :code:`Region` 时，控制流会有两种去处：

* 进入同 Op 的某一个 :code:`Region` (可能是自己)。
* 返回该 :code:`Region` 的父 Op，表示该 Op 的一次执行的结束。

具体去处由该 :code:`Region` 的父 Op 的语意决定。

注: 在控制流之前， 一个 :code:`Operation` 由它的输入、输出、属性以及类型信息构成。 加入控制流以后，一个 :code:`Operation` 的内容包含：它的输入(OpOperand)、输出（OpResult）、属性(AttributeMap)、后继块（BlockOperand）、:code:`Region` 组成。 新增了后继块和 :code:`Region` 。


.. _api_guide_Block:

=========
Block
=========

:code:`Block` 等价于基本块， 里面包含了一个算子列表(``std::list<Operaiton*>``)， 用来表示该基本块的计算语意。

当 :code:`Block` 的最后一个算子执行结束时，根据块内最后一个算子(终止符算子)的语意，控制流会有两种去处：

* 进入同 :code:`Region` 的另外一个 :code:`Block` , 该 :code:`Block` 一定是终止符算子的后继块。
* 返回该 :code:`Block` 的父 :code:`Region` , 表示该 :code:`Region` 的一次执行的结束。

.. _api_guide_Operation:

=============
Operation
=============

算子( :code:`Operation` )是有向图的节点。 算子信息分为四部分：输入(OpOperandImpl)、输出(OpResultImpl)、属性(Attribute)、类型信息(OpInfo)。 其中，输入和输出的数量因为在构造的时候才能确定，而且构造完以后，数量就不会再改变。

Attribute 来描述一个属性。用户可以在算子中临时存储一些运行时属性，但是运行时属性只能用来辅助计算，不允许改变计算语意。模型在导出时，默认会裁剪掉所有的运行时属性。

算子类型信息(OpInfo)，本质上是对相同类型的算子所具有的公共性质的抽象。

.. _api_guide_Weight:

=============
Weight
=============

权重属性是一种特殊的属性，权重属性的数据量一般会非常大，:code:`Paddle` 将权重单独存储，在模型中通过权重名对权重值进行获取和保存。目前 :code:`Paddle` 所有模型的权重都是 ``Variable`` 类型。

.. _api_guide_Operator:

=============
Operator
=============

在 :code:`Paddle` 中，所有对数据的操作都由 :code:`Operator` 表示 每个 :code:`Operator` 执行特定的功能，如矩阵乘法、卷积、激活函数等，通过组合这些 :code:`Operator`，可以构建复杂的计算图，实现模型的前向传播和反向传播。



.. _api_guide_Variable:

=========
Variable
=========

:code:`Paddle` 中的 :code:`Variable` 可以包含任何类型的值———在大多数情况下是一个 :ref:`Tensor <cn_user_guide_tensor>` 。

模型中所有的可学习参数都以 :code:`Variable` 的形式保留在内存空间中，您在绝大多数情况下都不需要自己来创建网络中的可学习参数， :code:`Paddle` 为几乎常见的神经网络基本计算模块都提供了封装。以静态图中最简单的全连接模型为例，调用 :code:`paddle.static.nn.fc` 会直接为全连接层创建连接权值( W )和偏置（ bias ）两个可学习参数，无需显示地调用 :code:`variable` 相关接口创建可学习参数。

.. _api_guide_Name:

=========
Name
=========

:code:`Paddle` 中部分网络层里包含了 :code:`name` 参数，如 :ref:`cn_api_static_nn_fc` 。此 :code:`name` 一般用来作为网络层输出、权重的前缀标识，具体规则如下：

* 用于网络层输出的前缀标识。若网络层中指定了 :code:`name` 参数，:code:`Paddle` 将以 ``name 值.tmp_数字`` 作为唯一标识对网络层输出进行命名；未指定 :code:`name` 参数时，则以 ``OP 名_数字.tmp_数字`` 的方式进行命名，其中的数字会自动递增，以区分同名 OP 下的不同网络层。

* 用于权重或偏置变量的前缀标识。若在网络层中通过 ``param_attr`` 和 ``bias_attr`` 创建了权重变量或偏置变量， 如 :ref:`cn_api_nn_embedding` 、 :ref:`cn_api_static_nn_fc` ，则 :code:`Paddle` 会自动生成 ``前缀.w_数字`` 或 ``前缀.b_数字`` 的唯一标识对其进行命名，其中 ``前缀`` 为用户指定的 :code:`name` 或自动生成的 ``OP 名_数字`` 。若在 ``param_attr`` 和 ``bias_attr`` 中指定了 :code:`name` ，则用此 :code:`name` ，不再自动生成。细节请参考示例代码。

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


上述示例中， ``fc_none`` 和 ``fc_none1`` 均未指定 :code:`name` 参数，则以 ``OP 名_数字.tmp_数字`` 分别对该 OP 输出进行命名：``fc_0.tmp_1`` 和 ``fc_1.tmp_1`` ，其中 ``fc_0``  和 ``fc_1`` 中的数字自动递增以区分两个全连接层； ``my_fc1`` 和 ``my_fc2`` 均指定了 :code:`name` 参数，但取值相同，:code:`Paddle` 以后缀 ``tmp_数字`` 进行区分，即 ``my_fc.tmp_1`` 和 ``my_fc.tmp_3`` 。

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
