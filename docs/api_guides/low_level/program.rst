基础概念
=======

Program
-------

:code:`Fluid` 使用类似编程语言的抽象语法树（AST）来描述神经网络配置，用户对计算的描述会被写入 :code:`Program` 中。在 Fluid 中，:code:`Program` 替代了传统框架中的“模型”概念。它通过三种执行结构：顺序执行、条件选择和循环执行，来表示复杂的模型。编写 :code:`Program` 类似于编写普通程序，如果你曾经编程过，你可以直接运用这些经验。

总结：

- Fluid 中的模型是通过 :code:`Program` 表示的，一个 :code:`Program` 可以包含多个嵌套的 :code:`Program`。
- :code:`Program` 由多个嵌套的 :code:`Block` 构成，:code:`Block` 的概念可以类比为 C++ 或 Java 中的一对大括号，或 Python 中的缩进块。
- :code:`Block` 中的计算通过顺序执行、条件选择或循环执行三种方式进行，这三者共同构成复杂的计算逻辑。
- :code:`Block` 包含计算描述和计算对象。计算的描述称为 Operator，计算对象（即 Operator 的输入和输出）统一表示为 Tensor。在 Fluid 中，Tensor 由 0 级 `LoD-Tensor <http://paddlepaddle.org/documentation/docs/zh/1.2/user_guides/howto/prepare_data/lod_tensor.html#permalink-4-lod-tensor>`_ 表示。

Block
------

:code:`Block` 是一个变量作用域的概念，类似于高级编程语言中的块结构。在编程语言中，块由一对大括号定义，包含局部变量定义和一系列指令或操作符。在深度学习中，控制流结构如编程语言中的 :code:`if-else` 和 :code:`for` 循环在 Fluid 中被映射为相应的操作符。

+----------------------+-------------------------+
| 编程语言       | Fluid                      |
+----------------+----------------------------+
| for、while 循环 | RNN、WhileOP               |
+----------------+----------------------------+
| if-else、switch | IfElseOp、SwitchOp         |
+----------------+----------------------------+
| 顺序执行       | 一系列的层                 |
+----------------+----------------------------+

如上所述，Fluid 中的 :code:`Block` 定义了一组操作符，这些操作符包括顺序执行、条件选择和循环执行，操作对象则是 Tensor。

Operator
---------

在 Fluid 中，所有的数据操作都由 :code:`Operator` 表示。在 Python 中，这些 :code:`Operator` 被封装成模块，如 :code:`paddle.fluid.layers` 和 :code:`paddle.fluid.nets`。

一些常见的 Tensor 操作可能由更基础的操作构成。为了简化开发，Fluid 内部对这些基础操作进行了封装，包括学习参数的创建、初始化等，从而减少了用户开发的工作量。

更多详情请参考 `Fluid Design Idea <../../advanced_usage/design_idea/fluid_design_idea.html>`_。

Variable
--------

在 Fluid 中，:code:`Variable` 可以包含任何类型的值，在大多数情况下是 LoD-Tensor。

所有模型中的可学习参数都以 :code:`Variable` 形式存储在内存中。在大多数情况下，你不需要手动创建网络中的可学习参数。Fluid 提供了几乎所有常见的神经网络基本计算模块的封装。例如，在全连接层中，调用 :code:`fluid.layers.fc` 就会自动创建该层的两个可学习参数：连接权重（W）和偏置，而不需要显式调用 :code:`Variable` 接口来创建它们。

Name
-----

在 Fluid 中，某些层包含 :code:`name` 参数，如 :ref:`api_fluid_layers_fc`。该 :code:`name` 参数通常用作网络层中输出和权重的前缀标识。命名规则如下：

- **输出层的前缀标识**：如果在层中指定了 :code:`name`，Fluid 会将输出命名为 ``nameValue.tmp_number``。如果未指定 :code:`name`，则会自动生成 ``OPName_number.tmp_number`` 来命名该层，其中的数字会自动递增，以区分同一个操作符下的不同网络层。
- **权重或偏置变量的前缀标识**：如果权重和偏置变量是通过 ``param_attr`` 和 ``bias_attr`` 在操作符中创建的，如 :ref:`api_fluid_layers_embedding` 和 :ref:`api_fluid_layers_fc`，Fluid 会生成 ``prefix.w_number`` 或 ``prefix.b_number`` 作为唯一标识来命名它们，其中 ``prefix`` 是用户指定的 :code:`name` 或默认生成的 ``OPName_number``。如果在 ``param_attr`` 和 ``bias_attr`` 中指定了 :code:`name`，则不会自动生成 :code:`name`。具体示例代码如下。

示例代码：

```python
import paddle.fluid as fluid
import numpy as np

x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=1)
emb = fluid.layers.embedding(input=x, size=(128, 100))  # embedding_0.w_0
emb = fluid.layers.Print(emb) # Tensor[embedding_0.tmp_0]

# 默认名称
fc_none = fluid.layers.fc(input=emb, size=1)  # fc_0.w_0, fc_0.b_0
fc_none = fluid.layers.Print(fc_none)  # Tensor[fc_0.tmp_1]

fc_none1 = fluid.layers.fc(input=emb, size=1)  # fc_1.w_0, fc_1.b_0
fc_none1 = fluid.layers.Print(fc_none1)  # Tensor[fc_1.tmp_1]

# ParamAttr 中的名称
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


示例说明：
* ``fc_none`` 与 ``fc_none1`` 未设置 :code:`name`，输出分别命名为 ``fc_0.tmp_1``、``fc_1.tmp_1``；
* ``my_fc1`` 和 ``my_fc2`` 使用了相同的 :code:`name`，系统自动区分为 ``my_fc.tmp_1`` 和 ``my_fc.tmp_3``；
* 权重变量命名为指定的 ``fc_weight``，偏置命名为 ``my_fc.b_0`` 和 ``my_fc.b_1``；
* 通过 ``ParamAttr`` 指定同一 name，实现了 ``my_fc1`` 与 ``my_fc2`` 权重共享。

.. _api_guide_ParamAttr:

=========
ParamAttr
=========

:code:`ParamAttr` 是一个用于控制参数属性的辅助类，可设置参数名称、学习率、是否可训练等属性。在网络层中通过 :code:`param_attr` 参数传入。

更多详细信息，请参阅 :ref:`cn_api_fluid_ParamAttr`。

=========
相关 API
=========

* 用户定义的神经网络由 :ref:`cn_api_fluid_Program` 构建。通常一个训练流程中涉及多个 :code:`Program`，如参数初始化用、训练用、测试用等；
* 使用 :ref:`cn_api_fluid_program_guard` 配合 :code:`with` 语句，可对默认的 :ref:`cn_api_fluid_default_startup_program` 和 :ref:`cn_api_fluid_default_main_program` 进行切换；
* 在 Fluid 中，控制流执行顺序通过以下 API 实现：
  - :ref:`cn_api_fluid_layers_IfElse`
  - :ref:`cn_api_fluid_layers_While`
  - :ref:`cn_api_fluid_layers_Switch`
  更多内容详见 :ref:`api_guide_control_flow`。
