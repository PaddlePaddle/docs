.. _cn_api_fluid_Program:

Program
-------------------------------

.. py:class::  paddle.static.Program


.. note::
默认情况下，Paddle 内部默认含有 :ref:`cn_api_fluid_default_startup_program` 和 :ref:`cn_api_fluid_default_main_program`，它们共享参数。:ref:`cn_api_fluid_default_startup_program` 只运行一次来初始化参数，:ref:`cn_api_fluid_default_main_program` 在每个 mini batch 中运行并更新权重。

Program 是 Paddle 对于计算图的一种静态描述，使用 Program 的构造函数可以创建一个 Program。Program 中包括至少一个 :ref:`api_guide_Block`，当 :ref:`api_guide_Block` 中存在条件选择的控制流 OP（例如 :ref:`cn_api_fluid_layers_While` 等）时，该 Program 将会含有嵌套着的 :ref:`api_guide_Block` 即控制流外部的 :ref:`api_guide_Block` 将包含着控制流内部的 :ref:`api_guide_Block`，而嵌套的 :ref:`api_guide_Block` 的元素访问控制将由具体的控制流 OP 来决定。关于 Program 具体的结构和包含的类型请参阅 `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_
。

一个 Program 的集合通常包含初始化程序（startup_program）与主程序（main_program），初始化程序是一个包含一些初始化工作的 Program，主程序将会包含用来训练的网络结构和变量，在使用同一个 :ref:`api_guide_executor` 执行时他们会共享初始化工作的结果，例如初始化的参数。一个 Program 的集合可以被用来测试或者训练，被用来训练时，``Paddle`` 将会利用所有用户使用的 OP 和变量来搭建一个训练网络，被用来测试时，可以通过调用 Program 相关的接口例如：``clone`` 剪去一些与测试无关的 OP 和变量，比如反向传播的 OP 和变量。


返回
:::::::::
Program，创建的空的 Program。

代码示例
::::::::::

COPY-FROM: paddle.static.Program


方法
::::::::::::
to_string(throw_on_error, with_details=False)
'''''''''

将 Program 转换为字符串。

**参数**

 - **throw_on_error** (bool) - 是否在没有设置必需字段时抛出异常。
 - **with_details** (bool) - 值为 true 时，打印更多关于变量和参数的信息，如 ``trainable``, ``optimize_attr`` 等。

**返回**

str，由 Program 转换得到的字符串。


**代码示例**

COPY-FROM: paddle.static.Program.to_string

clone(for_test=False)
'''''''''

.. note::
    1. ``Program.clone()`` 方法不会克隆例如 :ref:`cn_api_fluid_io_DataLoader` 这样的数据读取相关的部分，这可能会造成的数据读取部分在克隆后丢失；
    2. 此 API 当 ``for_test=True`` 时将会裁剪部分 OP 和变量。为防止错误的裁剪，推荐在 :ref:`cn_api_fluid_backward_append_backward` 和执行优化器之前使用；``clone(for_test=True)`` 。


当 ``for_test=True`` 时创建一个新的、仅包含当前 Program 前向内容的 Program。否则创建一个新的和当前 Program 完全相同的 Program。

有些 OP，在训练和测试之间的行为是不同的，比如 :ref:`cn_api_fluid_layers_batch_norm`。它们有一个属性 ``is_test`` 来控制行为。当 ``for_test=True`` 时，此方法将把它们的 ``is_test`` 属性更改为 True。

- 克隆 Program 用于训练时，将 ``for_test`` 设置为 False。
- 克隆 Program 用于测试时，将 ``for_test`` 设置为 True。虽然在这种情况下，如果在使用了优化器之后调用 ``clone`` 我们依旧会对 Program 当中反向执行以及优化器相关的内容进行自动裁剪，但是，我们强烈建议在使用优化器之前使用 ``clone`` 例如如果使用的是 :ref:`cn_api_paddle_optimizer_Momentum` 可以这样去使用：

**代码示例**

COPY-FROM: paddle.static.Program.clone:code-example-1

**参数**

    - **for_test** (bool) – 取值为 True 时，clone 方法内部会把 operator 的属性 ``is_test`` 设置为 True，并裁剪反向 OP 和参数优化 OP，默认值为 False。

**返回**

Program，当 ``for_test=True`` 时返回一个新的、仅包含当前 Program 前向内容的 Program。否则返回一个新的，和当前 Program 完全相同的 Program。


**代码示例**

.. note::
    Program 在 clone 后的顺序可能不同，这不会影响的训练或测试进程。在下面的示例中，我们提供了一个简单的方法 print_prog（Program）来打印程序描述，以确保 clone 后仍能得到同样的打印结果：

COPY-FROM: paddle.static.Program.clone:code-example-2

1. 克隆一个 Program，示例代码如下。

COPY-FROM: paddle.static.Program.clone:code-example3

2. 如果分别运行 train Program 和 test Program，则可以不使用 clone。

COPY-FROM: paddle.static.Program.clone:code-example4

上边两个代码片段生成和打印的 Program 是一样的。

**static** parse_from_string(binary_str)
'''''''''

通过对 `protobuf <https://en.wikipedia.org/wiki/Protocol_Buffers>`_ 的反序列化，转换成 Program。


**参数**

 - **binary_str** (str) – `protobuf <https://en.wikipedia.org/wiki/Protocol_Buffers>`_ 二进制字符串。

**返回**

Program，反序列化后的 Program。

**代码示例**

COPY-FROM: paddle.static.Program.parse_from_string

属性
::::::::::::
num_blocks
'''''''''

该 Program 中的 :ref:`api_guide_Block` 的个数。

**返回**

int，该 Program 中的 :ref:`api_guide_Block` 的个数。

**代码示例**

COPY-FROM: paddle.static.Program.num_blocks

random_seed
'''''''''

.. note::
    必须在相关 OP 被添加之前设置。

程序中随机运算符的默认随机种子。0 意味着随机生成随机种子。

**返回**

int64，该 Program 中当前正在使用的 random seed。

**代码示例**

COPY-FROM: paddle.static.Program.random_seed

global_block()
'''''''''

获取该 Program 的第一个 :ref:`api_guide_Block` 。

**返回**

:ref:`api_guide_Block`，该 Program 的第一个 :ref:`api_guide_Block`。

**代码示例**

COPY-FROM: paddle.static.Program.global_block


block(index)
'''''''''

返回该 Program 中，``index`` 指定的 :ref:`api_guide_Block` 。 ``index`` 类型为 ``int`` 。

**参数**

    - **index** (int) - 需要获取的 :ref:`api_guide_Block`  的 index。

**返回**

:ref:`api_guide_Block`，该 Program 中 index 对应的那个 :ref:`api_guide_Block`。

**代码示例**

COPY-FROM: paddle.static.Program.block


current_block()
'''''''''

获取当前 :ref:`api_guide_Block`。当前 :ref:`api_guide_Block`  是用来添加 OP 的。

**返回**

:ref:`api_guide_Block`，该 Program 中用户当前所在的 :ref:`api_guide_Block`。

**代码示例**

COPY-FROM: paddle.static.Program.current_block


list_vars()
'''''''''

获取当前 Program 中所有变量。返回值是一个可迭代对象（iterable object)。

**返回**

Generator，会 yield 每个 Program 中的变量。

**代码示例**

COPY-FROM: paddle.static.Program.list_vars


all_parameters()
'''''''''

获取当前 Program 中所有的 :ref:`api_guide_parameter`。返回值是一个列表。

**返回**

list[ :ref:`api_guide_parameter` ]，一个包含当前 Program 中所有参数的列表。

**代码示例**

COPY-FROM: paddle.static.Program.all_parameters


state_dict(mode='all', scope=None)
'''''''''

获取当前 ``Program`` 持久性变量。并将所有持久性变量存放在 dict 结构中。

**参数**

    - **mode** (str，可选) - 获取何种持久性变量。目前支持以下选项：(1) ``opt``：获得优化器的持久性变量放在 ``dict`` 结构中；(2) ``param``：获得组网中的持久性变量放在 ``dict`` 结构中，不包含优化器中的持久性变量；(3) ``all``：获得组网和优化器中的持久性变量放在 dict 结构中；默认值为 ``all``。
    - **scope** (Scope，可选) - 如果 scope 为 ``None``，通过 `paddle.static.global_scope()` 获取全局/默认作用域实例，并从中获取 ``state_dict``；否则从指定的 ``scope`` 获取 ``state_dict``。默认值为 ``None`` 。

**返回**

dict，包含持久性变量的 dict，键值是持久性变量的名字，值为持久性变量。

**代码示例**

COPY-FROM: paddle.static.Program.state_dict


set_state_dict(state_dict, scope=None)
'''''''''

将 ``state_dict`` 中的持久性变量设置到 ``Program`` 中。

**参数**

    - **state_dict** (dict) - 包含持久性变量的字典。键值是持久性变量的名字，值为持久性变量。
    - **scope** (Scope，可选) - 如果 scope 为 ``None``，通过 `paddle.static.global_scope()` 获取全局/默认作用域实例，并将 ``state_dict`` 中久性变量设置到这个作用域中；否则将 ``state_dict`` 设置到指定的 ``scope`` 中。默认值为 ``None`` 。

**返回**

无。

**代码示例**

COPY-FROM: paddle.static.Program.set_state_dict
