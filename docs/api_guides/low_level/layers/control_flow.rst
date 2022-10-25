.. _api_guide_control_flow:

######
控制流
######

在程序语言中，控制流(control flow)决定了语句的执行顺序，常见的控制流包括顺序执行、分支和循环等。PaddlePaddle Fluid 继承了这一概念，提供了多种控制流 API, 以控制深度学习模型在训练或者预测过程中的执行逻辑。

IfElse
======

条件分支，允许对同一个 batch 的输入，根据给定的条件，分别选择 :code:`true_block` 或 :code:`false_block` 中的逻辑进行执行，执行完成之后再将两个分支的输出合并为同一个输出。通常，条件表达式可由 :ref:`cn_api_fluid_layers_less_than`, :ref:`cn_api_fluid_layers_equal` 等逻辑比较 API 产生。

请参考 :ref:`cn_api_fluid_layers_IfElse`

**注意：** 强烈建议您使用新的 OP :ref:`cn_api_fluid_layers_cond` 而不是 ``IfElse``。:ref:`cn_api_fluid_layers_cond` 的使用方式更简单，并且调用该 OP 所用的代码更少且功能与 ``IfElse`` 一样。

Switch
======

多分支选择结构，如同程序语言中常见的 :code:`switch-case` 声明, 其根据输入表达式的取值不同，选择不同的分支执行。具体来说，Fluid 所定义的 :code:`Switch` 控制流有如下特性：

* case 的条件是个 bool 类型的值，即在 Program 中是一个张量类型的 Variable；
* 依次检查逐个 case，选择第一个满足条件的 case 执行，完成执行后即退出所属的 block；
* 如果所有 case 均不满足条件，会选择默认的 case 进行执行。

请参考 :ref:`cn_api_fluid_layers_Switch`

**注意：** 强烈建议您使用新的 OP :ref:`cn_api_fluid_layers_case` 而不是 ``Switch``。 :ref:`cn_api_fluid_layers_case` 的使用方式更简单，并且调用该 OP 所用的代码更少且功能与 ``Switch`` 一样。

While
=====

While 循环，当条件判断为真时，循环执行 :code:`While` 控制流所属 :code:`block` 内的逻辑，条件判断为假时退出循环。与之相关的 API 有

* :ref:`cn_api_fluid_layers_increment` ：累加 API，通常用于对循环次数进行计数；
* :ref:`cn_api_fluid_layers_array_read` ：从 :code:`LOD_TENSOR_ARRAY` 中指定的位置读入 Variable，进行计算；
* :ref:`cn_api_fluid_layers_array_write` ：将 Variable 写回到 :code:`LOD_TENSOR_ARRAY` 指定的位置，存储计算结果。

请参考 :ref:`cn_api_fluid_layers_While`

**注意：** 强烈建议您使用新的 OP :ref:`cn_api_fluid_layers_while_loop` 而不是 ``While``。 :ref:`cn_api_fluid_layers_while_loop` 的使用方式更简单，并且调用该 OP 所用的代码更少且功能与 ``While`` 一样。

DynamicRNN
==========

即动态 RNN，可处理一个 batch 不等长的序列数据，其接受 :code:`lod_level=1` 的 Variable 作为输入，在 :code:`DynamicRNN` 的 :code:`block` 内，用户需自定义 RNN 的单步计算逻辑。在每一个时间步，用户可将需记忆的状态写入到 :code:`DynamicRNN` 的 :code:`memory` 中，并将需要的输出写出到其 :code:`output` 中。

:ref:`cn_api_fluid_layers_sequence_last_step` 可获取 :code:`DynamicRNN` 最后一个时间步的输出。

请参考 :ref:`cn_api_fluid_layers_DynamicRNN`

StaticRNN
=========

即静态 RNN，只能处理固定长度的序列数据，接受 :code:`lod_level=0` 的 Variable 作为输入。与 :code:`DynamicRNN` 类似，在 RNN 的每单个时间步，用户需自定义计算逻辑，并可将状态和输出写出。

请参考 :ref:`cn_api_fluid_layers_StaticRNN`
