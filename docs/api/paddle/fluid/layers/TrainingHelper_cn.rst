.. _cn_api_fluid_layers_TrainingHelper:

TrainingHelper
-------------------------------


.. py:class:: paddle.fluid.layers.TrainingHelper(inputs, sequence_length, time_major=False)

TrainingHelper是 :ref:`cn_api_fluid_layers_DecodeHelper` 的子类。作为解码helper，它在每个解码时间步通过在完整序列输入 :code:`inputs` 的相应位置切片作为各步的输入，并且使用 :code:`argmax` 根据 :code:`cell.call()` 的输出进行采样。
由于要求有完整的序列输入 :code:`inputs` ，TrainingHelper主要用于以teach-forcing的方式进行最大似然训练，采样得到的内容通常不会使用。

参数
::::::::::::

  - **inputs** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。当 :code:`time_major == False` 时，tensor的形状应为 :math:`[batch\_size, sequence\_length, ...]`；当 :code:`time_major == True` 时，tensor的形状应为 :math:`[sequence\_length, batch\_size, ...]`。在解码的每一步都要从中切片取出相应的数据。
  - **sequence_length** (Variable) - 形状为 :math:`[batch\_size]` 的tensor。它存储了 :code:`inputs` 中每个样本的实际长度，可以据此来标识每个解码步中每个样本是否结束。
  - **time_major** (bool，可选) - 指示输入tensor和输出tensor中包含的tensor的数据组织。如果为False，则数据组织为batch为主，形状为 :math:`[batch\_size，sequence\_length，...]`。如果为True，则数据组织为time为主，形状为 :math:`[sequence\_length，batch\_size，...]`。默认值：False。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.TrainingHelper

方法
::::::::::::
initialize()
'''''''''

TrainingHelper初始化，其通过在完整序列输入 :code:`inputs` 中首个时间步的位置上切片，以此作为第一个解码步的输入，并给出每个序列是否结束的初始标识。这是 :ref:`cn_api_fluid_layers_BasicDecoder` 初始化的一部分。

**返回**
:code:`(initial_inputs, initial_finished)` 的二元组，:code:`initial_inputs` 是单个tensor变量或tensor变量组成的嵌套结构，tensor的形状是 :math:`[batch\_size, ...]` 。 :code:`initial_finished` 是一个bool类型且形状为 :math:`[batch\_size]` 的tensor。

**返回类型**
tuple
    
sample(time, outputs, states)
'''''''''

使用 :code:`argmax` 根据 `outputs` 进行采样。由于使用完整序列中的切片作为下一解码步的输入，采样得到的内容通常不会使用。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的tensor，表示当前解码的时间步长。其数据类型为int64。
  - **outputs** (Variable) - tensor变量，通常其数据类型为float32或float64，形状为 :math:`[batch\_size, vocabulary\_size]`，表示当前解码步预测产生的logit（未归一化的概率），和由 :code:`BasicDecoder.output_fn(BasicDecoder.cell.call())` 返回的 :code:`outputs` 是同一内容。
  - **states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构，和由 :code:`BasicDecoder.cell.call()` 返回的 :code:`new_states` 是同一内容。

**返回**
数据类型为int64形状为 :math:`[batch\_size]` 的tensor，表示采样得到的id。

**返回类型**
Variable        

next_inputs(time, outputs, states, sample_ids)
'''''''''

从完整序列输入中当前时间步的位置上切片，以此作为产生下一解码步的输入；同时直接使用输入参数中的 :code:`states` 作为下一解码步的状态；并比较当前时间与每个序列的大小，依此产生每个序列是否结束的标识。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的tensor，表示当前解码的时间步长。其数据类型为int64。
  - **outputs** (Variable) - tensor变量，通常其数据类型为float32或float64，形状为 :math:`[batch\_size, vocabulary\_size]`，表示当前解码步预测产生的logit（未归一化的概率），和由 :code:`BasicDecoder.output_fn(BasicDecoder.cell.call())` 返回的 :code:`outputs` 是同一内容。
  - **states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构，和由 :code:`BasicDecoder.cell.call()` 返回的 :code:`new_states` 是同一内容。
  - **sample_ids** (Variable) - 数据类型为int64形状为 :math:`[batch\_size]` 的tensor，和由 :code:`sample()` 返回的 :code:`sample_ids` 是同一内容。

**返回**
 :code:`(finished, next_inputs, next_states)` 的三元组。:code:`next_inputs, next_states` 均是单个tensor变量或tensor变量组成的嵌套结构，tensor的形状是 :math:`[batch\_size, ...]` ， :code:`next_states` 和输入参数中的 :code:`states` 相同；:code:`finished` 是一个bool类型且形状为 :math:`[batch\_size]` 的tensor。

**返回类型**
tuple
