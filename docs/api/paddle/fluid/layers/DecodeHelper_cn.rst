.. _cn_api_fluid_layers_DecodeHelper:

DecodeHelper
-------------------------------


.. py:class:: paddle.fluid.layers.DecodeHelper()

DecodeHelper是一个基类，其子类的实例将在 :ref:`cn_api_fluid_layers_BasicDecoder` 中使用。它提供了在动态解码时采样和产生下一解码步的输入的接口。

方法
::::::::::::
initialize()
'''''''''

初始化以产生第一个解码步的输入和每个序列是否结束的初始标识。这是 :ref:`cn_api_fluid_layers_BasicDecoder` 初始化的一部分。

**返回**
:code:`(initial_inputs, initial_finished)` 的二元组，:code:`initial_inputs` 是单个tensor变量或tensor变量组成的嵌套结构，tensor的形状是 :math:`[batch\_size, ...]` 。 :code:`initial_finished` 是一个bool类型且形状为 :math:`[batch\_size]` 的tensor。

**返回类型**
tuple
    
sample(time, outputs, states)
'''''''''

根据 :code:`outputs` 以特定的方式进行采样，该方法是 :code:`BasicDecoder.step` 中的一部分。

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

产生下一解码步的输入、状态，以及每个序列是否结束的标识。该方法是 :code:`BasicDecoder.step` 中的一部分。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的tensor，表示当前解码的时间步长。其数据类型为int64。
  - **outputs** (Variable) - tensor变量，通常其数据类型为float32或float64，形状为 :math:`[batch\_size, vocabulary\_size]`，表示当前解码步预测产生的logit（未归一化的概率），和由 :code:`BasicDecoder.output_fn(BasicDecoder.cell.call())` 返回的 :code:`outputs` 是同一内容。
  - **states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构，和由 :code:`BasicDecoder.cell.call()` 返回的 :code:`new_states` 是同一内容。
  - **sample_ids** (Variable) - 数据类型为int64形状为 :math:`[batch\_size]` 的tensor，和由 :code:`sample()` 返回的 :code:`sample_ids` 是同一内容。

**返回**
 :code:`(finished, next_inputs, next_states)` 的三元组。:code:`next_inputs, next_states` 均是单个tensor变量或tensor变量组成的嵌套结构，:code:`next_states` 和输入参数中的 :code:`states` 具有相同的结构、形状和数据类型；:code:`finished` 是一个bool类型且形状为 :math:`[batch\_size]` 的tensor。

**返回类型**
tuple
