.. _cn_api_fluid_layers_GreedyEmbeddingHelper:

GreedyEmbeddingHelper
-------------------------------


.. py:class:: paddle.fluid.layers.GreedyEmbeddingHelper(embedding_fn, start_tokens, end_token)

GreedyEmbeddingHelper是 :ref:`cn_api_fluid_layers_DecodeHelper` 的子类。作为解码helper，它使用 :code:`argmax` 进行采样，并将采样结果送入embedding层，以此作为下一解码步的输入。

参数
::::::::::::

  - **embedding_fn** (callable) - 作用于 :code:`argmax` 结果的函数，通常是一个将词id转换为词嵌入的embedding层，**注意**，这里要使用 :ref:`cn_api_fluid_embedding` 而非 :ref:`cn_api_fluid_layers_embedding`，因为选中的id的形状是 :math:`[batch\_size]`，如果使用后者则还需要在这里提供unsqueeze。
  - **start_tokens** (Variable) - 形状为 :math:`[batch\_size]` 、数据类型为int64、 值为起始标记id的tensor。
  - **end_token** (int) - 结束标记id。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.GreedyEmbeddingHelper

方法
::::::::::::
initialize()
'''''''''

GreedyEmbeddingHelper初始化，其使用构造函数中的 :code:`start_tokens` 作为第一个解码步的输入，并给出每个序列是否结束的初始标识。这是 :ref:`cn_api_fluid_layers_BasicDecoder` 初始化的一部分。

**返回**
:code:`(initial_inputs, initial_finished)` 的二元组，:code:`initial_inputs` 同构造函数中的 :code:`start_tokens` ； :code:`initial_finished` 是一个bool类型、值为False的tensor，其形状和 :code:`start_tokens` 相同。

**返回类型**
tuple
    
sample(time, outputs, states)
'''''''''

使用 :code:`argmax` 根据 `outputs` 进行采样。

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

对 :code:`sample_ids` 使用 :code:`embedding_fn`，以此作为下一解码步的输入；同时直接使用输入参数中的 :code:`states` 作为下一解码步的状态；并通过判别 :code:`sample_ids` 是否得到 :code:`end_token`，依此产生每个序列是否结束的标识。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的tensor，表示当前解码的时间步长。其数据类型为int64。
  - **outputs** (Variable) - tensor变量，通常其数据类型为float32或float64，形状为 :math:`[batch\_size, vocabulary\_size]`，表示当前解码步预测产生的logit（未归一化的概率），和由 :code:`BasicDecoder.output_fn(BasicDecoder.cell.call())` 返回的 :code:`outputs` 是同一内容。
  - **states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构，和由 :code:`BasicDecoder.cell.call()` 返回的 :code:`new_states` 是同一内容。
  - **sample_ids** (Variable) - 数据类型为int64形状为 :math:`[batch\_size]` 的tensor，和由 :code:`sample()` 返回的 :code:`sample_ids` 是同一内容。

**返回**
 :code:`(finished, next_inputs, next_states)` 的三元组。:code:`next_inputs, next_states` 均是单个tensor变量或tensor变量组成的嵌套结构，tensor的形状是 :math:`[batch\_size, ...]` ， :code:`next_states` 和输入参数中的 :code:`states` 相同；:code:`finished` 是一个bool类型且形状为 :math:`[batch\_size]` 的tensor。

**返回类型**
tuple
