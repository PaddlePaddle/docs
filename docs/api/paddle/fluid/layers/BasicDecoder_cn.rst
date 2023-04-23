.. _cn_api_fluid_layers_BasicDecoder:

BasicDecoder
-------------------------------


.. py:class:: paddle.fluid.layers.BasicDecoder(cell, helper, output_fn=None)

BasicDecoder是 :ref:`cn_api_fluid_layers_Decoder` 的子类，它组装了 :ref:`cn_api_fluid_layers_RNNCell` 和 :ref:`cn_api_fluid_layers_DecodeHelper` 的实例作为成员，其中DecodeHelper用来实现不同的解码策略。它依次执行以下步骤来完成单步解码：

1. 执行 :code:`cell_outputs, cell_states = cell.call(inputs, states)` 以获取输出和新的状态。

2. 执行 :code:`sample_ids = helper.sample(time, cell_outputs, cell_states)` 以采样id并将其作为当前步的解码结果。

3. 执行 :code:`finished, next_inputs, next_states = helper.next_inputs(time, cell_outputs, cell_states, sample_ids)` 以产生下一解码步的结束标识、输入和状态。

参数
::::::::::::

  - **cell** (RNNCell) - RNNCell的实例或者具有相同接口定义的对象。
  - **helper** (DecodeHelper) - DecodeHelper的实例。
  - **output_fn** (可选) - 处理cell输出的接口，在采样之前使用。默认值None。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.BasicDecoder

方法
::::::::::::
initialize(initial_cell_states)
'''''''''

初始化，包括helper的初始化和cell的初始化，cell初始化直接使用 :code:`initial_cell_states` 作为结果。

**参数**

  - **initial_cell_states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。这是由调用者 :ref:`cn_api_fluid_layers_dynamic_decode` 提供的参数。

**返回**
:code:`(initial_inputs, initial_states, finished)` 的三元组。:code:`initial_inputs, initial_states` 均是单个tensor变量或tensor变量组成的嵌套结构，:code:`finished` 是bool类型的tensor。 :code:`initial_inputs, finished` 与 :code:`helper.initialize()` 返回的内容相同；:code:`initial_states` 与输入参数中的 :code:`initial_cell_states` 的相同。

**返回类型**
tuple
    
.. py:class:: OutputWrapper(cell_outputs, sample_ids)

 :code:`step()` 的返回值中 :code:`outputs` 使用的数据结构，是一个由 :code:`cell_outputs` 和 :code:`sample_ids` 这两个字段构成的命名元组。

step(time, inputs, states, **kwargs)
'''''''''

按照以下步骤执行单步解码：

1. 执行 :code:`cell_outputs, cell_states = cell.call(inputs, states)` 以获取输出和新的状态。

2. 执行 :code:`sample_ids = helper.sample(time, cell_outputs, cell_states)` 以采样id并将其作为当前步的解码结果。

3. 执行 :code:`finished, next_inputs, next_states = helper.next_inputs(time, cell_outputs, cell_states, sample_ids)` 以产生下一解码步的结束标识、输入和状态。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的tensor，表示当前解码的时间步长。其数据类型为int64。
  - **inputs** (Variable) - tensor变量。在第一个解码时间步时与由 :code:`initialize()` 返回的 :code:`initial_inputs` 相同，其他时间步与由 :code:`step()` 返回的 :code:`next_inputs` 相同。
  - **states** (Variable) - tensor变量的结构。在第一个解码时间步时与 :code:`initialize()` 返回的 :code:`initial_states` 相同，其他时间步与由 :code:`step()` 返回的 :code:`next_states` 相同。
  - **kwargs** - 附加的关键字参数，由调用者 :ref:`cn_api_fluid_layers_dynamic_decode` 提供。

**返回**
 :code:`(outputs, next_states, next_inputs, finished)` 的四元组。:code:`outputs` 是包含 :code:`cell_outputs` 和 :code:`sample_ids` 两个字段的命名元组，其中 :code:`cell_outputs` 是 :code:`cell.call()` 的结果，:code:`sample_ids` 是 :code:`helper.sample()` 的结果；:code:`next_states, next_inputs` 分别和输入参数中的 :code:`states, inputs` 有相同的的结构、形状和数据类型；:code:`finished` 是一个bool类型的tensor，形状是 :math:`[batch\_size]` 。

**返回类型**
tuple
