.. _cn_api_fluid_layers_rnn:

rnn
-------------------------------



方法
::::::::::::
paddle.fluid.layers.rnn(cell, inputs, initial_states=None, sequence_length=None, time_major=False, is_reverse=False, **kwargs)
'''''''''



    

rnn创建一个由RNNCell :code:`cell` 指定的递归神经网络，该神经网络重复执行 :code:`cell.call()` 直至达到 :code:`inputs` 的最大长度。

**参数**

  - **cell** (RNNCell) - RNNCell的实例。
  - **inputs** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。当 :code:`time_major == False` 时，tensor的形状应为 :math:`[batch\_size, sequence\_length, ...]`；当 :code:`time_major == True` 时，tensor的形状应为 :math:`[sequence\_length, batch\_size, ...]`。它表示要在RNN中展开的输入。
  - **initial_states** (Variable，可选) - 初始状态，单个tensor变量或tensor变量组成的嵌套结构，表示RNN的初始状态。如果未提供，将使用 :code:`cell.get_initial_states` 产生初始状态。默认值None。
  - **sequence_length** (Variable，可选) - 序列长度，形状为 :math:`[batch\_size]` 的tensor。它存储每个实例的实际长度，从而使用户能够在批处理的时候，提取最后一个有效状态，以确保正确性。如果未提供，则不区分填充和非填充输入。默认值None。
  - **time_major** (bool，可选) - 指示输入tensor和输出tensor中包含的tensor的数据组织。如果为False，则数据组织为batch为主，形状为 :math:`[batch\_size，sequence\_length，...]`。如果为True，则数据组织为time为主，形状为 :math:`[sequence\_length，batch\_size，...]`。默认值：False。
  - **is_reverse** (bool，可选) - 指示是否以输入序列的相反顺序进行计算，为True时表示以输入序列的相反顺序进行计算。默认值：False。
  - **kwargs** - 其他关键字参数。参数传递给 :code:`cell.call`。
  
**返回**
一个元组 :code:`(final_outputs, final_states)`，包括 :code:`final_outputs` 和 :code:`final_states`，均为单个tensor变量或tensor变量的嵌套结构。:code:`final_outputs` 具有与 :code:`cell.call` 返回的 :code:`outputs` 相同的结构和数据类型，并且 :code:`final_outputs` 中的每个tensor是将所有时间步的 :code:`outputs` 中对应内容堆叠产生，因此其形状为 :math:`[batch\_size，sequence\_length，...]` （:code:`time_major == False` 时）或 :math:`[sequence\_length，batch\_size，...]` （:code:`time_major == True` 时）。:code:`final_states` 是最后一步的状态，因此具有和 :code:`initial_states` 相同的结构，形状和数据类型。

**返回类型**
tuple

**代码示例**
  
.. code-block:: python
            
  import paddle.fluid as fluid
  inputs = fluid.data(name="inputs",
                      shape=[-1, 32, 128],
                      dtype="float32")
  cell = fluid.layers.GRUCell(hidden_size=128)
  outputs = fluid.layers.rnn(cell=cell, inputs=inputs)

