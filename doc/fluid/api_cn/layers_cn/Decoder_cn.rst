.. _cn_api_fluid_layers_Decoder:

Decoder
-------------------------------


:api_attr: 声明式编程模式(静态图)

.. py:class:: paddle.fluid.layers.Decoder()

    
Decoder是dynamic_decode中使用的任何decoder实例的基类。它提供了为每一个时间步生成输出的接口，可用于生成序列。

Decoder提供的主要抽象为：

1. :code:`(initial_input, initial_state, finished) = initialize(inits)`，
为第一个解码步生成输入和状态，并给出指示batch中的每个序列是否结束的初始标识。

2. :code:`(output, next_state, next_input, finished) = step(time, input, state)`，
将输入和状态转换为输出和新的状态，为下一个解码步生成输入，并给出指示batch中的每个序列是否结束的标识。

3. :code:`(final_outputs, final_state) = finalize(outputs, final_state, sequence_lengths)`，
修改输出（所有时间步输出的堆叠）和最后的状态以做特殊用途。若无需修改堆叠得到的输出和来自最后一个时间步的状态，则无需实现。

与RNNCell相比，Decoder更为通用，因为返回的 :code:`next_input` 和 :code:`finished` 使它可以自行决定输入以及结束时机。


.. py:method:: initialize(inits)

在解码迭代之前调用一次。
    
参数：  
  - **inits** - 调用方提供的参数。 
    
返回：一个元组 :code:`(initial_inputs, initial_states, finished)` 。:code:`initial_inputs` 和 :code:`initial_states` 都是单个tensor变量或tensor变量组成的嵌套结构， :code:`finished` 是具有bool数据类型的tensor。

返回类型：tuple

.. py:method:: step(time, inputs, states)

在解码的每个时间步中被调用的接口

参数：  
  - **outputs** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。 结构和数据类型与 :code:`output_dtype` 相同。 tensor堆叠所有时间步长的输出从而具有shape :math:`[time\_step，batch\_size，...]` ，由调用者完成。 
  - **final_states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。 它是 :code:`decoder.step` 在最后一个解码步返回的 :code:`next_states`， 因此具有与任何时间步长的状态相同的结构，形状和数据类型。

返回：一个元组 :code:`(final_outputs, final_states)` 。:code:`final_outputs` 和 :code:`final_states` 都是单个tensor变量或tensor变量组成的嵌套结构。

返回类型：tuple