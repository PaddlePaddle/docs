.. _cn_api_fluid_layers_Decoder:

Decoder
-------------------------------



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


方法
::::::::::::
initialize(inits)
'''''''''

在解码迭代之前调用一次。
    
**参数**
  
  - **inits** - 调用方提供的参数。
    
**返回**
一个元组 :code:`(initial_inputs, initial_states, finished)` 。:code:`initial_inputs` 和 :code:`initial_states` 都是单个tensor变量或tensor变量组成的嵌套结构，:code:`finished` 是具有bool数据类型的tensor。

**返回类型**
tuple

step(time, inputs, states, **kwargs)
'''''''''

在解码的每个时间步中被调用的接口

**参数**
  
  - **time** (Variable) - 调用者提供的形状为[1]的tensor，表示当前解码的时间步长。其数据类型为int64。。
  - **inputs** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。在第一个解码时间步时与由 :code:`initialize()` 返回的 :code:`initial_inputs` 相同，其他时间步与由 :code:`step()` 返回的 :code:`next_inputs` 相同。
  - **states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。在第一个解码时间步时与 :code:`initialize()` 返回的 :code:`initial_states` 相同，其他时间步与由 :code:`step()` 返回的 :code:`beam_search_state` 相同。
  - **kwargs** - 附加的关键字参数，由调用者提供。

**返回**
一个元组 :code:`(outputs, next_states, next_inputs, finished)` 。:code:`next_states` 和 :code:`next_inputs` 都是单个tensor变量或tensor变量组成的嵌套结构，且结构、形状和数据类型均分别与输入参数中的 :code:`states` 和 :code:`inputs` 相同。:code:`outputs` 是单个tensor变量或tensor变量组成的嵌套结构。:code:`finished` 是一个bool类型的tensor变量。

**返回类型**
tuple

finalize(self, outputs, final_states, sequence_lengths)
'''''''''

如果提供了实现，将在整个解码迭代结束后被执行一次。

**参数**
  
  - **outputs** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。其中每个tensor的形状均为 :math:`[time\_step，batch\_size，...]`，是将所有解码步中与其对应的的输出进行堆叠的结果，这个过程由其调用者完成。
  - **final_states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构。它是 :code:`decoder.step` 在最后一个解码步返回的 :code:`next_states`，因此具有与任何时间步的状态相同的结构，形状和数据类型。
  - **kwargs** - 命名关键字参数，由提供调用者。

**返回**
一个元组 :code:`(final_outputs, final_states)` 。:code:`final_outputs` 和 :code:`final_states` 都是单个tensor变量或tensor变量组成的嵌套结构。

**返回类型**
tuple