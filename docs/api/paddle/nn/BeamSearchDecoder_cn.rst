.. _cn_api_fluid_layers_BeamSearchDecoder:

BeamSearchDecoder
-------------------------------



.. py:class:: paddle.nn.BeamSearchDecoder(cell, start_token, end_token, beam_size, embedding_fn=None, output_fn=None)




带 beam search 解码策略的解码器。该接口包装一个 cell 来计算概率，然后执行一个 beam search 步骤计算得分，并为每个解码步骤选择候选输出。更多详细信息请参阅 `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_

.. note::
    在使用 beam search 解码时，cell 的输入和状态将被扩展到 :math:`beam\_size`，得到 :math:`[batch\_size * beam\_size, ...]` 一样的形状，这个操作在 BeamSearchDecoder 中自动完成，因此，其他任何在 :code:`cell.call` 中使用的 Tensor，如果形状为 :math:`[batch\_size, ...]`，都必须先手动使用 :code:`BeamSearchDecoder.tile_beam_merge_with_batch` 接口扩展。最常见的情况是带注意机制的编码器输出。

参数
::::::::::::

  - **cell** (RNNCell) - RNNCell 的实例或者具有相同接口定义的对象。
  - **start_token** (int) - 起始标记 id。
  - **end_token** (int) - 结束标记 id。
  - **beam_size** (int) - 在 beam search 中使用的 beam 宽度。
  - **embedding_fn** (可选) - 处理选中的候选 id 的接口。它通常是一个将词 id 转换为词嵌入的嵌入层，其返回值将作为 :code:`cell.call` 接口的 :code:`input` 参数。**注意**，这里要使用 :ref:`cn_api_nn_Embedding` 而非 :ref:`cn_api_fluid_layers_embedding`，因为选中的 id 的形状是 :math:`[batch\_size, beam\_size]`，如果使用后者则还需要在这里提供 unsqueeze。如果 :code:`embedding_fn` 未提供，则必须在 :code:`cell.call` 中实现词嵌入转换。默认值 None。
  - **output_fn** (可选) - 处理 cell 输出的接口，在计算得分和选择候选标记 id 之前使用。默认值 None。

返回
::::::::::::
BeamSearchDecoder 的一个实例，可以用于传入 `paddle.nn.dynamic\_decode` 以实现解码过程。

代码示例
::::::::::::


COPY-FROM: paddle.nn.BeamSearchDecoder

方法
::::::::::::
tile_beam_merge_with_batch(x, beam_size)
'''''''''

扩展 Tensor 的 batch 维度。此函数的输入是形状为 :math:`[batch\_size, s_0, s_1, ...]` 的 Tensor t，由 minibatch 中的样本 :math:`t[0], ..., t[batch\_size - 1]` 组成。将其扩展为形状是 :math:`[batch\_size * beam\_size, s_0, s_1, ...]` 的 Tensor，由 :math:`t[0], t[0], ..., t[1], t[1], ...` 组成，每个 minibatch 中的样本重复 :math:`beam\_size` 次。

**参数**

  - **x** (Variable) - 形状为 :math:`[batch\_size, ...]` 的 tenosr。数据类型应为 float32，float64，int32，int64 或 bool。
  - **beam_size** (int) - 在 beam search 中使用的 beam 宽度。

**返回**

Tensor，形状为 :math:`[batch\_size * beam\_size, ...]` 的 Tensor，其数据类型与 :code:`x` 相同。


_split_batch_beams(x)
'''''''''

将形状为 :math:`[batch\_size * beam\_size, ...]` 的 Tensor 变换为形状为 :math:`[batch\_size, beam\_size, ...]` 的新 Tensor。

**参数**

  - **x** (Variable) - 形状为 :math:`[batch\_size * beam\_size, ...]` 的 tenosr。数据类型应为 float32，float64，int32，int64 或 bool。

**返回**

Tensor，形状为 :math:`[batch\_size, beam\_size, ...]` 的 Tensor，其数据类型与 :code:`x` 相同。

_merge_batch_beams(x)
'''''''''

将形状为 :math:`[batch\_size, beam\_size, ...]` 的 Tensor 变换为形状为 :math:`[batch\_size * beam\_size,...]` 的新 Tensor。

**参数**

  - **x** (Variable) - 形状为 :math:`[batch\_size, beam_size,...]` 的 Tenosr。数据类型应为 float32，float64，int32，int64 或 bool。

**返回**

Tensor，形状为 :math:`[batch\_size * beam\_size, ...]` 的 Tensor，其数据类型与 :code:`x` 相同。

_expand_to_beam_size(x)
'''''''''

此函数输入形状为 :math:`[batch\_size,s_0，s_1，...]` 的 Tensor t，由 minibatch 中的样本 :math:`t[0]，...，t[batch\_size-1]` 组成。将其扩展为形状 :math:`[ batch\_size,beam\_size,s_0，s_1，...]` 的 Tensor，由 :math:`t[0]，t[0]，...，t[1]，t[1]，...` 组成，其中每个 minibatch 中的样本重复 :math:`beam\_size` 次。

**参数**

  - **x** (Variable) - 形状为 :math:`[batch\_size, ...]` 的 tenosr。数据类型应为 float32，float64，int32，int64 或 bool。

**返回**

Tensor，具有与 :code:`x` 相同的形状和数据类型的 Tensor，其中未完成的 beam 保持不变，而已完成的 beam 被替换成特殊的 Tensor(Tensor 中所有概率质量被分配给 EOS 标记)。


_mask_probs(probs, finished)
'''''''''

屏蔽对数概率。该函数使已完成的 beam 将所有概率质量分配给 EOS 标记，而未完成的 beam 保持不变。

**参数**

  - **probs** (Variable) - 形状为 :math:`[batch\_size,beam\_size,vocab\_size]` 的 Tensor，表示对数概率。其数据类型应为 float32。
  - **finish** (Variable) - 形状为 :math:`[batch\_size,beam\_size]` 的 Tensor，表示所有 beam 的完成状态。其数据类型应为 bool。

**返回**

Tensor，具有与 :code:`x` 相同的形状和数据类型的 Tensor，其中未完成的 beam 保持不变，而已完成的 beam 被替换成特殊的 Tensor(Tensor 中所有概率质量被分配给 EOS 标记)。


_gather(x, indices, batch_size)
'''''''''

对 Tensor :code:`x` 根据索引 :code:`indices` 收集。

**参数**

  - **x** (Variable) - 形状为 :math:`[batch\_size, beam\_size,...]` 的 Tensor。
  - **index** (Variable) - 一个形状为 :math:`[batch\_size, beam\_size]` 的 int64 Tensor，表示我们用来收集的索引。
  - **batch_size** (Variable) - 形状为 :math:`[1]` 的 Tensor。其数据类型应为 int32 或 int64。

**返回**

Tensor，具有与 :code:`x` 相同的形状和数据类型的 Tensor，表示收集后的 Tensor。


initialize(initial_cell_states)
'''''''''

初始化 BeamSearchDecoder。

**参数**

  - **initial_cell_states** (Variable) - 单个 Tensor 变量或 Tensor 变量组成的嵌套结构。调用者提供的参数。

**返回**

tuple，一个元组 :code:`(initial_inputs, initial_states, finished)`。:code:`initial_inputs` 是一个 Tensor，当 :code:`embedding_fn` 为 None 时，该 Tensor t 的形状为 :math:`[batch\_size,beam\_size]`，值为 :code:`start_token`；否则使用 :code:`embedding_fn(t)` 返回的值。:code:`initial_states` 是 Tensor 变量的嵌套结构(命名元组，字段包括 :code:`cell_states，log_probs，finished，lengths`)，其中 :code:`log_probs，finished，lengths` 都含有一个 Tensor，形状为 :math:`[batch\_size, beam\_size]`，数据类型为 float32，bool，int64。:code:`cell_states` 具有与输入参数 :code:`initial_cell_states` 相同结构的值，但形状扩展为 :math:`[batch\_size,beam\_size,...]`。 :code:`finished` 是一个布尔型 Tensor，由 False 填充，形状为 :math:`[batch\_size,beam\_size]`。


_beam_search_step(time, logits, next_cell_states, beam_state)
'''''''''

计算得分并选择候选 id。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的 Tensor，表示当前解码的时间步长。其数据类型为 int64。
  - **logits** (Variable) - 形状为 :math:`[batch\_size,beam\_size,vocab\_size]` 的 Tensor，表示当前时间步的 logits。其数据类型为 float32。
  - **next_cell_states** (Variable) - 单个 Tensor 变量或 Tensor 变量组成的嵌套结构。它的结构，形状和数据类型与 :code:`initialize()` 的返回值 :code:`initial_states` 中的 :code:`cell_states` 相同。它代表该 cell 的下一个状态。
  - **beam_state** (Variable) - Tensor 变量的结构。在第一个解码步骤与 :code:`initialize()` 返回的 :code:`initial_states` 同，其他步骤与 :code:`step()` 返回的 :code:`beam_search_state` 相同。

**返回**

tuple，一个元组 :code:`(beam_search_output, beam_search_state)`。:code:`beam_search_output` 是 Tensor 变量的命名元组，字段为 :code:`scores，predicted_ids parent_ids`。其中 :code:`scores，predicted_ids，parent_ids` 都含有一个 Tensor，形状为 :math:`[batch\_size,beam\_size]`，数据类型为 float32 ，int64，int64。:code:`beam_search_state` 具有与输入参数 :code:`beam_state` 相同的结构，形状和数据类型。


step(time, inputs, states, **kwargs)
'''''''''

执行 beam search 解码步骤，该步骤使用 :code:`cell` 来计算概率，然后执行 beam search 步骤以计算得分并选择候选标记 ID。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的 Tensor，表示当前解码的时间步长。其数据类型为 int64。。
  - **inputs** (Variable) - Tensor 变量。在第一个解码时间步时与由 :code:`initialize()` 返回的 :code:`initial_inputs` 相同，其他时间步与由 :code:`step()` 返回的 :code:`next_inputs` 相同。
  - **states** (Variable) - Tensor 变量的结构。在第一个解码时间步时与 :code:`initialize()` 返回的 :code:`initial_states` 相同，其他时间步与由 :code:`step()` 返回的 :code:`beam_search_state` 相同。
  - **kwargs** - 附加的关键字参数，由调用者提供。

**返回**

tuple，一个元组 :code:`(beam_search_output，beam_search_state，next_inputs，finish)` 。:code:`beam_search_state` 和参数 :code:`states` 具有相同的结构，形状和数据类型。:code:`next_inputs` 与输入参数 :code:`inputs` 具有相同的结构，形状和数据类型。:code:`beam_search_output` 是 Tensor 变量的命名元组(字段包括 :code:`scores，predicted_ids，parent_ids` )，其中 :code:`scores，predicted_ids，parent_ids` 都含有一个 Tensor，形状为 :math:`[batch\_size,beam\_size]`，数据类型为 float32 ，int64，int64。:code:`finished` 是一个 bool 类型的 Tensor，形状为 :math:`[batch\_size,beam\_size]`。


finalize(outputs, final_states, sequence_lengths)
'''''''''

使用 :code:`gather_tree` 沿 beam search 树回溯并构建完整的预测序列。

**参数**

  - **outputs** (Variable) - Tensor 变量组成的结构(命名元组)，该结构和数据类型与 :code:`output_dtype` 相同。Tensor 将所有时间步的输出堆叠，因此具有形状 :math:`[time\_step，batch\_size,...]`。
  - **final_states** (Variable) - Tensor 变量组成的结构(命名元组)。它是 :code:`decoder.step` 在最后一个解码步骤返回的 :code:`next_states`，因此具有与任何时间步的 :code:`state` 相同的结构、形状和数据类型。
  - **sequence_lengths** (Variable) - Tensor，形状为 :math:`[batch\_size,beam\_size]`，数据类型为 int64。它包含解码期间确定的每个 beam 的序列长度。

**返回**

tuple，一个元组 :code:`(predicted_ids, final_states)`。:code:`predicted_ids` 是一个 Tensor，形状为 :math:`[time\_step，batch\_size,beam\_size]`，数据类型为 int64。:code:`final_states` 与输入参数 :code:`final_states` 相同。
