.. _cn_api_fluid_layers_dynamic_decode:

dynamic_decode
-------------------------------



.. py:method:: paddle.nn.dynamic_decode(decoder, inits=None, max_step_num=None, output_time_major=False, impute_finished=False, is_test=False, return_length=False, **kwargs):




该接口重复执行 :code:`decoder.step()` 直到 其返回的表示完成状态的Tensor中的值全部为True或解码步骤达到 :code:`max_step_num`。

:code:`decode.initialize()` 会在解码循环之前被调用一次。如果 :code:`decoder` 实现了 :code:`finalize` 方法，则 :code:`decoder.finalize()` 在解码循环后将被调用一次。

参数:
  - **decoder** (Decoder) - 解码器的实例。
  - **inits** (object，可选) - 传递给 :code:`decoder.initialize` 的参数。默认为None。
  - **max_step_num** (int，可选) - 最大步数。如果未提供，解码直到解码过程完成（ :code:`decode.step()` 返回的表示完成状态的Tensor中的值全部为True）。默认为None。
  - **output_time_major** (bool，可选) - 指明最终输出(此方法的第一个返回值)中包含的Tensor的数据布局。如果为False，其将使用batch优先的数据布局, 此时的形状为 :math:`[batch\_size，seq\_len，...]`。如果为True，其将使用time优先的数据布局，此时的形状为 :math:`[seq\_len，batch\_size，...]`。默认值为False。
  - **impute_finished** (bool，可选) - 若为True并且 :code:`decoder.tracks_own_finished` 为False，对于当前批次中完成状态为结束的样本，将会拷贝其上一步的状态，而非像未结束的实例那样使用 :code:`decode.step()` 返回的 :code:`next_states` 作为新的状态，这保证了返回的最终状态 :code:`final_states` 是正确的；否则，不会区分是否结束，也没有这个拷贝操作。若 :code:`final_states` 会被使用，则这里应该设置为True，这会一定程度上影响速度。默认为False。
  - **is_test** (bool，可选) - 标识是否是预测模式，预测模式下内存占用会更少。默认为False。
  - **return_length** (bool，可选) - 标识是否在返回的元组中额外包含一个存放了所有解码序列实际长度的Tensor。默认为False。
  - **kwargs** - 其他命名关键字参数。这些参数将传递给 :code:`decoder.step`。

返回：若 :code:`return_length` 为True，则返回三元组 :code:`(final_outputs, final_states, sequence_lengths)` ，否则返回二元组 :code:`(final_outputs, final_states)` 。 :code:`final_outputs, final_states` 包含了最终的输出和状态，这两者都是Tensor或Tensor的嵌套结构。:code:`final_outputs` 具有与 :code:`decoder.step()` 返回的 :code:`outputs` 相同的结构和数据类型， 且其中的每个tensor都是将所有解码步中与其对应的的输出进行堆叠的结果；如果 :code:`decoder` 实现了 :code:`finalize` 方法，这些tensor也可能会通过 :code:`decoder.finalize()` 进行修改。:code:`final_states` 是最后时间步的状态，和 :code:`decoder.initialize()` 返回的初始状态具有相同的结构，形状和数据类型。:code:`sequence_lengths` 是int64类型的tensor，和 :code:`decoder.initialize()` 返回的 :code:`finished` 具有相同的形状，其保存了所有解码序列实际长度。

返回类型：tuple

**示例代码**

.. code-block:: python

    import paddle
    from paddle.nn import BeamSearchDecoder, dynamic_decode
    from paddle.nn import GRUCell, Linear, Embedding
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = GRUCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(decoder_cell,
                                start_token=0,
                                end_token=1,
                                beam_size=4,
                                embedding_fn=trg_embeder,
                                output_fn=output_layer)
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder,
                            inits=decoder_cell.get_initial_states(encoder_output),
                            max_step_num=10)
