.. _cn_api_fluid_layers_dynamic_decode:

dynamic_decode
-------------------------------



.. py:method:: dynamic_decode(decoder, inits=None, max_step_num=None, output_time_major=False, **kwargs):

:api_attr: 声明式编程模式（静态图)


    
该接口重复执行 :code:`decoder.step()` 直到 其返回的表示完成状态的Tensor中的值全部为True或解码步骤达到 :code:`max_step_num`。

:code:`decode.initialize()` 会在解码循环之前被调用一次。如果 :code:`decoder` 实现了 :code:`finalize` 方法，则 :code:`decoder.finalize()` 在解码循环后将被调用一次。

参数:
  - **decoder** (Decoder) - 解码器的实例。
  - **inits** (object，可选) - 传递给 :code:`decoder.initialize` 的参数。默认为None。
  - **max_step_num** (int，可选) - 最大步数。如果未提供，解码直到解码过程完成（ :code:`decode.step()` 返回的表示完成状态的Tensor中的值全部为True）。默认为None。
  - **output_time_major** (bool，可选) - 指明最终输出(此方法的第一个返回值)中包含的Tensor的数据布局。如果为False，其将使用batch优先的数据布局, 此时的形状为 :math:`[batch\_size，seq\_len，...]`。如果为True，其将使用time优先的数据布局，此时的形状为 :math:`[seq\_len，batch\_size，...]`。默认值为False。
  - **kwargs** - 其他命名关键字参数。这些参数将传递给 :code:`decoder.step`。

返回:一个二元组 :code:`(final_outputs，final_states)`, 其包含了最终的输出和状态，这两者都是Tensor或Tensor的嵌套结构。:code:`final_outputs` 具有与 :code:`decoder.output_dtype` 相同的结构和数据类型， 其中的每个tensor都是对所有解码时间步对应输出的堆叠。 这些tensor也可能会通过 :code:`decoder.finalize` 进行修改。:code:`final_states` 是最后时间步的状态，和 :code:`decoder.initialize` 返回的初始状态具有相同的结构，其中的tensor也具有相同的形状 和数据类型。

返回类型：tuple

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    from paddle.fluid.layers import GRUCell, BeamSearchDecoder, dynamic_decode
    encoder_output = fluid.data(name="encoder_output",
                            shape=[-1, 32, 128],
                            dtype="float32")
    trg_embeder = lambda x: fluid.embedding(
        x, size=[10000, 128], param_attr=fluid.ParamAttr(name="trg_embedding"))
    output_layer = lambda x: layers.fc(x,
                                    size=10000,
                                    num_flatten_dims=len(x.shape) - 1,
                                    param_attr=fluid.ParamAttr(name=
                                                                "output_w"),
                                    bias_attr=False)
    decoder_cell = GRUCell(hidden_size=128)
    decoder = BeamSearchDecoder(decoder_cell,
                                start_token=0,
                                end_token=1,
                                beam_size=4,
                                embedding_fn=trg_embeder,
                                output_fn=output_layer)
    outputs = dynamic_decode(	
        decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output))
