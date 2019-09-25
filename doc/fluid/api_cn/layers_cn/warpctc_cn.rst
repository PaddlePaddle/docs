.. _cn_api_fluid_layers_warpctc:

warpctc
-------------------------------

.. py:function:: paddle.fluid.layers.warpctc(input, label, blank=0, norm_by_times=False, use_cudnn=False)

该OP用于计算 `CTC loss <https://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 。该OP的底层调用了第三方 `baidu-research::warp-ctc <https://github.com/baidu-research/warp-ctc>`_ 的实现。

参数：
    - **input** (Variable) - 可以是3-D Tensor或2-D LoDTensor。当输入类型是3-D Tensor时，则表示输入是经过padding的定长序列，其shape必须是 ``[seq_length, batch_size, num_classes + 1]`` 。当输入类型是2-D LoDTensor时，则表示输入为变长序列，其shape必须为 ``[Lp，num_classes+1]`` ， ``Lp`` 是所有输入序列长度之和。以上shape中的 ``num_classes`` 是实际类别数，不包括空白标签。该输入不需要经过softmax操作，因为该OP的内部对 ``input`` 做了softmax操作。数据类型仅支持float32。
    - **label** (Variable) - 可以是2-D Tensor或2-D LoDTensor，shape为 ``[Lg，1]`` ，其中 ``Lg`` 为是所有labels序列的长度和。 ``label`` 中的数值为字符ID。当label是2-D Tensor时，则表示当前label为经过padding的定长序列。当label是2-D LoDTensor时，则表示当前label为变长序列。数据类型支持int32。
    - **input_length** (Variable) - 必须是1-D Tensor。仅在输入为定长序列时使用，表示输入数据中每个序列的长度，shape为 ``[batch_size]`` 。数据类型支持int64。默认为None。
    - **label_length** (Variable) - 必须是1-D Tensor。仅在label为定长序列时使用，表示label中每个序列的长度，shape为 ``[batch_size]`` 。数据类型支持int64。默认为None。
    - **blank** (int，可选) - 空格标记的ID，其取值范围为 ``[0，num_classes+1)`` 。数据类型支持int32。缺省值为0。
    - **norm_by_times** (bool，可选) - 是否根据序列长度对梯度进行正则化。数据类型支持bool。缺省值为False。

返回：Shape为[batch_size，1]的2-D Tensor，表示每一个序列的CTC loss。数据类型与 ``input`` 一致。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name='label', shape=[11, 8],
                              dtype='float32', lod_level=1)
    predict = fluid.layers.data(name='predict', shape=[11, 1],
                                dtype='float32')
    cost = fluid.layers.warpctc(input=predict, label=label)
