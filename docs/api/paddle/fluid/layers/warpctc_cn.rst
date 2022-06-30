.. _cn_api_fluid_layers_warpctc:

warpctc
-------------------------------

.. py:function:: paddle.fluid.layers.warpctc(input, label, blank=0, norm_by_times=False, input_length=None, label_length=None)




该OP用于计算 `CTC loss <https://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 。该OP的底层调用了第三方 `baidu-research::warp-ctc <https://github.com/baidu-research/warp-ctc>`_ 的实现。

参数
::::::::::::

    - **input** (Variable) - 可以是3-D Tensor或2-D LoDTensor。当输入类型是3-D Tensor时，则表示输入是经过padding的定长序列，其 shape 必须是 ``[seq_length, batch_size, num_classes + 1]``。当输入类型是2-D LoDTensor时，则表示输入为变长序列，其shape必须为 ``[Lp，num_classes+1]`` ， ``Lp`` 是所有输入序列长度之和。以上 shape 中的 ``num_classes`` 是实际类别数，不包括空白标签。该输入不需要经过 softmax 操作，因为该OP的内部对 ``input`` 做了 softmax 操作。数据类型仅支持float32。
    - **label** (Variable) - 可以是3-D Tensor或2-D LoDTensor，需要跟 ``input`` 保持一致。当输入类型为3-D Tensor时，表示输入是经过 padding 的定长序列，其 shape 为 ``[batch_size, label_length]``，其中，``label_length`` 是最长的 label 序列的长度。当输入类型是2-D LoDTensor时，则表示输入为变长序列，其shape必须为 ``[Lp, 1]``，其中 ``Lp`` 是所有 label 序列的长度和。``label`` 中的数值为字符ID。数据类型支持int32。 
    - **blank** (int，可选) - 空格标记的ID，其取值范围为 ``[0，num_classes+1)``。数据类型支持int32。缺省值为0。
    - **norm_by_times** (bool，可选) - 是否根据序列长度对梯度进行正则化。数据类型支持 bool。缺省值为False。 
    - **input_length** (Variable) - 必须是1-D Tensor。仅在输入为定长序列时使用，表示输入数据中每个序列的长度，shape为 ``[batch_size]``。数据类型支持int64。默认为None。
    - **label_length** (Variable) - 必须是1-D Tensor。仅在label为定长序列时使用，表示 label 中每个序列的长度，shape为 ``[batch_size]``。数据类型支持int64。默认为None。

返回
::::::::::::
Shape为[batch_size，1]的2-D Tensor，表示每一个序列的CTC loss。数据类型与 ``input`` 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    # using LoDTensor
    import paddle.fluid as fluid
    import numpy as np

    # lengths of logit sequences
    seq_lens = [2,6]
    # lengths of label sequences
    label_lens = [2,3]
    # class num
    class_num = 5

    logits = fluid.data(name='logits',shape=[None, class_num+1],
                        dtype='float32',lod_level=1)
    label = fluid.data(name='label', shape=[None, 1],
                       dtype='int32', lod_level=1)
    cost = fluid.layers.warpctc(input=logits, label=label)
    place = fluid.CPUPlace()
    x = fluid.create_lod_tensor(
             np.random.rand(np.sum(seq_lens), class_num+1).astype("float32"),
             [seq_lens], place)
    y = fluid.create_lod_tensor(
             np.random.randint(0, class_num, [np.sum(label_lens), 1]).astype("int32"),
             [label_lens], place)
    exe = fluid.Executor(place)
    output= exe.run(fluid.default_main_program(),
                    feed={"logits": x,"label": y},
                    fetch_list=[cost.name])
    print(output)

COPY-FROM: paddle.fluid.layers.warpctc