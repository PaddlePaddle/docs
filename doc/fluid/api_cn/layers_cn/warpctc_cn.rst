.. _cn_api_fluid_layers_warpctc:

warpctc
-------------------------------

.. py:function:: paddle.fluid.layers.warpctc(input, label, blank=0, norm_by_times=False, use_cudnn=False)

该操作符集成了 `开源Warp-CTC库 <https://github.com/baidu-research/warp-ctc>`_ ，计算基于神经网络的时序类分类（CTC）损失。原生softmax激活函数集成到Wrap-CTC库中，操作符也可称作含CTC的softmax，将输入张量每一行的值正则化。

参数：
    - **input** （Variable） - 变长序列的非尺度化概率，是一个含LoD信息的二维张量。shape为[Lp，num_classes+1]，Lp是所有输出序列长度之和，num_classes是实际类别数。（不包括空白标签）
    - **label** (Variable） - 变长序列中正确标记的数据，是一个含LoD信息的二维张量。shape为[Lg，1]，Lg是所有标签长度之和
    - **blank** （int，默认0） - 基于神经网络的时序类分类（CTC）损失的空白标签索引，在半开区间间隔内[0，num_classes+1]
    - **norm_by_times** （bool，默认false） - 是否利用时间步长（即序列长度）的数量对梯度进行正则化。如果warpctc层后面跟着mean_op则无需对梯度正则化。
    - **use_cudnn** (bool, 默认false) - 是否使用cudnn

返回：基于神经网络的时序类分类（CTC）损失，是一个shape为[batch_size，1]的二维张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name='label', shape=[11, 8],
                              dtype='float32', lod_level=1)
    predict = fluid.layers.data(name='predict', shape=[11, 1],
                                dtype='float32')
    cost = fluid.layers.warpctc(input=predict, label=label)




