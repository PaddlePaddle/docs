.. _cn_api_fluid_dygraph_FC:

FC
-------------------------------

.. py:class:: paddle.fluid.dygraph.FC(name_scope, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, is_test=False, dtype='float32')


**全连接层**

该接口将在神经网络中构建一个全连接层。其输入可以是一个 ``Tensor`` 或多个 ``Tensor`` 组成的list（详见参数说明），该接口会为每个输入的Tensor创建一个权重（weights）变量，即一个从每个输入单元到每个输出单元的全连接权重矩阵。全连接层将每个输入Tensor和其对应的权重(weights)相乘得到shape为 :math:`[M, size]` 输出Tensor，其中 ``M`` 为batch_size大小。如果有多个输入Tensor，则多个shape为 :math:`[M, size]` 的Tensor计算结果会被累加起来，作为最终输出。如果 ``bias_attr`` 非空，则会创建一个偏置变量（bias Variable），并把它累加到输出结果中。如果 ``act`` 非空，将会在输出结果上应用相应的激活函数。

当输入为单个 ``Tensor`` ：

.. math::

        \\Out = Act({XW + b})\\



当输入为多个 ``Tensor`` 组成的list时：

.. math::

        \\Out=Act(\sum^{N-1}_{i=0}X_iW_i+b) \\


上述等式中：
  - :math:`N` ：输入的数目,如果输入是Tensor列表，N等于len（input）
  - :math:`X_i` ：第i个输入的Tensor
  - :math:`W_i` ：对应第i个输入张量的第i个权重矩阵
  - :math:`b` ：该层创建的bias参数
  - :math:`Act` ：激活函数
  - :math:`Out` ：输出Tensor

::
            
        Case 1： 
            给定单个输入Tensor data_1, 且num_flatten_dims = 2:
                data_1.data = [[[0.1, 0.2],
                               [0.3, 0.4]]]
                data_1.shape = (1, 2, 2) # 1是batch_size
                
                fc = FC("fc", 1, num_flatten_dims=2)
                out = fc(data_1)

            则输出为：
                out.data = [[0.83234344], [0.34936576]]
                out.shape = (1, 2, 1)


        Case 2: 
            给定多个Tensor组成的list:
                data_1.data = [[[0.1, 0.2],
                               [0.3, 0.4]]]
                data_1.shape = (1, 2, 2) # 1 是 batch_size

                data_2 = [[[0.1, 0.2, 0.3]]]
                data_2.shape = (1, 1, 3)

                fc = FC("fc", 2)
                out = fc([data_1, data_2])

            则输出为：
                out.data = [[0.18669507, 0.1893476]]
                out.shape = (1, 2)

参数:
  - **name_scope** (str) – 类的名称。
  - **size** (int) – 全连接层输出单元的数目，即输出 ``Tensor`` 的特征维度。
  - **num_flatten_dims** (int, 可选) – fc层可以接受一个维度大于2的tensor。此时， 它首先会被扁平化(flattened)为一个二维矩阵。 参数 ``num_flatten_dims`` 决定了输入tensor的flattened方式: 前 ``num_flatten_dims`` (包含边界，从1开始数) 个维度会被扁平化为最终矩阵的第一维 (维度即为矩阵的高), 剩下的 rank(X) - num_flatten_dims 维被扁平化为最终矩阵的第二维 (即矩阵的宽)。 例如， 假设X是一个五维tensor，其形可描述为[2, 3, 4, 5, 6], 且num_flatten_dims = 3。那么扁平化的矩阵形状将会如此： [2 x 3 x 4, 5 x 6] = [24, 30]。默认为1。
  - **param_attr** (ParamAttr|list of ParamAttr, 可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr|list of ParamAttr, 可选) – 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **act** (str, 可选) – 应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations` ，默认值为None。
  - **is_test** (bool, 可选) – 表明当前执行是否处于测试阶段的标志。默认为False。
  - **dtype** (str, 可选) – 权重的数据类型，可以为float32或float64。默认为float32。

返回：无

**代码示例**

..  code-block:: python

    from paddle.fluid.dygraph.base import to_variable
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import FC
    import numpy as np

    data = np.random.uniform( -1, 1, [30, 10, 32] ).astype('float32')
    with fluid.dygraph.guard():
        fc = FC( "fc", 64, num_flatten_dims=2)
        data = to_variable(data)
        conv = fc(data)

