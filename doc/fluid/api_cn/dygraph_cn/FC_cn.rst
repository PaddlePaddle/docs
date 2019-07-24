.. _cn_api_fluid_dygraph_FC:

FC
-------------------------------

.. py:class:: paddle.fluid.dygraph.FC(name_scope, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, is_test=False, dtype='float32')




**全连接层**

该函数在神经网络中建立一个全连接层。 它可以将一个或多个tensor（ ``input`` 可以是一个list或者Variable，详见参数说明）作为自己的输入，并为每个输入的tensor创立一个变量，称为“权”（weights），等价于一个从每个输入单元到每个输出单元的全连接权矩阵。FC层用每个tensor和它对应的权相乘得到形状为[M, size]输出tensor，M是批大小。如果有多个输入tensor，那么形状为[M, size]的多个输出张量的结果将会被加起来。如果 ``bias_attr`` 非空，则会新创建一个偏向变量（bias variable），并把它加入到输出结果的运算中。最后，如果 ``act`` 非空，它也会加入最终输出的计算中。

当输入为单个张量：

.. math::

        \\Out = Act({XW + b})\\



当输入为多个张量：

.. math::

        \\Out=Act(\sum^{N-1}_{i=0}X_iW_i+b) \\


上述等式中：
  - :math:`N` ：输入的数目,如果输入是变量列表，N等于len（input）
  - :math:`X_i` ：第i个输入的tensor
  - :math:`W_i` ：对应第i个输入张量的第i个权重矩阵
  - :math:`b` ：该层创立的bias参数
  - :math:`Act` ：activation function(激励函数)
  - :math:`Out` ：输出tensor

::

            Given:
                data_1.data = [[[0.1, 0.2],
                               [0.3, 0.4]]]
                data_1.shape = (1, 2, 2) # 1 is batch_size

                data_2 = [[[0.1, 0.2, 0.3]]]
                data_2.shape = (1, 1, 3)

                out = fluid.layers.fc(input=[data_1, data_2], size=2)

            Then:
                out.data = [[0.18669507, 0.1893476]]
                out.shape = (1, 2)


参数:
  - **name_scope** (str) – 该类的名称
  - **size** (int) – 该层输出单元的数目
  - **num_flatten_dims** (int, 默认为1) – fc层可以接受一个维度大于2的tensor。此时， 它首先会被扁平化(flattened)为一个二维矩阵。 参数 ``num_flatten_dims`` 决定了输入tensor的flattened方式: 前 ``num_flatten_dims`` (包含边界，从1开始数) 个维度会被扁平化为最终矩阵的第一维 (维度即为矩阵的高), 剩下的 rank(X) - num_flatten_dims 维被扁平化为最终矩阵的第二维 (即矩阵的宽)。 例如， 假设X是一个五维tensor，其形可描述为(2, 3, 4, 5, 6), 且num_flatten_dims = 3。那么扁平化的矩阵形状将会如此： (2 x 3 x 4, 5 x 6) = (24, 30)
  - **param_attr** (ParamAttr|list of ParamAttr|None) – 该层可学习的参数/权的参数属性
  - **bias_attr** (ParamAttr|list of ParamAttr, default None) – 该层bias变量的参数属性。如果值为False， 则bias变量不参与输出单元运算。 如果值为None，bias变量被初始化为0。默认为 None。
  - **act** (str|None) – 应用于输出的Activation（激励函数）
  - **is_test** (bool) – 表明当前执行是否处于测试阶段的标志
  - **dtype** (str) – 权重的数据类型


弹出异常：``ValueError`` - 如果输入tensor的维度小于2

**代码示例**

..  code-block:: python

    from paddle.fluid.dygraph.base import to_variable
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import FC
    import numpy as np

    data = np.random.uniform( -1, 1, [30, 10, 32] ).astype('float32')
    with fluid.dygraph.guard():
        fc = FC( "fc", 64, num_flatten_dims=2)
        data = to_variable( data )
        conv = fc( data )




