.. _cn_api_fluid_layers_sequence_softmax:

sequence_softmax
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_softmax(input, use_cudnn=False, name=None)

该函数计算每一个序列所有时间步中的softmax激活函数。每个时间步的维度应为1。

输入张量的形状可为 :math:`[N，1]` 或者 :math:`[N]` , :math:`N` 是所有序列长度之和。

对mini-batch的第i序列：

.. math::

    Out\left ( X[lod[i]:lod[i+1]],: \right ) = \frac{exp(X[lod[i]:lod[i+1],:])}{\sum (exp(X[lod[i]:lod[i+1],:]))}

例如，对有3个序列（可变长度）的mini-batch，每个包含2，3，2时间步，其lod为[0,2,5,7]，则在 :math:`X[0:2,:],X[2:5,:],X[5:7,:]` 中进行softmax运算，并且 :math:`N` 的结果为7.

参数：
    - **input** (Variable) - 输入变量，为LoDTensor
    - **use_cudnn** (bool) - 是否用cudnn核，仅当下载cudnn库才有效。默认：False
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。默认：None

返回：sequence_softmax的输出

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_sequence_softmax = fluid.layers.sequence_softmax(input=x)










