.. _cn_api_fluid_layers_sequence_softmax:

sequence_softmax
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_softmax(input, use_cudnn=False, name=None)

**注意：该OP的输入只能是LoDTensor，如果要处理的输入是Tensor类型，请使用paddle.fluid.layers.softmax()**

该OP的输入类型为LoDTensor，LoDTensor中包含LoD信息和Tensor信息，该OP根据LoD信息对Tensor的第0维度进行划分，在划分的每一个区间内部进行运算。

对第i个区间内的元素的计算公式如下：

.. math::

    Out\left ( X[lod[i]:lod[i+1]],: \right ) = \frac{exp(X[lod[i]:lod[i+1],:])}{\sum (exp(X[lod[i]:lod[i+1],:]))}

输入张量的维度可为 :math:`[N，1]` 或者 :math:`[N]` 。

例如，对有3个样本的batch，每个样本的长度为2，3，2，其lod信息为[0,2,5,7]，根据lod信息将第0维度划分为3份，在:math:`X[0:2,:], X[2:5,:], X[5:7,:]`  中进行softmax运算。

参数：
    - **input** (Variable) - 维度为 math:[N, 1] 或者 math:[N] 的LoDTensor，数据类型为float32或float64。
    - **use_cudnn** (bool，可选) - 是否用cudnn核，仅当下载cudnn库且在gpu训练的时候生效。才有效。数据类型为bool型，默认：False。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。若设为None，则自动为该层命名。数据类型为string，默认：None。

返回：根据区间计算softmax之后的LoDTensor，其维度与input的维度一致，数据类型与input的数据类型一致。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_sequence_softmax = fluid.layers.sequence_softmax(input=x)










