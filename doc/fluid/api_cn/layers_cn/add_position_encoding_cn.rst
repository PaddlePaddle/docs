.. _cn_api_fluid_layers_add_position_encoding:

add_position_encoding
-------------------------------

.. py:function:: paddle.fluid.layers.add_position_encoding(input, alpha, beta, name=None)

:alias_main: paddle.nn.functional.add_position_encoding
:alias: paddle.nn.functional.add_position_encoding,paddle.nn.functional.extension.add_position_encoding
:old_api: paddle.fluid.layers.add_position_encoding



该OP将输入inpu中每个位置（序列中的位置）的特征与对应的位置编码加权求和，位置编码可参考论文: `Attention Is All You Need <http://arxiv.org/pdf/1706.03762.pdf>`_

输出的计算公式如下：

.. math::

    PE(pos, 2i) &= \sin{(pos / 10000^{2i / P})}\\
    PE(pos, 2i + 1) &= \cos{(pos / 10000^{2i / P})}\\
    Out(:, pos, i) &= \alpha * input(:, pos, i) + \beta * PE(pos, i)

其中:
    - PE(pos, 2i): pos位置对应的编码中偶数特征位上的值
    - PE(pos, 2i + 1): pos位置对应的编码中奇数特征位上的值

参数:
    - **input**  (Variable) – Tensor或LoD level为1的LoDTensor。Tensor时，其形状为 :math:`[N, M, P]` ，其中 :math:`N` 表示batch size， :math:`M` 表示序列长度， :math:`P` 为特征维度大小；LoDTensor时，其形状为 :math:`[N, P]` ，其中 :math:`N` 表示所有序列长度之和， :math:`P` 为特征维度大小。数据类型为float32或float64。
    - **alpha**  (float) – 加权求和时输入input的权重系数
    - **beta**  (float) – 加权求和时位置编码的权重系数
    - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。


返回:  加上位置编码后的Tensor或LoDTensor，和输入（input）具有相同数据类型和形状及LoD信息。

返回类型: Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    tensor = fluid.data(
        name='tensor',
        shape=[None, 64, 512],
        dtype='float32')
    position_tensor = fluid.layers.add_position_encoding(
        input=tensor, alpha=1.0, beta=1.0)










