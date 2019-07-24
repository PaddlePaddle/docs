.. _cn_api_fluid_layers_softmax:

softmax
-------------------------------

.. py:function:: paddle.fluid.layers.softmax(input, use_cudnn=False, name=None, axis=-1)

softmax操作符的输入是任意阶的张量，输出张量和输入张量的维度相同。

输入变量的 ``axis`` 维会被排列到最后一维。然后逻辑上将输入张量压平至二维矩阵。矩阵的第二维（行数）和输入张量的 ``axis`` 维相同。第一维（列数）
是输入张量除最后一维之外的所有维长度乘积。对矩阵的每一行来说,softmax操作将含有任意实数值的K维向量(K是矩阵的宽度,也就是输入张量 ``axis`` 维度的大小)压缩成K维含有取值为[0,1]中实数的向量，并且这些值和为1。


softmax操作符计算k维向量输入中所有其他维的指数和指数值的累加和。维的指数比例和所有其他维的指数值之和作为softmax操作符的输出。

对矩阵中的每行i和每列j有：

.. math::

    Out[i,j] = \frac{exp(X[i,j])}{\sum_j exp(X[i,j])}

参数：
    - **input** (Variable) - 输入变量
    - **use_cudnn** (bool) - 是否用cudnn核，只有在cudnn库安装时有效。为了数学稳定性，默认该项为False。
    - **name** (str|None) - 该层名称（可选）。若为空，则自动为该层命名。默认：None
    - **axis** (Variable) - 执行softmax计算的维度索引，应该在 :math:`[-1，rank-1]` 范围内，其中rank是输入变量的秩。 默认值：-1。

返回： softmax输出

返回类型：变量（Variable）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[2], dtype='float32')
    fc = fluid.layers.fc(input=x, size=10)
    # 在第二维执行softmax
    softmax = fluid.layers.softmax(input=fc, axis=1)
    # 在最后一维执行softmax
    softmax = fluid.layers.softmax(input=fc, axis=-1)









