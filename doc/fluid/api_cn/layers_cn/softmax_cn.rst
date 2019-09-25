.. _cn_api_fluid_layers_softmax:

softmax
-------------------------------

.. py:function:: paddle.fluid.layers.softmax(input, use_cudnn=False, name=None, axis=-1)

softmax输出张量和输入张量的维度相同。

输入变量的 ``axis`` 维会被排列到最后一维。在OP底层计算上将输入张量resize成二维矩阵。矩阵的第二维（行数）和输入张量的 ``axis`` 维相同。第一维（列数）
是输入张量除最后一维之外的所有维长度乘积。对矩阵的每一行来说,softmax函数将含有任意实数值的K维向量(K是矩阵的宽度,也就是输入张量 ``axis`` 维度的大小),resize成K维含有取值为[0,1]的向量，并且这些值和为1。

softmax函数计算k维向量输入中所有其他维的指数指数值的累加和。该维的指数比例和所有其他维的指数值之和作为softmax函数的输出。

对矩阵中的每行i和每列j有：

.. math::


    Out[i,j] = \frac{exp(X[i,j])}{\sum_j exp(X[i,j])}

参数：
    - **input** (Tensor|LoDTensor)- 数据类型为float32，float64。激活函数softmax的输入。
    - **use_cudnn** (bool) - 是否使用cudnn，只有cudnn库安装时该参数才有效。为了底层数学计算的稳定性，默认该项为False。
    - **name** (str|None) - 该层名称（可选）。若为空，则自动为该层命名。默认：None
    - **axis** (int) - 指定softmax计算的轴，应该在 :math:`[-1，rank-1]` 范围内，其中rank是输入变量的秩。 默认值：-1。-1为最后一维。

返回： softmax函数的输出。

返回类型：Variable（Tensor），数据类型为float32或float64的Tensor。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    data = fluid.layers.data(name="input", shape=[-1, 3],dtype="float32")
    result = fluid.layers.softmax(data,axis=1)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.rand(3, 3).astype("float32")
    output= exe.run(feed={"input": x},
                     fetch_list=[result[0]])
    print(output)
    """
    output:
    array([0.22595254, 0.39276356, 0.38128382], dtype=float32)]
    """









