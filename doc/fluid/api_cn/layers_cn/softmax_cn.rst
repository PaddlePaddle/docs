.. _cn_api_fluid_layers_softmax:

softmax
-------------------------------

.. py:function:: paddle.fluid.layers.softmax(input, use_cudnn=False, name=None, axis=-1)

该OP实现了softmax层。输入 ``input`` 的 ``axis`` 维会被置换到最后一维。然后，将输入 ``Tensor`` 在逻辑上展平为二维矩阵。矩阵第二维（行长度）和输入 ``axis`` 维的长度相同，第一维（列长度）是输入除最后一维之外的其他所有维长度的乘积。对于矩阵的每一行，softmax操作将包含任意实数值的K维向量（K是输入 ``axis`` 维的长度）压缩为包含\[0,1\]范围内任意实数的K维向量，并且K维实数的和为1。

softmax操作计算K维向量中指定维的指数和其他维指数值的总和。然后给定维的指数与其他维指数值之和的比率就是softmax操作的输出。

对矩阵中的每行i和每列j有：

.. math::


    Out[i,j] = \frac{exp(X[i,j])}{\sum_j exp(X[i,j])}

参数：
    - **input** (Variable) - 任意维度的多维 ``Tensor`` ，数据类型为float32或float64。
    - **use_cudnn** (bool, 可选) - 指示是否用cudnn核，只有在cudnn库安装时有效。为了提高数值的稳定性，默认值：False。
    - **name** (str, 可选) - 层名称。若为空，则自动为该层命名。默认值：None。
    - **axis** (int, 可选) - 指示进行softmax计算的维度索引，其范围应为 :math:`[-1，rank-1]` ，其中rank是输入变量的秩。默认值：-1。

返回：表示softmax操作结果的 ``Tensor`` ，数据类型和 ``input`` 一致，返回维度和 ``input`` 一致。

返回类型：Variable

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

