.. _cn_api_fluid_layers_softmax:

softmax
-------------------------------

.. py:function:: paddle.fluid.layers.softmax(input, use_cudnn=False, name=None, axis=-1)

该OP实现了softmax层。在OP的计算过程中，输入 ``input`` 的 ``axis`` 维会被置换到最后一维（操作1）；然后将输入 ``Tensor`` 在逻辑上变换为二维矩阵（操作2），二维矩阵矩阵第一维（列长度）是输入除最后一维之外的其他维长度的乘积，第二维（行长度）和输入 ``axis`` 维的长度相同；对于矩阵的每一行，softmax操作对其进行重新缩放，使得该行的每个元素在 \[0,1\] 范围内，并且总和为1；softmax操作执行完成后，执行操作2和操作1的逆运算，将二维矩阵恢复至和输入 ``input`` 相同的维度。

softmax操作计算K维向量（K是输入第 ``axis`` 维的长度）中指定维的指数值和全部维指数值的和。指定维的指数值与全部维指数值之和的比值就是softmax操作的输出。对于二维矩阵中的第i行和第j列有：

.. math::


    Out[i,j] = \frac{exp(X[i,j])}{\sum_j exp(X[i,j])}

参数：
    - **input** (Variable) - 任意维度的多维 ``Tensor`` ，数据类型为float32或float64。
    - **use_cudnn** (bool, 可选) - 指示是否用cudnn核，只有在cudnn库安装时有效。为了提高数值的稳定性，默认值：False。
    - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值：None。
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

