.. _cn_api_fluid_layers_softmax:

softmax
-------------------------------

.. py:function:: paddle.fluid.layers.softmax(input, use_cudnn=False, name=None, axis=-1)

该OP实现了softmax层。OP的计算过程如下：

步骤1：输入 ``input`` 的 ``axis`` 维会被置换到最后一维；

步骤2：将输入 ``Tensor`` 在逻辑上变换为二维矩阵。二维矩阵第一维（列长度）是输入除最后一维之外的其他维度值的乘积，第二维（行长度）和输入 ``axis`` 维的长度相同；对于矩阵的每一行，softmax操作对其进行重新缩放，使得该行的每个元素在 \[0,1\] 范围内，并且总和为1；

步骤3：softmax操作执行完成后，执行步骤1和步骤2的逆运算，将二维矩阵恢复至和输入 ``input`` 相同的维度。

上述步骤2中softmax操作计算过程如下：

    - 对于二维矩阵的每一行，计算K维向量（K是输入第 ``axis`` 维的长度）中指定位置的指数值和全部位置指数值的和。

    - 指定位置指数值与全部位置指数值之和的比值就是softmax操作的输出。

对于二维矩阵中的第i行和第j列有：

.. math::


    Out[i,j] = \frac{exp(X[i,j])}{\sum_j exp(X[i,j])}

- 示例1（矩阵一共有三维。axis = -1，表示沿着最后一维（即第三维）做softmax操作）

.. code-block:: python

  输入

    X.shape = [2, 3, 4] 

    X.data = [[[2.0, 3.0, 4.0, 5.0],
               [3.0, 4.0, 5.0, 6.0],
               [7.0, 8.0, 8.0, 9.0]],
              [[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [6.0, 7.0, 8.0, 9.0]]]

    axis = -1

  输出

    Out.shape = [2, 3, 4]

    Out.data = [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                 [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                 [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                 [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                 [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]

- 示例2（矩阵一共有三维。axis = 1，表示沿着第二维做softmax操作）

.. code-block:: python

  输入

    X.shape = [2, 3, 4] 

    X.data = [[[2.0, 3.0, 4.0, 5.0],
               [3.0, 4.0, 5.0, 6.0],
               [7.0, 8.0, 8.0, 9.0]],
              [[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [6.0, 7.0, 8.0, 9.0]]]

    axis = 1

  输出

    Out.shape = [2, 3, 4]

    Out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                 [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                 [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                 [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                 [0.72747516, 0.72747516, 0.72747516, 0.72747516]]] 


参数
::::::::::::

    - **input** (Variable) - 任意维度的多维 ``Tensor``，数据类型为float32或float64。
    - **use_cudnn** (bool，可选) - 指示是否用cudnn库。当 ``use_cudnn`` 为True时，在安装GPU版本Paddle并且本机安装cudnn库的前提下，使用GPU训练或推理时才有效。默认值：False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **axis** (int，可选) - 指示进行softmax计算的维度索引，其范围应为 :math:`[-1，rank-1]`，其中rank是输入变量的秩。默认值：-1（表示对最后一维做softmax操作）。

返回
::::::::::::
表示softmax操作结果的 ``Tensor``，数据类型和 ``input`` 一致，返回维度和 ``input`` 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

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

