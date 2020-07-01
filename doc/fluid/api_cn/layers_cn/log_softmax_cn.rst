.. _cn_api_fluid_layers_log_softmax:

log_softmax
-------------------------------
.. py:function:: paddle.fluid.layers.log_softmax(input, axis=None, dtype=None, name=None)


**log_softmax激活层：**

.. math::

        \\output = \frac{1}{1 + e^{-input}}\\

参数
::::::::::::

    - **input** (Variable) - 任意维度的多维 ``Tensor`` ，数据类型为float32或float64。
    - **axis** (int, 可选) - 指示进行LogSoftmax计算的维度索引，其范围应为 :math:`[-1，rank-1]` ，其中rank是输入变量的秩。默认值：None（与-1效果相同，表示对最后一维做LogSoftmax操作）。
    - **dtype** (np.dtype|core.VarDesc.VarType|str) - 期望输出``Tensor``的数据类型。如果指定了``dtype``，输入tensor的数据类型将在计算前被转换为``dtype``类型，可以有效防止数据溢出。默认值：None。支持的类型：float32或float64。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
::::::::::::
表示log_softmax操作结果的 ``Tensor`` ，数据类型和 ``dtype`` 或者 ``input`` 一致，返回维度和 ``input`` 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    data = np.array([[[-2.0, 3.0, -4.0, 5.0],
                    [3.0, -4.0, 5.0, -6.0],
                    [-7.0, -8.0, 8.0, 9.0]],
                    [[1.0, -2.0, -3.0, 4.0],
                    [-5.0, 6.0, 7.0, -8.0],
                    [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        res = fluid.layers.log_softmax(data, -1)
        # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
        #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
        #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
        #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
        #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
        #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
