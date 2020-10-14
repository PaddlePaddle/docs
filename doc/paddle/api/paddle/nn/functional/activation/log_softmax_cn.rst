.. _cn_api_nn_cn_log_softmax:

log_softmax
-------------------------------
.. py:function:: paddle.nn.functional.log_softmax(x, axis=-1, dtype=None, name=None)

该OP实现了log_softmax层。OP的计算公式如下：

.. math::

    Out[i, j] = log(softmax(x)) = log(\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

参数
::::::::::
    - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
    - axis (int, 可选) - 指定对输入 ``x`` 进行运算的轴。``axis`` 的有效范围是[-D, D)，D是输入 ``x`` 的维度， ``axis`` 为负值时与 :math:`axis + D` 等价。默认值为-1。
    - dtype (str|np.dtype|core.VarDesc.VarType, 可选) - 输入Tensor的数据类型。如果指定了 ``dtype`` ，则输入Tensor的数据类型会在计算前转换到 ``dtype`` 。``dtype``可以用来避免数据溢出。如果 ``dtype`` 为None，则输出Tensor的数据类型和 ``x`` 相同。默认值为None。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，形状和 ``x`` 相同，数据类型为 ``dtype`` 或者和 ``x`` 相同。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F
    import numpy as np

    paddle.disable_static()

    x = np.array([[[-2.0, 3.0, -4.0, 5.0],
                    [3.0, -4.0, 5.0, -6.0],
                    [-7.0, -8.0, 8.0, 9.0]],
                    [[1.0, -2.0, -3.0, 4.0],
                    [-5.0, 6.0, 7.0, -8.0],
                    [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
    x = paddle.to_tensor(x)
    out1 = F.log_softmax(x)
    out2 = F.log_softmax(x, dtype='float64')
    # out1's data type is float32; out2's data type is float64
    # out1 and out2's value is as follows:
    # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
    #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
    #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
    #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
    #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
    #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
