.. _cn_api_nn_LogSoftmax:

LogSoftmax
-------------------------------
.. py:class:: paddle.nn.LogSoftmax(axis=None)

:alias_main: paddle.nn.LogSoftmax
:alias: paddle.nn.LogSoftmax,paddle.nn.layer.LogSoftmax,paddle.nn.layer.activation.LogSoftmax




**LogSoftmax激活层：**

.. math::

        \\output = \frac{1}{1 + e^{-input}}\\

参数:
    - **axis** (int, 可选) - 指示进行LogSoftmax计算的维度索引，其范围应为 :math:`[-1，rank-1]` ，其中rank是输入变量的秩。默认值：None（与-1效果相同，表示对最后一维做LogSoftmax操作）。

返回：无

**代码示例**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.nn as nn
    import numpy as np
    
    data = np.array([[[-2.0, 3.0, -4.0, 5.0], [3.0, -4.0, 5.0, -6.0], [-7.0, -
        8.0, 8.0, 9.0]], [[1.0, -2.0, -3.0, 4.0], [-5.0, 6.0, 7.0, -8.0], [6.0,
        7.0, 8.0, 9.0]]]).astype('float32')
    my_log_softnmax = nn.LogSoftmax()
    with paddle.imperative.guard():
        data = paddle.imperative.to_variable(data)
        res = my_log_softnmax(data)
        # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
        #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
        #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
        #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
        #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
        #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]

