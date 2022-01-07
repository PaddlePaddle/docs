.. _cn_api_nn_cn_prelu:

prelu
-------------------------------

.. py:function:: paddle.nn.functional.prelu(x, weight, data_format="NCHW", name=None)

prelu激活层（PRelu Activation Operator）。计算公式如下：

.. math::

    prelu(x) = max(0, x) + weight * min(0, x)

其中，:math:`x` 和 `weight` 为输入的 Tensor

参数
::::::::::
    - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
    - weight (Tensor) - 可训练参数，数据类型同``x`` 一致，形状支持2种：[1] 或者 [in]，其中`in`为输入的通道数。
    - data_format (str，可选) – 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" 或者 "NDHWC"。默认值："NCHW"。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F
    import numpy as np

    data = np.array([[[[-2.0,  3.0, -4.0,  5.0],
                       [ 3.0, -4.0,  5.0, -6.0],
                       [-7.0, -8.0,  8.0,  9.0]],
                      [[ 1.0, -2.0, -3.0,  4.0],
                       [-5.0,  6.0,  7.0, -8.0],
                       [ 6.0,  7.0,  8.0,  9.0]]]], 'float32')
    x = paddle.to_tensor(data)
    w = paddle.to_tensor(np.array([0.25]).astype('float32'))
    out = F.prelu(x, w)
    # [[[[-0.5 ,  3.  , -1.  ,  5.  ],
    #    [ 3.  , -1.  ,  5.  , -1.5 ],
    #    [-1.75, -2.  ,  8.  ,  9.  ]],
    #   [[ 1.  , -0.5 , -0.75,  4.  ],
    #    [-1.25,  6.  ,  7.  , -2.  ],
    #    [ 6.  ,  7.  ,  8.  ,  9.  ]]]]
