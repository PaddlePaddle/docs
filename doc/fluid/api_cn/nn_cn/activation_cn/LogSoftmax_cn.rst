.. _cn_api_nn_LogSoftmax:

LogSoftmax
-------------------------------
.. py:class:: paddle.nn.LogSoftmax(axis=-1, name=None)

LogSoftmax激活层，计算公式如下：

.. math::

    Out[i, j] = log(softmax(x)) 
              = log(\\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

参数
::::::::::
    - axis (int, 可选) - 指定对输入Tensor进行运算的轴。``axis`` 的有效范围是[-D, D)，D是输入Tensor的维度， ``axis`` 为负值时与 :math:`axis + D` 等价。默认值为-1。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    x = np.array([[[-2.0, 3.0, -4.0, 5.0],
                    [3.0, -4.0, 5.0, -6.0],
                    [-7.0, -8.0, 8.0, 9.0]],
                    [[1.0, -2.0, -3.0, 4.0],
                    [-5.0, 6.0, 7.0, -8.0],
                    [6.0, 7.0, 8.0, 9.0]]], 'float32')
    m = paddle.nn.LogSoftmax()
    x = paddle.to_tensor(x)
    out = m(x)
    # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
    #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
    #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
    #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
    #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
    #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
