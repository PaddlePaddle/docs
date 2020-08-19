.. _cn_api_nn_PRelu:

PRelu
-------------------------------
.. py:class:: paddle.nn.PRelu(num_parameters=1, init=0.25, weight_attr=None, name=None)

PRelu激活层（PRelu Activation Operator）。计算公式如下：

如果使用近似计算：

.. math::

    PReLU(x) = max(0, x) + \weight * min(0, x)

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - num_parameters (int, 可选) - 可训练`weight`数量，支持2种输入：1 - 输入中的所有元素使用同一个`weight`值; 输入的通道数 - 在同一个通道中的元素使用同一个`weight`值。默认为1。
    - init (float, 可选) - `weight`的初始值。默认为0.25。
    - weight_attr (ParamAttr, 可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    data = np.array([[[[-2.0,  3.0, -4.0,  5.0],
                       [ 3.0, -4.0,  5.0, -6.0],
                       [-7.0, -8.0,  8.0,  9.0]],
                      [[ 1.0, -2.0, -3.0,  4.0],
                       [-5.0,  6.0,  7.0, -8.0],
                       [ 6.0,  7.0,  8.0,  9.0]]]], 'float32')
    x = paddle.to_tensor(data)
    m = paddle.nn.PReLU(1, 0.25)
    out = m(x)
    # [[[[-0.5 ,  3.  , -1.  ,  5.  ],
    #    [ 3.  , -1.  ,  5.  , -1.5 ],
    #    [-1.75, -2.  ,  8.  ,  9.  ]],
    #   [[ 1.  , -0.5 , -0.75,  4.  ],
    #    [-1.25,  6.  ,  7.  , -2.  ],
    #    [ 6.  ,  7.  ,  8.  ,  9.  ]]]]