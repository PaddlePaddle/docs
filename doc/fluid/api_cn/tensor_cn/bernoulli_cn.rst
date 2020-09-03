.. _cn_api_tensor_bernoulli:

bernoulli
-------------------------------

.. py:function:: paddle.bernoulli(x, name=None)




该OP以输入 ``x`` 为概率，生成一个伯努利分布（0-1分布）的Tensor，输出Tensor的形状和数据类型与输入 ``x`` 相同。

.. math::
   out_i \sim Bernoulli(p = x_i)

参数：
    - **x** (Tensor) - 输入的概率值。数据类型为 ``float32`` 、``float64`` .
    - **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：
    Tensor：伯努利分布的随机Tensor，形状和数据类型为与输入 ``x`` 相同。


**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    x = paddle.rand([2, 3])
    print(x.numpy())
    # [[0.11272584 0.3890902  0.7730957 ]
    # [0.10351662 0.8510418  0.63806665]]

    out = paddle.bernoulli(x)
    print(out.numpy())
    # [[0. 0. 1.]
    # [0. 0. 1.]]








