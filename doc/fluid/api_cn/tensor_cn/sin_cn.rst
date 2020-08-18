.. _cn_api_tensor_sin:

sin
-------------------------------

.. py:function:: paddle.sin(x, name=None)



计算输入的正弦值。

.. math::
        out = sin(x)

参数：
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为： float16, float32, float64。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：
    - Tensor，对输入x计算sin值后的Tensor，形状、数据类型同输入x一致。


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x_data = np.array([-0.4, -0.2, 0.1, 0.3])
    x = paddle.to_tensor(x_data)
    out = paddle.sin(x)
    print(out.numpy())
    # [-0.38941834 -0.19866933  0.09983342  0.29552021]
