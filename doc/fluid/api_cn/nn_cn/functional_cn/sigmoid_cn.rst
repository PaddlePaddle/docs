.. _cn_api_nn_functional_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.nn.functional.sigmoid(x, name=None)



sigmoid激活函数

.. math::
    out = \frac{1}{1 + e^{-x}}


参数：
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：
    - Tensor，对输入x进行sigmoid激活后的Tensor，形状、数据类型与输入x一致。


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn.functional as F
    paddle.disable_static()
    x_data = np.array([-0.4, -0.2, 0.1, 0.3])
    x = paddle.to_tensor(x_data)
    out = F.sigmoid(x)
    print(out.numpy())
    # [0.40131234 0.450166   0.52497919 0.57444252]
