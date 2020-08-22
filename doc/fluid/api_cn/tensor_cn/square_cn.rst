.. _cn_api_tensor_cn_square:

square
-------------------------------

.. py:function:: paddle.square(x,name=None)




该OP执行逐元素取平方运算。

.. math::
    out = x^2

参数:
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64, float16, int32, int64。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：
    - Tensor，对输入x取平方后的Tensor，形状、数据类型与输入x一致。


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x_data = np.array([-0.4, -0.2, 0.1, 0.3])
    x = paddle.to_tensor(x_data)
    out = paddle.square(x)
    print(out.numpy())
    # [0.16 0.04 0.01 0.09]
