.. _cn_api_tensor_cn_round:

round
-------------------------------

.. py:function:: paddle.round(x, name=None)



该OP将输入中的数值四舍五入到最接近的整数数值。

参数:
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为： float16, float32, float64。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：
    - Tensor，对输入x四舍五入后的Tensor，形状、数据类型与输入x一致。


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x_data = np.array([-0.5, -0.2, 0.6, 1.5])
    x = paddle.to_tensor(x_data)
    out = paddle.round(x)
    print(out.numpy())
    # [-1. -0.  1.  2.]
