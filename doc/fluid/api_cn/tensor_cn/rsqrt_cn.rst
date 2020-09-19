.. _cn_api_tensor_cn_rsqrt:

rsqrt
-------------------------------

.. py:function:: paddle.rsqrt(x, name=None)




该OP为rsqrt激活函数。

注：输入x应确保为非 **0** 值，否则程序会抛异常退出。

其运算公式如下：

.. math::
    out = \frac{1}{\sqrt{x}}


参数:
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：
    - Tensor，对输入x进行rsqrt激活后的Tensor，形状、数据类型与输入x一致。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x_data = np.array([0.1, 0.2, 0.3, 0.4])
    x = paddle.to_tensor(x_data)
    out = paddle.rsqrt(x)
    print(out.numpy())
    # [3.16227766 2.23606798 1.82574186 1.58113883]

