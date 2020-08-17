.. _cn_api_tensor_cn_rsqrt:

rsqrt
-------------------------------

.. py:function:: paddle.rsqrt(x, name=None)

:alias_main: paddle.rsqrt
:alias: paddle.rsqrt,paddle.tensor.rsqrt,paddle.tensor.math.rsqrt
:old_api: paddle.fluid.layers.rsqrt



该OP为rsqrt激活函数。

注：输入x应确保为非 **0** 值，否则程序会抛异常退出。

其运算公式如下：

.. math::
    out = \frac{1}{\sqrt{x}}


参数:
    - **x** (Tensor) - 支持任意维度的Tensor。数据类型为float32，float64。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。


返回：对输入x进行rsqrt激活函数计算后的Tensor，数据shape和输入x的shape一致。

返回类型：Tensor，数据类型和输入数据类型一致。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x_data = np.array([0.1, 0.2, 0.3, 0.4])
    x = paddle.to_variable(x_data)
    out = paddle.rsqrt(x)
    print(out.numpy())
    # [3.16227766 2.23606798 1.82574186 1.58113883]

