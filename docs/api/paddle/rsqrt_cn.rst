.. _cn_api_fluid_layers_rsqrt:

rsqrt
-------------------------------

.. py:function:: paddle.rsqrt(x, name=None)




该OP为rsqrt激活函数。

注：输入x应确保为非 **0** 值，否则程序会抛异常退出。

其运算公式如下：

.. math::
    out = \frac{1}{\sqrt{x}}


参数:
    - **x** (Tensor) – 输入是多维Tensor，数据类型可以是float32和float64。 
    - **name** (str，可选）— 这一层的名称（可选）。如果设置为None，则将自动命名这一层。默认值为None。

返回：Tensor，对输入x进行rsqrt激活函数计算结果，数据shape、类型和输入x的shape、类型一致。

**代码示例**：

.. code-block:: python

        import paddle

        x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
        out = paddle.rsqrt(x)
        # [3.16227766 2.23606798 1.82574186 1.58113883]
