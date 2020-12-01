.. _cn_api_fluid_layers_has_inf:

has_inf
-------------------------------

.. py:function:: paddle.has_inf(x)




检查输入的变量(x)中是否包含无穷数(inf)。

参数：
    - **x** (Tensor) - 被检查的变量Tensor。

返回：Tensor，存储输出值，包含一个bool型数值，指明输入中是否包含无穷数(inf)。

**代码示例**：

.. code-block:: python

    import paddle
    data = paddle.randn(shape=[4, 32, 32], dtype="float32")
    res = paddle.has_inf(data)
    # [False]
