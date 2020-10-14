.. _cn_api_fluid_layers_has_nan:

has_nan
-------------------------------

.. py:function:: paddle.has_nan(x)




检查输入的变量(x)中是否包含NAN。

参数：
  - **x** (Tensor) - 被检查的变量Tensor。

返回：Tensor，存储输出值，包含一个bool型数值，指明输入中是否包含NAN。

**代码示例**：

.. code-block:: python

    import paddle
    data = paddle.randn(shape=[2,3], dtype="float32")
    res = paddle.has_nan(data)
    # [False]




