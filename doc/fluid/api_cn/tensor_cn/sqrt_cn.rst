.. _cn_api_tensor_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.sqrt(x, name=None)

:alias_main: paddle.sqrt
:alias: paddle.sqrt,paddle.tensor.sqrt,paddle.tensor.math.sqrt
:update_api: paddle.fluid.layers.sqrt



计算输入的算数平方根。

.. math::
        out=\sqrt x=x^{1/2}

.. note::
    请确保输入中的数值是非负数。

参数:

    - **x** (Tensor) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：返回类型为Tensor， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

    import paddle
    paddle.disable_static()
    x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
    out = paddle.sqrt(x)
    print(out.numpy())
    # [0.31622777 0.4472136  0.54772256 0.63245553]
