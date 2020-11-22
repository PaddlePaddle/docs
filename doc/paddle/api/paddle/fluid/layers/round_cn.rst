.. _cn_api_fluid_layers_round:

round
-------------------------------

.. py:function:: paddle.round(x, name=None)





该OP将输入中的数值四舍五入到最接近的整数数值。

.. code-block:: python

  输入：
    x.shape = [4]
    x.data = [1.2, -0.9, 3.4, 0.9]

  输出：
    out.shape = [4]
    out.data = [1., -1., 3., 1.]

参数:

    - **x** (Tensor) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：返回类型为Tensor， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.to_tensor([1.2, -0.9, 3.4, 0.9], dtype='float32')
    result = paddle.round(x)
    print(result) # result=[1., -1., 3., 1.]



