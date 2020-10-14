.. _cn_api_fluid_layers_log:

log
-------------------------------

.. py:function:: paddle.log(x, name=None)





Log激活函数（计算自然对数）

.. math::
                  \\Out=ln(x)\\


参数:
  - **x** (Tensor) – 该OP的输入为Tensor。数据类型为float32，float64。 
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Tensor, Log算子自然对数输出，数据类型与输入一致。

**代码示例**

.. code-block:: python

    import paddle

    x = [[2,3,4], [7,8,9]]
    x = paddle.to_tensor(x, dtype='float32')
    res = paddle.log(x)
    # [[0.693147, 1.09861, 1.38629], [1.94591, 2.07944, 2.19722]]


