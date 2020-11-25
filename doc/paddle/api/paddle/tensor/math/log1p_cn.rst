.. _cn_api_paddle_tensor_log1p:

log1p
-------------------------------

.. py:function:: paddle.log1p(x, name=None)





该OP计算Log1p（加一的自然对数）结果。

.. math::
                  \\Out=ln(x+1)\\


参数:
  - **x** (Tensor) – 指定输入为一个多维的Tensor。数据类型为float32，float64。 
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Tensor, Log1p算子自然对数输出，数据类型，形状为输入一致

**代码示例**

..  code-block:: python

    import paddle
    
    data = paddle.to_tensor([[0], [1]], dtype='float32')
    res = paddle.log1p(data)
    # [[0.], [0.6931472]] 
    
