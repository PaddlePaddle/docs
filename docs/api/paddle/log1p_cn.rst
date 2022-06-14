.. _cn_api_paddle_tensor_log1p:

log1p
-------------------------------

.. py:function:: paddle.log1p(x, name=None)


计算 Log1p（自然对数 + 1）结果。

.. math::
                  \\Out=ln(x+1)\\


参数
::::::::::::

  - **x** (Tensor) – 输入为一个多维的 Tensor，数据类型为 float32，float64。
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回
::::::::::::
计算 ``x`` 的自然对数 + 1后的 Tensor，数据类型，形状与 ``x`` 一致。

代码示例
::::::::::::

..  code-block:: python

    import paddle
    
    data = paddle.to_tensor([[0], [1]], dtype='float32')
    res = paddle.log1p(data)
    # [[0.], [0.6931472]] 
    
