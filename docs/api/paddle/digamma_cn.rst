.. _cn_api_tensor_digamma:

digamma
----------------

.. py:function:: paddle.digamma(x, name=None)


逐元素计算输入Tensor的digamma函数值

.. math::
    \\Out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }\\


参数
:::::::::
  - **x** (Tensor) – 该OP的输入为Tensor。数据类型为float32，float64。 
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回
:::::::::
``Tensor``, digamma函数计算结果，数据类型和维度大小与输入一致。

代码示例
:::::::::

.. code-block:: python

    import paddle

    data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
    res = paddle.digamma(data)
    print(res)
    # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #       [[-0.57721591,  0.03648996],
    #        [ nan       ,  5.32286835]])

