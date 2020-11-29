.. _cn_api_paddle_tensor_log1p:

log1p
-------------------------------

.. py:function:: paddle.log1p(x, name=None)

:alias_main: paddle.log1p
:alias: paddle.log1p,paddle.tensor.log1p,paddle.tensor.math.log1p




该OP计算Log1p（加一的自然对数）结果。

.. math::
                  \\Out=ln(x+1)\\


参数:
  - **x** (Tensor) – 指定输入为一个多维的Tensor。数据类型为float32，float64。 
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Log1p算子自然对数输出

返回类型: Tensor - 该OP的输出为一个多维的Tensor，数据类型为输入一致。


**代码示例**

..  code-block:: python

    import paddle
    import numpy as np
    
    paddle.enable_imperative()
    x = np.array([[1, 2], [3, 4]]).astype('float32')
    x1 = paddle.imperative.to_variable(x)
    
    out1 = paddle.log1p(x1)
    print(out1.numpy())
    # [[0.6931472 1.0986123]
    #  [1.3862944 1.609438 ]]
    
