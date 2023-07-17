.. _cn_api_paddle_tensor_math_logaddexp:

logaddexp
-------------------------------

.. py:function:: paddle.logaddexp(x, y, name=None)

计算 ``x`` 和 ``y`` 的以 e 为底的指数的和的自然对数。计算公式如下：

.. math::
   logaddexp(x) = \log\add (exp(x)+exp(y))

参数
::::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：int32，int64，float16，float32、float64。
    - **y** (Tensor) - 输入的 Tensor，数据类型为：int32，int64，float16，float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，根据上述公式计算的 logaddexp(x) 结果

代码示例
::::::::::

COPY-FROM: paddle.logaddexp
