.. _cn_api_paddle_polygamma:

polygamma
-------------------------------

.. py:function:: paddle.polygamma(x, n, name=None)


对于给定的 ``x`` 逐元素计算每个元素的多伽马函数值，其中 ``x`` 的大小无特殊限制。返回一个多伽马函数上的 Tensor。

.. math::
    \Phi^n(x) = \frac{d^n}{dx^n} [\ln(\Gamma(x))]

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。
    - **n** (int) - 指定需要求解 ``n`` 阶多伽马函数，它的数据类型是 int。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::
    - ``Tensor`` (Tensor)：在 x 处的多伽马函数的值。


代码示例
::::::::::

COPY-FROM: paddle.polygamma
