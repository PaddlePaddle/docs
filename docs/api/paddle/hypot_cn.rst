.. _cn_api_paddle_i0:

i0
-------------------------------

.. py:function:: paddle.hypot(x, y, name=None)


`hypot` 函数对于给定直角三角形直角边 `x`, `y` 实现斜边长度求解的计算;

.. math::
    out= \sqrt{x^2 + y^2} $$

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::
    - ``Tensor`` (Tensor)：在 x 处的第一类零阶修正贝塞尔曲线函数的值。


代码示例
::::::::::

COPY-FROM: paddle.i0
