.. _cn_api_paddle_i0:

i0
-------------------------------

.. py:function:: paddle.hypot(x, y, name=None)


`hypot` 函数对于给定直角三角形直角边 `x`, `y` 实现斜边长度求解的计算;

.. math::
    out= \sqrt{x^2 + y^2} $$

参数
::::::::::
    - **x** (Tensor) – 输入Tensor，它的数据类型可以是 float32，float64, int32, int6。
    - **x** (Tensor) – 输入 Tensor，它的数据类型可以是 float32，float64,int32, int64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::
    - ``out`` (Tensor)：一个 n-d Tensor。如果 x、y 具有不同的形状并且是可广播的，则得到的张量形状是广播后 x 和 y 的形状。如果 x、y 具有相同的形状，则其形状与 x 和 y 相同。


代码示例
::::::::::

COPY-FROM: paddle.hypot
