.. _cn_api_paddle_isclose:

isclose
-------------------------------

.. py:function:: paddle.isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)

逐个检查 x 和 y 的所有元素是否均满足如下条件：

..  math::
    \left| x - y \right| \leq atol + rtol \times \left| y \right|

该 API 的行为类似于 :math:`numpy.isclose`，即逐个比较两个 Tensor 的所有元素是否在一定容忍误差范围内视为相等。

参数
:::::::::

    - **x** (Tensor) - 输入的 `Tensor`，数据类型为 float16、float32、float64、complex64 或 complex128。
    - **y** (Tensor) - 输入的 `Tensor`，数据类型为 float16、float32、float64、complex64 或 complex128。
    - **rtol** (float，可选) - 相对容忍误差，默认值为 1e-5。
    - **atol** (float，可选) - 绝对容忍误差，默认值为 1e-8。
    - **equal_nan** (bool，可选) - 如果设置为 True，则两个 NaN 数值将被视为相等，默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
计算得到的布尔类型 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.isclose
