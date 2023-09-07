.. _cn_api_paddle_cos:

cos
-------------------------------

.. py:function:: paddle.cos(x, name=None)




余弦函数。

输入范围是 `(-inf, inf)`，输出范围是 `[-1,1]`。

.. math::

    out = cos(x)

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64 、float16。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.cos
