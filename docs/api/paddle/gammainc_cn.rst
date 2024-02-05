.. _cn_api_paddle_gammainc:

gammainc
-------------------------------

.. py:function:: paddle.gammainc(x, y, name=None)
计算正则化下不完全伽马函数。

.. math::
    P(x, y) = \frac{1}{\Gamma(x)} \int_{0}^{y} t^{x-1} e^{-t} dt

参数
::::::::::::

    - **x** - 非负参数 Tensor。数据类型必须为 float32， float64。
    - **y** - 正参数 Tensor。数据类型必须为 float32， float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出为 Tensor，计算的正则化下不完全伽马函数值。

代码示例
::::::::::::

COPY-FROM: paddle.gammainc
