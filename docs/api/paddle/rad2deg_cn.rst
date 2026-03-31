.. _cn_api_paddle_rad2deg:

rad2deg
-------------------------------

.. py:function:: paddle.rad2deg(x, name=None, *, out=None)

将元素从弧度转换为度

.. math::

    rad2deg(x)=180/ \pi * x

参数
:::::::::

    - **x**  (Tensor) - 输入的 Tensor，数据类型为：int32、int64、float32、float64。别名 ``input``。
    - **name**  (str，可选) - 操作的名称（可选，默认值为 None）。更多信息请参见 :ref:`api_guide_Name`。

关键字参数
:::::::::

    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::

输出 Tensor，与 ``x`` 维度相同、数据类型相同（输入为 int 时，输出数据类型为 float32）。

代码示例
:::::::::

COPY-FROM: paddle.rad2deg
