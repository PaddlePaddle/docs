.. _cn_api_paddle_deg2rad:

deg2rad
-------------------------------

.. py:function:: paddle.deg2rad(x, name=None)

将元素从度转换为弧度

.. math::

    deg2rad(x)=\pi * x / 180

参数
:::::::::

- **x**  (Tensor) - 输入的 Tensor，数据类型为：int32、int64、float32、float64。
- **name**  (str，可选) - 操作的名称（可选，默认值为 None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出 Tensor，与 ``x`` 维度相同、数据类型相同（输入为 int 时，输出数据类型为 float32）。

代码示例
:::::::::

COPY-FROM: paddle.deg2rad
