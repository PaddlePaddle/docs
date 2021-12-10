.. _cn_api_paddle_tensor_deg2rad:

deg2rad
-------------------------------

.. py:function:: paddle.deg2rad(x, name=None)

将元素从弧度的角度转换为度

.. math::

    deg2rad(x)=\pi * x / 180

参数
:::::::::

- **x**  (Tensor) - 输入的Tensor，数据类型为：int32、int64、float32、float64。
- **name**  (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出Tensor，与 ``x`` 维度相同、数据类型相同（输入为int时，输出数据类型为float32）。

代码示例
:::::::::

COPY-FROM: paddle.deg2rad
