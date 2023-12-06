.. _cn_api_paddle_moveaxis:

moveaxis
-------------------------------

.. py:function:: paddle.moveaxis(x, source, destination, name=None)

将输入 Tensor ``x`` 的轴从 ``source`` 位置移动到 ``destination`` 位置，其他轴按原来顺序排布。同时根据新的 shape，重排 Tensor 中的数据。

参数
:::::::::
    - **x** (Tensor) - 输入的 N-D Tensor，数据类型为：bool、int32、int64、float32、float64、complex64、complex128。
    - **source** (int|tuple|list) - 将被移动的轴的位置，其每个元素必须为不同的整数。
    - **destination** (int|tuple|list) - 轴被移动后的目标位置，其每个元素必须为不同的整数。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``：将轴移动后的 Tensor

代码示例
:::::::::

COPY-FROM: paddle.moveaxis
