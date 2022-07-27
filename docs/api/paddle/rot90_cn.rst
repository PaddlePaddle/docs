.. _cn_api_tensor_rot90:

rot90
-------------------------------

.. py:function:: paddle.rot90(x, k=1, axes=[0, 1], name=None):



沿axes指定的平面将n维tensor旋转90度。当k为正数，旋转方向为从axes[0]到axes[1]，当k为负数，旋转方向为从axes[1]到axes[0]，k的绝对值表示旋转次数。

参数
::::::::::

    - **x** (Tensor) - 输入张量。维度为多维，数据类型为bool, int32, int64, float16, float32或float64。float16只在gpu上支持。
    - **k** (int，可选) - 旋转方向和次数，默认值：1。
    - **axes** (list|tuple，可选) - axes指定旋转的平面，维度必须为2。默认值为[0, 1]。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::

    - 在指定平面axes上翻转指定次数后的张量，与输入x数据类型相同。


代码示例
::::::::::

COPY-FROM: paddle.rot90
