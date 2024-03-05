.. _cn_api_paddle_rot90:

rot90
-------------------------------

.. py:function:: paddle.rot90(x, k=1, axes=[0, 1], name=None):



沿 axes 指定的平面将 n 维 tensor 旋转 90 度。当 k 为正数，旋转方向为从 axes[0]到 axes[1]，当 k 为负数，旋转方向为从 axes[1]到 axes[0]，k 的绝对值表示旋转次数。

参数
::::::::::

    - **x** (Tensor) - 输入 Tensor。维度为多维，数据类型为 bool, int32, int64, float16, float32 或 float64。float16 只在 gpu 上支持。
    - **k** (int，可选) - 旋转方向和次数，默认值：1。
    - **axes** (list|tuple，可选) - axes 指定旋转的平面，维度必须为 2。默认值为[0, 1]。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::

    - 在指定平面 axes 上翻转指定次数后的 Tensor，与输入 x 数据类型相同。


代码示例
::::::::::

COPY-FROM: paddle.rot90
