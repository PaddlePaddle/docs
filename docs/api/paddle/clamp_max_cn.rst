.. _cn_api_paddle_clamp_max:

clamp_max
-------------------------------

.. py:function:: paddle.clamp_max(input, max=None, *, out=None)

该 API 将输入中的所有元素进行裁剪，使得输出元素不大于给定的最大值，具体公式如下：

.. math::

        Out = MIN(x, max)

参数
::::::::::::

    - **input** (Tensor) - 输入的 Tensor，数据类型为：float16、float32、float64、int32、int64。
    - **max** (float|int|Tensor，可选) - 裁剪的最大值，输入中大于该值的元素将由该元素代替。数据类型可以是 float32 或形状为 [] 的 0-D Tensor，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor。默认值为 None。

返回
::::::::::::
输出 Tensor，与 ``input`` 维度相同。当 ``input`` 数据类型为 int32 或 int64 且 ``max`` 为 float 类型时，输出 Tensor 的数据类型为 float32，否则与输入 ``input`` 的数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.clamp_max
