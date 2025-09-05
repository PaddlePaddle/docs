.. _cn_api_paddle_clip:

clip
-------------------------------

.. py:function:: paddle.clip(x, min=None, max=None, name=None, *, out=None)




将输入的所有元素进行剪裁，使得输出元素限制在[min, max]内，具体公式如下：

.. math::

        Out = MIN(MAX(x, min), max)

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，如 ``clip(input=tensor_x, min=0.3, max=0.7)`` 等价于 ``clip(x=tensor_x, min=0.3, max=0.7)``。

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，数据类型为：bfloat16、float16、float32、float64、int32、int64。别名： ``input``。
    - **min** (float|int|Tensor，可选) - 裁剪的最小值，输入中小于该值的元素将由该元素代替，若参数为空，则不对输入的最小值做限制。数据类型可以是 float32 或形状为[]的 0-D Tensor，类型可以为 bfloat16、float16、float32、float64、int32，默认值为 None。
    - **max** (float|int|Tensor，可选) - 裁剪的最大值，输入中大于该值的元素将由该元素代替，若参数为空，则不对输入的最大值做限制。数据类型可以是 float32 或形状为[]的 0-D Tensor，类型可以为 bfloat16、float16、float32、float64、int32，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor。默认值为 None。

返回
::::::::::::
输出 Tensor，与 ``x`` 维度相同。当 ``x`` 数据类型为 int32 或 int64 并且 ``min`` 或 ``max`` 有一个为 float 类型时，输出 Tensor 的数据类型为 float32，否则与输入 ``x`` 的数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.clip
