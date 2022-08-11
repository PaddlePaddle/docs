.. _cn_api_tensor_clip:

clip
-------------------------------

.. py:function:: paddle.clip(x, min=None, max=None, name=None)




将输入的所有元素进行剪裁，使得输出元素限制在[min, max]内，具体公式如下：

.. math::

        Out = MIN(MAX(x, min), max)

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64、int32、int64。
    - **min** (float32|Tensor，可选) - 裁剪的最小值，输入中小于该值的元素将由该元素代替，若参数为空，则不对输入的最小值做限制。数据类型可以是 float32 或形状为[1]的 Tensor，类型可以为 int32、float32、float64，默认值为 None。
    - **max** (float32|Tensor，可选) - 裁剪的最大值，输入中大于该值的元素将由该元素代替，若参数为空，则不对输入的最大值做限制。数据类型可以是 float32 或形状为[1]的 Tensor，类型可以为 int32、float32、float64，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.clip
