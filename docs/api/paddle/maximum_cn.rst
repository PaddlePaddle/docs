.. _cn_api_paddle_maximum:

maximum
-------------------------------

.. py:function:: paddle.maximum(x, y, name=None)


逐元素对比输入的两个 Tensor，并且把各个位置更大的元素保存到返回结果中。

等式是：

.. math::
        out = max(x, y)

.. note::
   ``paddle.maximum`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

   .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::
   - **x** （Tensor）- 输入的 Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **y** （Tensor）- 输入的 Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
   ``Tensor``，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。


代码示例
::::::::::

COPY-FROM: paddle.maximum
