.. _cn_api_paddle_fmax:

fmax
-------------------------------

.. py:function:: paddle.fmax(x, y, name=None, *, out=None)


比较两个 Tensor 对应位置的元素，返回一个包含该元素最大值的新 Tensor。如果两个元素其中一个是 nan 值，则直接返回另一个值，如果两者都是 nan 值，则返回第一个 nan 值。

等式是：

.. math::
        out = fmax(x, y)

.. note::
   ``paddle.fmax`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

   .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::
   - **x** （Tensor）- 输入的 Tensor。数据类型为 ``bfloat16`` 、 ``float16`` 、 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。别名 ``input`` 。
   - **y** （Tensor）- 输入的 Tensor。数据类型为 ``bfloat16`` 、 ``float16`` 、 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。别名 ``other`` 。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::
   - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
   ``Tensor``，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。


代码示例
::::::::::

COPY-FROM: paddle.fmax
