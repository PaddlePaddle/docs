.. _cn_api_paddle_subtract:

subtract
-------------------------------

.. py:function:: paddle.subtract(x, y, name=None, *, alpha=1, out=None)


.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，参数名 ``other`` 可替代 ``y`` ，如 ``subtract(input=tensor_x, other=tensor_y, ...)`` 等价于 ``subtract(x=tensor_x, y=tensor_y, ...)``。

逐元素相减算子，输入 ``x`` 与输入 ``y`` 逐元素相减，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
        out = x - alpha \times y

.. note::
   ``paddle.subtract`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

   .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::
   - **x** (Tensor) - 输入的 Tensor。数据类型为 ``bfloat16`` 、 ``float16`` 、 ``float32`` 、 ``float64``、 ``int16`` 、 ``int32`` 、 ``int64`` 、 ``complex64`` 或  ``complex128`` 。
     ``别名: input``
   - **y** (Tensor) - 输入的 Tensor。数据类型为 ``bfloat16`` 、 ``float16`` 、 ``float32`` 、 ``float64``、 ``int16`` 、 ``int32`` 、 ``int64`` 、 ``complex64`` 或  ``complex128`` 。
     ``别名: other``
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
   - **alpha** (Number, 可选) - 对 ``y`` 的缩放因子，默认值为 1。
   - **out** (Tensor, 可选) - 输出 Tensor，默认值为 None。

返回
:::::::::
   ``Tensor``，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。


代码示例
::::::::::

COPY-FROM: paddle.subtract
