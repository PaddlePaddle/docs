.. _cn_api_paddle_remainder:

remainder
-------------------------------

.. py:function:: paddle.remainder(x, y, name=None, *, out=None)


逐元素取模算子。公式为：

.. math::
        \\out = x \% y\\

.. note::
     ``paddle.remainder``  支持广播，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

.. note::
    别名支持: 参数名  ``input``  可替代  ``x`` ，  ``other``  可替代  ``y`` ;

参数
:::::::::

  - **x** (Tensor) - 多维 Tensor。数据类型为 bfloat16 、float16 、float32 、float64、int32 或 int64。
     ``别名：input`` 
  - **y** (Tensor) - 多维 Tensor。数据类型为 bfloat16 、float16 、float32 、float64、int32 或 int64。
     ``别名：other`` 
  - **name** (str，可选)  - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
  - **out** (Tensor，可选) - 输出的结果。该参数为仅关键字参数，默认值为 None。

返回
:::::::::
 ``Tensor`` ，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。

代码示例
:::::::::

COPY-FROM: paddle.remainder
