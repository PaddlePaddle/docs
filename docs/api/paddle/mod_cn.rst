.. _cn_api_tensor_mod:

mod
-------------------------------

.. py:function:: paddle.mod(x, y, name=None)

逐元素取模算子。公式为：

.. math::
        out = x \% y

.. note::
    ``paddle.mod`` 支持广播。关于广播规则，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::
        - **x** (Tensor) - 多维 Tensor。数据类型为 float16、float32 、float64、int32 或 int64。
        - **y** (Tensor) - 多维 Tensor。数据类型为 float16、float32 、float64、int32 或 int64。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。

代码示例
:::::::::

COPY-FROM: paddle.mod
