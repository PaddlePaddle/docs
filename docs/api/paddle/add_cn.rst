.. _cn_api_paddle_add:

add
-------------------------------

.. py:function:: paddle.add(x, y, name=None, *, alpha=1, out=None)



逐元素相加算子，输入 ``x`` 与输入 ``y`` 逐元素相加，并将各个位置的输出元素保存到返回结果中。

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，参数名 ``other`` 可替代 ``y`` ，如 ``add(input=tensor_x, other=tensor_y, ...)`` 等价于 ``add(x=tensor_x, y=tensor_y, ...)`` 。
    输入 ``x`` 与输入 ``y`` 必须和广播为相同形状，关于广播规则，请参见 `Tensor 介绍`_ .

        .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7


等式为：

.. math::
        Out = X + alpha \times Y

- :math:`X`：多维 Tensor。
- :math:`Y`：多维 Tensor。

以下情况使用该算子，该情况为：
1. ``X`` 与 ``Y`` 的形状一样。
2. ``Y`` 的形状是 ``X`` 的一部分连续的形状。


参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：bool、bfloat16、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
      ``别名: input``
    - **y** (Tensor) - 输入的 Tensor，数据类型为：bool、bfloat16、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
      ``别名: other``
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **alpha** (Number, 可选) - 对 ``y`` 的缩放因子，默认值为 1。
    - **out** (Tensor, 可选) - 输出 Tensor，默认值为 None。

返回
:::::::::
多维 Tensor，数据类型与 ``x`` 相同，维度为广播后的形状。


代码示例
:::::::::

COPY-FROM: paddle.add
