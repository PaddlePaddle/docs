.. _cn_api_fluid_layers_logical_xor:

logical_xor
-------------------------------

.. py:function:: paddle.logical_xor(x, y, out=None, name=None)

逐元素的对 ``X`` 和 ``Y`` 进行逻辑异或运算。

.. math::
        Out = (X || Y) \&\& !(X \&\& Y)

.. note::
    ``paddle.logical_xor`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
::::::::::::

        - **x** （Tensor）- 输入的 `Tensor`，支持的数据类型为 bool, int8, int16, in32, in64, float32, float64。
        - **y** （Tensor）- 输入的 `Tensor`，支持的数据类型为 bool, int8, int16, in32, in64, float32, float64。
        - **out** （Tensor，可选）- 指定算子输出结果的 `Tensor`，可以是程序中已经创建的任何 Tensor。默认值为 None，此时将创建新的 Tensor 来保存输出结果。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 ``Tensor``，维度``x`` 维度相同，存储运算后的结果。

代码示例
::::::::::::

COPY-FROM: paddle.logical_xor
