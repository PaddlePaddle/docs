.. _cn_api_paddle_ceil:

ceil
-------------------------------

.. py:function:: paddle.ceil(x, name=None)




向上取整运算函数。

.. math::
    out = \left \lceil x \right \rceil


.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，如 ``input=tensor_x`` 等价于 ``x=tensor_x``。

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，数据类型支持 float32, float64, float16, bfloat16, uint8, int8, int16, int32, int64。
      别名： ``input``
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **out** （Tensor，可选） - 指定输出结果的 `Tensor`，默认值为 None。

返回
::::::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.ceil
