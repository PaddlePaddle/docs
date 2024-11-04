.. _cn_api_paddle_logical_not:

logical_not
-------------------------------

.. py:function:: paddle.logical_not(x, out=None, name=None)




逐元素的对 ``X``  Tensor 进行逻辑非运算

.. math::
        Out = !X

参数
::::::::::::

        - **x** （Tensor）- 逻辑非运算的输入，是一个 Tensor，支持的数据类型为 bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128。
        - **out** （Tensor，可选）- 指定算子输出结果的 Tensor，可以是程序中已经创建的任何 Tensor。默认值为 None，此时将创建新的 Tensor 来保存输出结果。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，与 ``x`` 维度相同，数据类型相同。


代码示例
::::::::::::

COPY-FROM: paddle.logical_not
